"""Convert dated capacity events into weekly retained-capacity fractions.

``capacity_events.csv`` (Annex B) has one row per event::

    event_id, plant, from_date, to_date, capacity_loss_pct

The solver plans a single week ``[planning_date, planning_date + 6]`` and
takes ``capacity_overrides = {plant: retained_fraction}``. This module is
the one place that turns dated events into those fractions.

Rules
-----
* ``from_date`` and ``to_date`` are inclusive calendar dates.
* ``capacity_loss_pct`` is a percentage of the plant's *original* weekly
  capacity (30 means 30% lost). Values are never reinterpreted as fractions;
  if every value is <= 1 a warning is returned because that often signals a
  unit mistake.
* Capacity is assumed to be spread evenly over the 7 days of the week, so an
  event is pro-rated by the days it overlaps the planning week. On each day
  the losses of all active events for a plant are added (each is a share of
  original capacity) and capped at 100%. The retained fraction is the mean
  daily retained share::

      retained = mean over days d of (1 - min(100, sum active loss_pct on d) / 100)

  A 30% loss covering the whole week therefore gives ``0.70``; covering 2 of
  the 7 days gives ``1 - 0.30 * 2/7``.
* Events that do not overlap the planning week are ignored and reported.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from app.solver.cost_carbon import (
    CAPACITY_DECIMALS,
    DataValidationError,
    parse_planning_date,
    to_json_compatible,
    validate_capacity_overrides,
)

REQUIRED_EVENT_COLUMNS: tuple[str, ...] = ("event_id", "plant", "from_date", "to_date", "capacity_loss_pct")

#: Length of the planning period in days (the solver's weekly capacity).
PLANNING_HORIZON_DAYS: int = 7

CAPACITY_EVENT_ASSUMPTIONS: tuple[str, ...] = (
    "Capacity events use inclusive from/to dates; capacity_loss_pct is a percentage of original weekly capacity.",
    "Event losses are pro-rated by days of overlap with the 7-day planning week (capacity assumed evenly spread).",
    "Concurrent events on the same plant add their losses, capped at 100% per day.",
)


def _validate_events(events_df: pd.DataFrame, known_plants: Optional[Iterable[str]]) -> pd.DataFrame:
    if not isinstance(events_df, pd.DataFrame):
        raise DataValidationError(f"events_df must be a pandas DataFrame, got {type(events_df).__name__}")
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in events_df.columns]
    if missing:
        raise DataValidationError(f"capacity events are missing required columns: {', '.join(missing)}")
    df = events_df.loc[:, list(REQUIRED_EVENT_COLUMNS)].copy()

    for col in REQUIRED_EVENT_COLUMNS:
        nulls = df[col].isna()
        if col in ("event_id", "plant"):
            nulls |= df[col].astype(str).str.strip().eq("")
        if nulls.any():
            raise DataValidationError(f"capacity events column {col} has missing values at rows: {df.index[nulls].tolist()}")
    df["event_id"] = df["event_id"].astype(str).str.strip()
    df["plant"] = df["plant"].astype(str).str.strip()

    dup = df["event_id"].duplicated(keep=False)
    if dup.any():
        raise DataValidationError(f"capacity events have duplicate event_id values: {sorted(set(df.loc[dup, 'event_id']))}")

    for col in ("from_date", "to_date"):
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad = parsed.isna()
        if bad.any():
            raise DataValidationError(f"capacity events column {col} has invalid dates: {df.loc[bad, col].tolist()}")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        df[col] = parsed.dt.normalize()
    reversed_ = df["to_date"] < df["from_date"]
    if reversed_.any():
        raise DataValidationError(f"capacity events end before they start: {df.loc[reversed_, 'event_id'].tolist()}")

    if pd.api.types.is_bool_dtype(df["capacity_loss_pct"]):
        raise DataValidationError("capacity_loss_pct must be numeric, not boolean")
    pct = pd.to_numeric(df["capacity_loss_pct"], errors="coerce")
    bad = pct.isna() | ~np.isfinite(pct.astype(float)) | (pct < 0) | (pct > 100)
    if bad.any():
        raise DataValidationError(
            f"capacity_loss_pct must be a percentage in [0, 100]: {df.loc[bad, ['event_id', 'capacity_loss_pct']].to_dict('records')}"
        )
    df["capacity_loss_pct"] = pct.astype(float)

    if known_plants is not None:
        known = set(known_plants)
        unknown = sorted(set(df["plant"]) - known)
        if unknown:
            raise DataValidationError(f"capacity events refer to unknown plants {unknown}; known plants: {sorted(known)}")
    return df


def capacity_overrides_from_events(
    events_df: pd.DataFrame,
    planning_date: Any,
    known_plants: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Derive solver ``capacity_overrides`` for the planning week from dated events.

    Args:
        events_df: Table with :data:`REQUIRED_EVENT_COLUMNS` (not mutated).
        planning_date: First day of the 7-day planning week.
        known_plants: Optional plant IDs; events for other plants raise.

    Returns:
        JSON-compatible dict with ``capacity_overrides`` ({plant: retained
        fraction}, only plants affected this week), ``planning_window``
        (``start``/``end`` ISO dates), ``applied_events`` (with
        ``overlap_days``), ``ignored_events``, ``warnings`` and ``assumptions``.

    Raises:
        DataValidationError: on missing columns or values, duplicate IDs,
            invalid or reversed dates, percentages outside [0, 100], or
            unknown plants.
    """
    start = parse_planning_date(planning_date)
    days = pd.date_range(start, periods=PLANNING_HORIZON_DAYS, freq="D")
    end = days[-1]
    events = _validate_events(events_df, known_plants)

    applied: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    daily_loss: Dict[str, np.ndarray] = {}
    for ev in events.itertuples(index=False):
        active = (days >= ev.from_date) & (days <= ev.to_date)
        overlap = int(active.sum())
        record = {
            "event_id": ev.event_id,
            "plant": ev.plant,
            "from_date": ev.from_date,
            "to_date": ev.to_date,
            "capacity_loss_pct": ev.capacity_loss_pct,
        }
        if overlap == 0:
            ignored.append({**record, "reason": "does not overlap the planning week"})
            continue
        applied.append({**record, "overlap_days": overlap})
        losses = daily_loss.setdefault(ev.plant, np.zeros(PLANNING_HORIZON_DAYS))
        losses[active] += ev.capacity_loss_pct

    overrides = {
        plant: round(float(np.mean(1.0 - np.minimum(losses, 100.0) / 100.0)), CAPACITY_DECIMALS)
        for plant, losses in sorted(daily_loss.items())
    }

    warnings: List[str] = []
    positive = events.loc[events["capacity_loss_pct"] > 0, "capacity_loss_pct"]
    if not positive.empty and (positive <= 1.0).all():
        warnings.append(
            "All capacity_loss_pct values are <= 1; they are treated as percentages (0.3 = 0.3%), not fractions"
        )
    for plant, losses in daily_loss.items():
        if (losses > 100.0).any():
            warnings.append(f"Concurrent events on plant {plant} exceed 100% loss on some days; capped at 100%")

    return to_json_compatible({
        "capacity_overrides": overrides,
        "planning_window": {"start": start, "end": end},
        "applied_events": applied,
        "ignored_events": ignored,
        "warnings": warnings,
        "assumptions": list(CAPACITY_EVENT_ASSUMPTIONS),
    })


def weekly_retained_fractions(
    events_df: pd.DataFrame,
    planning_date: Any,
    horizon_weeks: int,
    known_plants: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Retained capacity fraction per plant and planning week from dated events.

    Week ``w`` covers ``planning_date + 7*w`` to ``+6`` days. Each week is
    converted with :func:`capacity_overrides_from_events`, so pro-rating and
    concurrent-loss rules are identical. Plants/weeks not listed keep 100%.

    Returns:
        ``{"by_week": {(plant, week): fraction}, "applied_events": [...],
        "ignored_events": [...], "warnings": [...], "assumptions": [...]}``.
        ``by_week`` has tuple keys (internal use); the other values are JSON-compatible.
    """
    start = parse_planning_date(planning_date)
    _validate_events(events_df, known_plants)  # fail fast on bad input
    by_week: Dict[tuple, float] = {}
    applied: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for week in range(horizon_weeks):
        conv = capacity_overrides_from_events(events_df, start + pd.Timedelta(days=7 * week), known_plants)
        for plant, frac in conv["capacity_overrides"].items():
            by_week[(plant, week)] = frac
        for ev in conv["applied_events"]:
            rec = applied.setdefault(ev["event_id"], {**{k: v for k, v in ev.items() if k != "overlap_days"}, "weeks": []})
            rec["weeks"].append({"week": week, "overlap_days": ev["overlap_days"]})
        warnings.extend(w for w in conv["warnings"] if w not in warnings)
    ignored = [
        {"event_id": str(ev), "reason": "does not overlap the planning horizon"}
        for ev in _validate_events(events_df, known_plants)["event_id"] if str(ev) not in applied
    ]
    return {
        "by_week": by_week,
        "applied_events": list(applied.values()),
        "ignored_events": ignored,
        "warnings": warnings,
        "assumptions": list(CAPACITY_EVENT_ASSUMPTIONS),
    }


def effective_weekly_limits(
    limits: Mapping[tuple, int],
    planning_date: Any,
    horizon_weeks: int,
    capacity_overrides: Optional[Mapping[str, Any]] = None,
    capacity_events: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Effective capacity of every (plant, product, week) - the single rule used by solver and validator.

    ``limit[p, q, w] = weekly_capacity[p, q] * override[p] * event_fraction[p, w]``

    ``capacity_overrides`` (``{plant: retained_fraction}``, no dates) apply to
    every week of the horizon. Dated ``capacity_events`` reduce only the weeks
    they overlap (see :func:`weekly_retained_fractions`).

    Returns:
        ``{"limits": {(plant, product, week): capacity}, "events": <weekly_retained_fractions
        result without by_week> or None}``.
    """
    plants = {p for p, _ in limits}
    overrides = validate_capacity_overrides(capacity_overrides, plants)
    events_info = None
    by_week: Dict[tuple, float] = {}
    if capacity_events is not None and not capacity_events.empty:
        conv = weekly_retained_fractions(capacity_events, planning_date, horizon_weeks, known_plants=plants)
        by_week = conv.pop("by_week")
        events_info = to_json_compatible(conv)
    out = {
        (plant, product, week): round(float(cap) * overrides.get(plant, 1.0) * by_week.get((plant, week), 1.0), CAPACITY_DECIMALS)
        for (plant, product), cap in limits.items()
        for week in range(horizon_weeks)
    }
    return {"limits": out, "events": events_info}


def events_from_scenario(scenario: Mapping[str, Any], planning_date: Any) -> pd.DataFrame:
    """Build a one-row events table from an API scenario payload.

    ``scenario`` needs ``plant`` and ``capacity_loss_pct``; ``from_date`` /
    ``to_date`` default to the planning week and ``event_id`` to a
    descriptive ID. Validation happens in :func:`capacity_overrides_from_events`.
    """
    start = parse_planning_date(planning_date)
    plant = scenario.get("plant")
    pct = scenario.get("capacity_loss_pct")
    if plant is None or pct is None:
        raise DataValidationError("scenario must include 'plant' and 'capacity_loss_pct'")
    if isinstance(pct, bool) or not isinstance(pct, (int, float)) or not math.isfinite(float(pct)):
        raise DataValidationError(f"scenario capacity_loss_pct must be a number, got {pct!r}")
    return pd.DataFrame([{
        "event_id": scenario.get("event_id") or f"scenario-{plant}-{pct:g}pct",
        "plant": plant,
        "from_date": scenario.get("from_date") or start,
        "to_date": scenario.get("to_date") or start + pd.Timedelta(days=PLANNING_HORIZON_DAYS - 1),
        "capacity_loss_pct": pct,
    }])
