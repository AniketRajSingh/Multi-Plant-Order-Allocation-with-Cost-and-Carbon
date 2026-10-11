"""Input validation and per-candidate cost / carbon / delivery metrics.

This module is the single source of truth for the formulas shared by the MILP
solver (``app.solver.milp``), the independent validator
(``app.solver.validator``) and the trade-off generator
(``app.solver.tradeoff``).

Cost model
----------
Landed cost follows the formula specified for this project exactly::

    unit_landed_cost = (unit_cost + transport_cost_per_unit) * (1 + duty_pct / 100)
    total_cost       = qty * unit_landed_cost

**Known limitation - energy monetary cost is NOT included.** The authoritative
``plants.csv`` schema (Annex B) supplies ``energy_kwh_per_unit`` and
``carbon_kg_per_kwh`` but no energy price per kWh. No tariff is invented and
the missing component is not silently treated as zero-valued input: it is
simply not modelled, and this is reported in :data:`COST_MODEL_ASSUMPTIONS`
(also attached to the returned DataFrame as ``attrs["assumptions"]``).
:func:`compute_unit_landed_cost` accepts an optional ``energy_cost_per_unit``
so an energy component can be added once an authoritative price input exists.

Carbon model
------------
::

    energy_carbon_kg_per_unit = energy_kwh_per_unit * carbon_kg_per_kwh
    unit_carbon_kg            = energy_carbon_kg_per_unit + carbon_kg_per_unit
    total_carbon_kg           = qty * unit_carbon_kg

All carbon values are kilograms; tonnes are only produced for reporting
(``kg / 1000``). Import duty is never applied to carbon.

Delivery model
--------------
``arrival_date = planning_date + transit_days`` (calendar days, production
lead time not modelled) and ``is_late = arrival_date > due``.

Capacity model
--------------
``weekly_capacity`` is a plant-wide figure repeated on each plant-product row.
Repeated values must agree (otherwise :class:`DataValidationError`).
``capacity_overrides`` map a plant to the *fraction of capacity retained*
(``{"B": 0.70}`` = Plant B keeps 70%, i.e. loses 30%).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# ============================================================================
# SCHEMA CONSTANTS
# ============================================================================

REQUIRED_ORDER_COLUMNS: tuple[str, ...] = ("order_id", "product", "qty", "destination", "due")
REQUIRED_PLANT_COLUMNS: tuple[str, ...] = (
    "plant",
    "product",
    "weekly_capacity",
    "unit_cost",
    "energy_kwh_per_unit",
    "carbon_kg_per_kwh",
)
REQUIRED_TARIFF_COLUMNS: tuple[str, ...] = (
    "plant",
    "destination",
    "transport_cost_per_unit",
    "transit_days",
    "duty_pct",
    "carbon_kg_per_unit",
)

#: Column order of the DataFrame returned by :func:`calculate_candidate_metrics`.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "order_id",
    "plant",
    "product",
    "qty",
    "destination",
    "due",
    "eligible",
    "weekly_capacity",
    "unit_cost",
    "transport_cost_per_unit",
    "duty_pct",
    "unit_landed_cost",
    "total_cost",
    "energy_kwh_per_unit",
    "carbon_kg_per_kwh",
    "energy_carbon_kg_per_unit",
    "carbon_kg_per_unit",
    "unit_carbon_kg",
    "total_carbon_kg",
    "transit_days",
    "arrival_date",
    "is_late",
)

KG_PER_TONNE: float = 1000.0

#: Effective capacities are rounded to this many decimals to avoid spurious
#: infeasibility from floating-point products such as ``1100 * 0.7``.
CAPACITY_DECIMALS: int = 9

COST_MODEL_ASSUMPTIONS: tuple[str, ...] = (
    "Landed cost = (unit_cost + transport_cost_per_unit) * (1 + duty_pct/100); duty applies to production plus transport.",
    "Energy monetary cost is NOT included: plants.csv supplies no energy price per kWh, so no tariff is assumed.",
    "Carbon = energy_kwh_per_unit * carbon_kg_per_kwh + transport carbon_kg_per_unit, in kg; duty is not applied to carbon.",
    "Arrival date = planning_date + transit_days (calendar days); production lead time is not modelled.",
    "weekly_capacity is plant-wide (shared by all products of a plant) for a single one-week planning period.",
    "Carbon intensity is treated as constant over the planning period (no week-specific factors in the schema).",
)


class DataValidationError(ValueError):
    """Raised when solver input data violates the required schema or domain rules."""


# ============================================================================
# GENERIC HELPERS
# ============================================================================

_MAX_LISTED = 10


def _fmt_items(items: Sequence[Any]) -> str:
    """Format up to ``_MAX_LISTED`` items for an error message."""
    items = list(items)
    shown = ", ".join(str(i) for i in items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        shown += f", ... ({len(items)} total)"
    return shown


def to_json_compatible(value: Any) -> Any:
    """Recursively convert pandas/NumPy/date values into JSON-serialisable builtins.

    Dates and timestamps become ISO ``YYYY-MM-DD`` strings (time components
    are dropped because the model works in whole days), NaN / NA / NaT become
    ``None``, NumPy scalars become Python scalars.
    """
    if isinstance(value, Mapping):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_json_compatible(v) for v in value.tolist()]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    return value


def parse_planning_date(planning_date: Any) -> pd.Timestamp:
    """Parse the planning start date into a normalised (midnight) ``Timestamp``.

    Accepts ``datetime.date``, ``datetime.datetime``, ``pandas.Timestamp`` or
    an ISO date string.

    Raises:
        DataValidationError: if the value is missing or not a valid date.
    """
    if planning_date is None or isinstance(planning_date, (bool, int, float, np.number)):
        raise DataValidationError(f"planning_date must be a date or ISO date string, got {planning_date!r}")
    try:
        ts = pd.Timestamp(planning_date)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"planning_date {planning_date!r} is not a valid date") from exc
    if pd.isna(ts):
        raise DataValidationError("planning_date must not be missing")
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _require_dataframe(df: Any, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise DataValidationError(f"{name} is empty")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {_fmt_items(missing)}")


def _clean_strings(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Validate non-null identifier columns and strip surrounding whitespace (in place on a copy)."""
    for col in columns:
        raw = df[col]
        null_mask = raw.isna()
        stripped = raw.astype(str).str.strip()
        null_mask |= stripped.eq("")
        if null_mask.any():
            rows = df.index[null_mask].tolist()
            raise DataValidationError(f"{name}.{col} has missing values at rows: {_fmt_items(rows)}")
        df[col] = stripped


def _clean_numeric(
    df: pd.DataFrame,
    col: str,
    name: str,
    *,
    integer: bool = False,
    min_value: float = 0.0,
    strictly_positive: bool = False,
) -> None:
    """Validate a numeric column (non-null, numeric, finite, bounded) and coerce its dtype."""
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        raise DataValidationError(f"{name}.{col} must be numeric, not boolean")
    null_mask = raw.isna()
    if null_mask.any():
        raise DataValidationError(
            f"{name}.{col} has missing values at rows: {_fmt_items(df.index[null_mask].tolist())}"
        )
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = numeric.isna() | ~np.isfinite(numeric.astype(float))
    if bad.any():
        bad_vals = [f"row {i}={raw.loc[i]!r}" for i in df.index[bad]]
        raise DataValidationError(f"{name}.{col} has non-numeric or non-finite values: {_fmt_items(bad_vals)}")
    numeric = numeric.astype(float)
    if integer:
        non_int = numeric != np.round(numeric)
        if non_int.any():
            bad_vals = [f"row {i}={raw.loc[i]!r}" for i in df.index[non_int]]
            raise DataValidationError(f"{name}.{col} must contain whole numbers: {_fmt_items(bad_vals)}")
    if strictly_positive:
        bad_mask = numeric <= 0
        rule = "> 0"
    else:
        bad_mask = numeric < min_value
        rule = f">= {min_value:g}"
    if bad_mask.any():
        bad_vals = [f"row {i}={raw.loc[i]!r}" for i in df.index[bad_mask]]
        raise DataValidationError(f"{name}.{col} must be {rule}: {_fmt_items(bad_vals)}")
    df[col] = numeric.astype("int64") if integer else numeric


def _clean_dates(df: pd.DataFrame, col: str, name: str) -> None:
    raw = df[col]
    null_mask = raw.isna()
    if null_mask.any():
        raise DataValidationError(
            f"{name}.{col} has missing values at rows: {_fmt_items(df.index[null_mask].tolist())}"
        )
    parsed = pd.to_datetime(raw, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        bad_vals = [f"row {i}={raw.loc[i]!r}" for i in df.index[bad]]
        raise DataValidationError(f"{name}.{col} has invalid dates: {_fmt_items(bad_vals)}")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    df[col] = parsed.dt.normalize()


def _reject_duplicates(df: pd.DataFrame, keys: List[str], name: str) -> None:
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        combos = sorted({" / ".join(str(v) for v in row) for row in df.loc[dup, keys].itertuples(index=False)})
        raise DataValidationError(f"{name} has duplicate {'-'.join(keys)} records: {_fmt_items(combos)}")


# ============================================================================
# TABLE PREPARATION (validated copies; caller DataFrames are never mutated)
# ============================================================================

def prepare_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of the orders table restricted to the required columns."""
    _require_dataframe(orders_df, "orders_df")
    _require_columns(orders_df, REQUIRED_ORDER_COLUMNS, "orders_df")
    df = orders_df.loc[:, list(REQUIRED_ORDER_COLUMNS)].copy()
    _clean_strings(df, ["order_id", "product", "destination"], "orders_df")
    _clean_numeric(df, "qty", "orders_df", integer=True, strictly_positive=True)
    _clean_dates(df, "due", "orders_df")
    _reject_duplicates(df, ["order_id"], "orders_df")
    return df.reset_index(drop=True)


def prepare_plants(plants_df: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of the plants table restricted to the required columns."""
    _require_dataframe(plants_df, "plants_df")
    _require_columns(plants_df, REQUIRED_PLANT_COLUMNS, "plants_df")
    df = plants_df.loc[:, list(REQUIRED_PLANT_COLUMNS)].copy()
    _clean_strings(df, ["plant", "product"], "plants_df")
    _clean_numeric(df, "weekly_capacity", "plants_df", integer=True, strictly_positive=True)
    for col in ("unit_cost", "energy_kwh_per_unit", "carbon_kg_per_kwh"):
        _clean_numeric(df, col, "plants_df")
    _reject_duplicates(df, ["plant", "product"], "plants_df")
    plant_capacities(df)  # raises on inconsistent repeated capacity values
    return df.reset_index(drop=True)


def prepare_tariffs(tariffs_df: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of the transport/duty table restricted to the required columns."""
    _require_dataframe(tariffs_df, "tariffs_df")
    _require_columns(tariffs_df, REQUIRED_TARIFF_COLUMNS, "tariffs_df")
    df = tariffs_df.loc[:, list(REQUIRED_TARIFF_COLUMNS)].copy()
    _clean_strings(df, ["plant", "destination"], "tariffs_df")
    for col in ("transport_cost_per_unit", "duty_pct", "carbon_kg_per_unit"):
        _clean_numeric(df, col, "tariffs_df")
    _clean_numeric(df, "transit_days", "tariffs_df", integer=True)
    _reject_duplicates(df, ["plant", "destination"], "tariffs_df")
    return df.reset_index(drop=True)


# ============================================================================
# CAPACITY
# ============================================================================

def plant_capacities(plants_df: pd.DataFrame) -> Dict[str, int]:
    """Return one plant-wide weekly capacity per plant.

    ``plants_df`` has one row per plant-product, each repeating the plant's
    capacity. Capacities are *not* summed across rows; conflicting values for
    the same plant raise :class:`DataValidationError`.
    """
    caps: Dict[str, int] = {}
    conflicts: List[str] = []
    for plant, values in plants_df.groupby("plant", sort=True)["weekly_capacity"]:
        unique = sorted({int(v) for v in values})
        if len(unique) > 1:
            conflicts.append(f"{plant}: {unique}")
        caps[str(plant)] = unique[0]
    if conflicts:
        raise DataValidationError(f"plants_df has inconsistent weekly_capacity for the same plant: {_fmt_items(conflicts)}")
    return caps


def validate_capacity_overrides(
    capacity_overrides: Optional[Mapping[str, Any]],
    known_plants: Iterable[str],
) -> Dict[str, float]:
    """Validate ``{plant: retained_fraction}`` overrides.

    A value of ``0.70`` means the plant retains 70% of its capacity (a 30%
    loss). Fractions must be real numbers in ``[0, 1]`` and plants must exist.

    Raises:
        DataValidationError: on unknown plants or invalid fractions.
    """
    if capacity_overrides is None:
        return {}
    if not isinstance(capacity_overrides, Mapping):
        raise DataValidationError(
            f"capacity_overrides must be a mapping of plant -> retained fraction, got {type(capacity_overrides).__name__}"
        )
    known = set(known_plants)
    clean: Dict[str, float] = {}
    for plant, frac in capacity_overrides.items():
        key = str(plant).strip()
        if key not in known:
            raise DataValidationError(f"capacity_overrides refers to unknown plant {plant!r}; known plants: {sorted(known)}")
        if isinstance(frac, (bool, np.bool_)) or not isinstance(frac, (int, float, np.integer, np.floating)):
            raise DataValidationError(f"capacity_overrides[{plant!r}] must be a number in [0, 1], got {frac!r}")
        value = float(frac)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise DataValidationError(
                f"capacity_overrides[{plant!r}]={frac!r} must be a retained fraction in [0, 1] "
                "(e.g. 0.70 for a 30% capacity loss)"
            )
        if key in clean:
            raise DataValidationError(f"capacity_overrides lists plant {key!r} more than once")
        clean[key] = value
    return clean


def effective_capacities(
    capacities: Mapping[str, int],
    capacity_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Apply validated retained-fraction overrides to plant-wide capacities."""
    overrides = validate_capacity_overrides(capacity_overrides, capacities.keys())
    return {
        plant: round(float(cap) * overrides.get(plant, 1.0), CAPACITY_DECIMALS)
        for plant, cap in capacities.items()
    }


# ============================================================================
# FORMULAS
# ============================================================================

def compute_unit_landed_cost(
    unit_cost: Any,
    transport_cost_per_unit: Any,
    duty_pct: Any,
    energy_cost_per_unit: Any = None,
) -> Any:
    """Landed cost per unit: ``(unit_cost + transport [+ energy]) * (1 + duty_pct/100)``.

    ``energy_cost_per_unit`` is an extension point for when an authoritative
    energy price exists. It defaults to ``None`` meaning "not modelled" (the
    current project formula); it is *not* a zero-valued placeholder for data.
    Works element-wise on scalars, NumPy arrays and pandas Series.
    """
    pre_duty = unit_cost + transport_cost_per_unit
    if energy_cost_per_unit is not None:
        pre_duty = pre_duty + energy_cost_per_unit
    return pre_duty * (1.0 + duty_pct / 100.0)


def compute_energy_carbon_per_unit(energy_kwh_per_unit: Any, carbon_kg_per_kwh: Any) -> Any:
    """Energy-related production carbon per unit, in kg."""
    return energy_kwh_per_unit * carbon_kg_per_kwh


def compute_unit_carbon(energy_carbon_kg_per_unit: Any, transport_carbon_kg_per_unit: Any) -> Any:
    """Total carbon per unit (energy + transport), in kg. Duty is never applied."""
    return energy_carbon_kg_per_unit + transport_carbon_kg_per_unit


def kg_to_tonnes(kg: Optional[float]) -> Optional[float]:
    """Convert kilograms to tonnes for reporting only."""
    return None if kg is None else kg / KG_PER_TONNE


# ============================================================================
# CANDIDATE METRICS
# ============================================================================

def calculate_candidate_metrics(
    orders_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    planning_date: Any,
) -> pd.DataFrame:
    """Compute cost, carbon and delivery metrics for every order x plant candidate.

    One row is returned per (order, plant) for every plant listed in
    ``plants_df``. ``eligible`` is ``True`` only when the plant-product pair
    exists in ``plants_df``; for ineligible rows the product-dependent
    metrics (costs, carbon) are NaN and must not be used.

    Args:
        orders_df: Orders table (Annex B ``orders.csv`` schema).
        plants_df: Plant-product table (``plants.csv`` schema).
        tariffs_df: Transport and duty table (``transport_duties.csv`` schema).
        planning_date: Start of the planning week; arrival dates count from it.

    Returns:
        DataFrame with :data:`CANDIDATE_COLUMNS`. ``due`` and ``arrival_date``
        are normalised ``datetime64`` values; ``is_late`` is a nullable boolean
        (NA when no transport record exists for an ineligible candidate).
        ``attrs["assumptions"]`` lists the cost-model assumptions and
        ``attrs["warnings"]`` lists non-fatal data observations.

    Raises:
        DataValidationError: for missing columns/values, invalid numbers or
            dates, duplicate keys, inconsistent capacities, products that no
            plant can make, or missing transport rows for eligible
            plant-destination combinations.
    """
    planning = parse_planning_date(planning_date)
    orders = prepare_orders(orders_df)
    plants = prepare_plants(plants_df)
    tariffs = prepare_tariffs(tariffs_df)
    capacities = plant_capacities(plants)

    known_products = set(plants["product"])
    no_plant = orders.loc[~orders["product"].isin(known_products)]
    if not no_plant.empty:
        detail = [f"{r.order_id} ({r.product})" for r in no_plant.itertuples(index=False)]
        raise DataValidationError(f"Orders whose product no plant can manufacture: {_fmt_items(detail)}")

    eligible_pairs = orders.merge(plants[["plant", "product"]], on="product", how="inner")
    required_routes = eligible_pairs[["plant", "destination"]].drop_duplicates()
    routes = required_routes.merge(tariffs[["plant", "destination"]], on=["plant", "destination"], how="left", indicator=True)
    missing = routes.loc[routes["_merge"] == "left_only"]
    if not missing.empty:
        combos = sorted(f"{r.plant} -> {r.destination}" for r in missing.itertuples(index=False))
        raise DataValidationError(f"tariffs_df is missing transport records for required plant-destination pairs: {_fmt_items(combos)}")

    plant_ids = pd.DataFrame({"plant": sorted(capacities)})
    grid = orders.assign(_order_pos=np.arange(len(orders))).merge(plant_ids, how="cross")
    plant_metrics = plants.drop(columns=["weekly_capacity"]).assign(eligible=True)
    grid = grid.merge(plant_metrics, on=["plant", "product"], how="left")
    grid["eligible"] = grid["eligible"].astype("boolean").fillna(False).astype(bool)
    grid = grid.merge(tariffs, on=["plant", "destination"], how="left")
    grid["weekly_capacity"] = grid["plant"].map(capacities).astype("int64")

    grid["unit_landed_cost"] = compute_unit_landed_cost(
        grid["unit_cost"], grid["transport_cost_per_unit"], grid["duty_pct"]
    )
    grid["total_cost"] = grid["qty"] * grid["unit_landed_cost"]
    grid["energy_carbon_kg_per_unit"] = compute_energy_carbon_per_unit(
        grid["energy_kwh_per_unit"], grid["carbon_kg_per_kwh"]
    )
    grid["unit_carbon_kg"] = compute_unit_carbon(grid["energy_carbon_kg_per_unit"], grid["carbon_kg_per_unit"])
    grid["total_carbon_kg"] = grid["qty"] * grid["unit_carbon_kg"]

    grid["transit_days"] = grid["transit_days"].astype("Int64")
    grid["arrival_date"] = planning + pd.to_timedelta(grid["transit_days"].astype("float64"), unit="D")
    is_late = (grid["arrival_date"] > grid["due"]).astype("boolean")
    is_late[grid["arrival_date"].isna()] = pd.NA
    grid["is_late"] = is_late

    result = (
        grid.sort_values(["_order_pos", "plant"], kind="stable")
        .loc[:, list(CANDIDATE_COLUMNS)]
        .reset_index(drop=True)
    )

    warnings: List[str] = []
    overdue = orders.loc[orders["due"] < planning, "order_id"].tolist()
    if overdue:
        warnings.append(f"Orders due before the planning date (late on every plant): {_fmt_items(overdue)}")
    ignored = [c for c in plants_df.columns if "price" in str(c).lower()]
    if ignored:
        warnings.append(f"plants_df columns ignored by the specified cost formula: {_fmt_items(ignored)}")

    result.attrs["assumptions"] = list(COST_MODEL_ASSUMPTIONS)
    result.attrs["warnings"] = warnings
    result.attrs["planning_date"] = planning.date().isoformat()
    result.attrs["energy_cost_included"] = False
    return result
