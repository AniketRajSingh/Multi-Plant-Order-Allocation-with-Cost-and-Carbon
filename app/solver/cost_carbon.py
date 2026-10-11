"""Input validation and per-candidate cost / carbon / delivery metrics.

This module is the single source of truth for the formulas shared by the MILP
solver (``app.solver.milp``), the independent validator
(``app.solver.validator``) and the trade-off generator
(``app.solver.tradeoff``).

Cost model
----------
Landed cost covers production, energy, transport and import duty (R2)::

    energy_cost_per_unit = energy_kwh_per_unit * energy_price_per_kwh
    unit_landed_cost     = (unit_cost + energy_cost_per_unit + transport_cost_per_unit)
                           * (1 + duty_pct / 100)
    total_cost           = qty * unit_landed_cost

``energy_price_per_kwh`` is not in the Annex B column list, but the guide
(Section 4.1) says plants.csv holds an energy price and extra columns are
allowed. When the column is present it is validated (no missing or negative
values) and energy cost is included. When it is absent, no tariff is invented:
landed cost is ``(unit_cost + transport) * (1 + duty_pct/100)`` and the result's
assumptions say energy cost is not included.

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
Planning is multi-week: week ``w`` starts on ``planning_date + 7*w``. An order
produced in week ``w`` ships at the start of that week, so
``arrival_date = week_start + transit_days`` and ``is_late = arrival_date > due``
(production lead time within the week is not modelled). For week 0 this is
``planning_date + transit_days``. :func:`calculate_candidate_metrics` reports
week 0; :func:`expand_to_weeks` produces one row per eligible order x plant x
week.

Weekly carbon intensity
-----------------------
Section 4.4 expects carbon intensity that changes by week. If a weekly table
(``plant, week_start, carbon_kg_per_kwh``) is supplied, the energy carbon of a
production week uses the row whose week covers that week's start. Otherwise,
or for weeks missing from the table, ``plants.csv`` ``carbon_kg_per_kwh`` is
used and a warning is returned.

Capacity model
--------------
``weekly_capacity`` is a limit per plant-product row **per week**: all orders
of product P produced at plant X in the same week share the
``weekly_capacity`` of row (X, P). Different
products of the same plant may have different limits (team decision,
2026-10-11). A plant's total capacity (sum of its rows) is used for
plant-level utilisation reporting only.
``capacity_overrides`` map a plant to the *fraction of capacity retained*
(``{"B": 0.70}`` = Plant B keeps 70%, i.e. loses 30%), applied to every
product row of that plant.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
#: Optional plants.csv column (Section 4.1 "energy price"); included in landed cost when present.
ENERGY_PRICE_COLUMN = "energy_price_per_kwh"

REQUIRED_CARBON_WEEKLY_COLUMNS: tuple[str, ...] = ("plant", "week_start", "carbon_kg_per_kwh")

#: Upper bound on the planning horizon, to keep the model size bounded.
MAX_HORIZON_WEEKS = 52

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
    "energy_price_per_kwh",
    "energy_cost_per_unit",
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

#: Assumptions that hold for every run; energy-cost and carbon-intensity
#: statements depend on the inputs and are added by :func:`cost_model_assumptions`.
COST_MODEL_ASSUMPTIONS: tuple[str, ...] = (
    "Carbon = energy_kwh_per_unit * carbon_kg_per_kwh + transport carbon_kg_per_unit, in kg; duty is not applied to carbon.",
    "Multi-week plan: week w starts at planning_date + 7*w; an order ships at the start of its production week, so arrival = week start + transit_days.",
    "weekly_capacity is a per plant-product limit for each week; plant utilisation = plant load / sum of its product limits over the horizon.",
)


def cost_model_assumptions(energy_cost_included: bool, weekly_carbon: bool) -> List[str]:
    """Assumption statements for one run, depending on which optional inputs were present."""
    out = list(COST_MODEL_ASSUMPTIONS)
    if energy_cost_included:
        out.insert(0, "Landed cost = (unit_cost + energy_kwh_per_unit*energy_price_per_kwh + transport_cost_per_unit) * (1 + duty_pct/100).")
    else:
        out.insert(0, "Landed cost = (unit_cost + transport_cost_per_unit) * (1 + duty_pct/100); Energy monetary cost is NOT included: plants.csv has no energy_price_per_kwh column.")
    out.append(
        "Carbon intensity per plant and week from the weekly carbon table (plants.csv value where a week is missing)."
        if weekly_carbon else
        "Carbon intensity from plants.csv, constant over the horizon (no weekly carbon table supplied)."
    )
    return out


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
    """Return a validated copy of the plants table: required columns plus ``energy_price_per_kwh`` if present."""
    _require_dataframe(plants_df, "plants_df")
    _require_columns(plants_df, REQUIRED_PLANT_COLUMNS, "plants_df")
    cols = list(REQUIRED_PLANT_COLUMNS) + ([ENERGY_PRICE_COLUMN] if ENERGY_PRICE_COLUMN in plants_df.columns else [])
    df = plants_df.loc[:, cols].copy()
    _clean_strings(df, ["plant", "product"], "plants_df")
    _clean_numeric(df, "weekly_capacity", "plants_df", integer=True, strictly_positive=True)
    for col in cols[3:]:
        _clean_numeric(df, col, "plants_df")
    _reject_duplicates(df, ["plant", "product"], "plants_df")
    return df.reset_index(drop=True)


def prepare_carbon_weekly(carbon_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Validated copy of a weekly carbon-intensity table (``plant, week_start, carbon_kg_per_kwh``)."""
    _require_dataframe(carbon_weekly_df, "carbon_weekly_df")
    _require_columns(carbon_weekly_df, REQUIRED_CARBON_WEEKLY_COLUMNS, "carbon_weekly_df")
    df = carbon_weekly_df.loc[:, list(REQUIRED_CARBON_WEEKLY_COLUMNS)].copy()
    _clean_strings(df, ["plant"], "carbon_weekly_df")
    _clean_dates(df, "week_start", "carbon_weekly_df")
    _clean_numeric(df, "carbon_kg_per_kwh", "carbon_weekly_df")
    _reject_duplicates(df, ["plant", "week_start"], "carbon_weekly_df")
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

def capacity_limits(plants_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Return the weekly capacity limit of every ``(plant, product)`` row.

    These are the binding capacity constraints: orders of product P at
    plant X together may not exceed limit ``(X, P)``.
    """
    return {
        (str(r.plant), str(r.product)): int(r.weekly_capacity)
        for r in plants_df.sort_values(["plant", "product"]).itertuples(index=False)
    }


def plant_capacities(plants_df: pd.DataFrame) -> Dict[str, int]:
    """Return each plant's total weekly capacity (sum of its product limits).

    Used for plant-level utilisation reporting only; the binding constraints
    are the per plant-product limits from :func:`capacity_limits`.
    """
    totals: Dict[str, int] = {}
    for (plant, _product), cap in capacity_limits(plants_df).items():
        totals[plant] = totals.get(plant, 0) + cap
    return totals


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
    """Apply validated retained-fraction overrides to plant capacity totals."""
    overrides = validate_capacity_overrides(capacity_overrides, capacities.keys())
    return {
        plant: round(float(cap) * overrides.get(plant, 1.0), CAPACITY_DECIMALS)
        for plant, cap in capacities.items()
    }


def effective_capacity_limits(
    limits: Mapping[Tuple[str, str], int],
    capacity_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[Tuple[str, str], float]:
    """Apply plant-level retained-fraction overrides to every plant-product limit."""
    overrides = validate_capacity_overrides(capacity_overrides, {plant for plant, _ in limits})
    return {
        key: round(float(cap) * overrides.get(key[0], 1.0), CAPACITY_DECIMALS)
        for key, cap in limits.items()
    }


def nest_by_plant(values: Mapping[Tuple[str, str], Any]) -> Dict[str, Dict[str, Any]]:
    """Turn ``{(plant, product): v}`` into JSON-friendly ``{plant: {product: v}}``."""
    nested: Dict[str, Dict[str, Any]] = {}
    for (plant, product), v in values.items():
        nested.setdefault(plant, {})[product] = v
    return nested


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
    limits = capacity_limits(plants)
    plant_list = sorted({plant for plant, _ in limits})

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

    plant_ids = pd.DataFrame({"plant": plant_list})
    grid = orders.assign(_order_pos=np.arange(len(orders))).merge(plant_ids, how="cross")
    plant_metrics = plants.assign(eligible=True)
    grid = grid.merge(plant_metrics, on=["plant", "product"], how="left")
    grid["eligible"] = grid["eligible"].astype("boolean").fillna(False).astype(bool)
    grid = grid.merge(tariffs, on=["plant", "destination"], how="left")
    grid["weekly_capacity"] = grid["weekly_capacity"].astype("Int64")  # NA for ineligible rows

    energy_included = ENERGY_PRICE_COLUMN in plants.columns
    if energy_included:
        grid["energy_cost_per_unit"] = grid["energy_kwh_per_unit"] * grid[ENERGY_PRICE_COLUMN]
    else:
        grid[ENERGY_PRICE_COLUMN] = np.nan
        grid["energy_cost_per_unit"] = np.nan
    grid["unit_landed_cost"] = compute_unit_landed_cost(
        grid["unit_cost"], grid["transport_cost_per_unit"], grid["duty_pct"],
        energy_cost_per_unit=grid["energy_cost_per_unit"] if energy_included else None,
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
    ignored = [c for c in plants_df.columns if "price" in str(c).lower() and c != ENERGY_PRICE_COLUMN]
    if ignored:
        warnings.append(f"plants_df columns not used by the cost formula: {_fmt_items(ignored)}")

    result.attrs["assumptions"] = cost_model_assumptions(energy_included, weekly_carbon=False)
    result.attrs["warnings"] = warnings
    result.attrs["planning_date"] = planning.date().isoformat()
    result.attrs["energy_cost_included"] = energy_included
    return result


# ============================================================================
# MULTI-WEEK EXPANSION
# ============================================================================

def default_horizon_weeks(candidates: pd.DataFrame, planning_date: Any) -> int:
    """Weeks from the planning date through the week containing the latest due date (at least 1)."""
    planning = parse_planning_date(planning_date)
    days = int((candidates["due"].max() - planning).days)
    return max(1, days // 7 + 1)


def validate_horizon_weeks(horizon_weeks: Any) -> int:
    """Validate an explicit horizon: a whole number of weeks in ``[1, MAX_HORIZON_WEEKS]``."""
    if isinstance(horizon_weeks, (bool, np.bool_)) or not isinstance(horizon_weeks, (int, np.integer)):
        raise DataValidationError(f"horizon_weeks must be a whole number, got {horizon_weeks!r}")
    if not 1 <= int(horizon_weeks) <= MAX_HORIZON_WEEKS:
        raise DataValidationError(f"horizon_weeks must be between 1 and {MAX_HORIZON_WEEKS}, got {horizon_weeks}")
    return int(horizon_weeks)


def expand_to_weeks(
    candidates: pd.DataFrame,
    planning_date: Any,
    horizon_weeks: int,
    carbon_weekly_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per eligible (order, plant, production week) with week-specific delivery and carbon.

    Cost does not depend on the week. Arrival, lateness, ``days_late`` and
    carbon (when a weekly carbon table is given) do. Ineligible candidates
    are dropped. ``attrs["warnings"]`` lists weeks that fell back to the
    plants.csv carbon intensity.
    """
    planning = parse_planning_date(planning_date)
    weeks = pd.DataFrame({"production_week": np.arange(horizon_weeks)})
    weeks["week_start"] = planning + pd.to_timedelta(7 * weeks["production_week"], unit="D")
    eligible = candidates.loc[candidates["eligible"]].reset_index(drop=True)
    out = eligible.assign(_row=np.arange(len(eligible))).merge(weeks, how="cross")

    out["arrival_date"] = out["week_start"] + pd.to_timedelta(out["transit_days"].astype("float64"), unit="D")
    out["is_late"] = (out["arrival_date"] > out["due"]).astype(bool)
    out["days_late"] = (out["arrival_date"] - out["due"]).dt.days.clip(lower=0).astype("int64")

    warnings: List[str] = []
    out["carbon_intensity_source"] = "plants.csv"
    if carbon_weekly_df is not None:
        weekly = prepare_carbon_weekly(carbon_weekly_df).rename(
            columns={"week_start": "_cw_start", "carbon_kg_per_kwh": "_cw_value"}
        ).sort_values("_cw_start")
        keys = out[["plant", "week_start"]].drop_duplicates().sort_values("week_start")
        matched = pd.merge_asof(
            keys, weekly, left_on="week_start", right_on="_cw_start", by="plant",
            direction="backward", tolerance=pd.Timedelta(days=6),
        )
        out = out.merge(matched[["plant", "week_start", "_cw_value"]], on=["plant", "week_start"], how="left")
        has = out["_cw_value"].notna()
        out.loc[has, "carbon_kg_per_kwh"] = out.loc[has, "_cw_value"]
        out.loc[has, "carbon_intensity_source"] = "weekly table"
        missing = matched.loc[matched["_cw_value"].isna()]
        if not missing.empty:
            gaps = [f"{r.plant} {r.week_start.date()}" for r in missing.itertuples(index=False)]
            warnings.append(f"No weekly carbon intensity for {_fmt_items(gaps)}; plants.csv value used")
        out = out.drop(columns=["_cw_value"])

    out["energy_carbon_kg_per_unit"] = compute_energy_carbon_per_unit(out["energy_kwh_per_unit"], out["carbon_kg_per_kwh"])
    out["unit_carbon_kg"] = compute_unit_carbon(out["energy_carbon_kg_per_unit"], out["carbon_kg_per_unit"])
    out["total_carbon_kg"] = out["qty"] * out["unit_carbon_kg"]

    out = out.sort_values(["_row", "production_week"], kind="stable").drop(columns=["_row"]).reset_index(drop=True)
    out.attrs = dict(candidates.attrs)
    out.attrs["warnings"] = list(candidates.attrs.get("warnings", [])) + warnings
    out.attrs["assumptions"] = cost_model_assumptions(
        bool(candidates.attrs.get("energy_cost_included")), weekly_carbon=carbon_weekly_df is not None
    )
    out.attrs["horizon_weeks"] = horizon_weeks
    return out
