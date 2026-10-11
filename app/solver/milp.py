"""Multi-week binary MILP order allocation using PuLP and the bundled CBC solver.

Model
-----
Weeks ``w = 0 .. H-1`` start on ``planning_date + 7*w``. ``H`` defaults to the
weeks from the planning date through the latest due date
(:func:`app.solver.cost_carbon.default_horizon_weeks`).

Decision variables: ``x[i, p, w] in {0, 1}``: order ``i`` is produced at
plant ``p`` in week ``w``. One variable per eligible (order, plant, week) row
from :func:`app.solver.cost_carbon.expand_to_weeks`. Each ineligible
(order, plant) pair also gets one variable that is fixed to 0.

Constraints:

A. Each order is assigned exactly once: ``sum_{p,w} x[i, p, w] == 1``
   (``<= 1`` in partial mode; see below).
B. Weekly plant-product capacity:
   ``sum_{i of product q} qty[i] * x[i, p, w] <= limit[p, q, w]``.
C. Eligibility: ineligible (order, plant) variables are fixed to 0.
D. Optional hard delivery: with ``require_on_time=True``, late rows are fixed to 0.
E. Capacity changes: ``limit[p, q, w] = weekly_capacity[p, q] * override[p] * event_fraction[p, w]``
   (:func:`app.solver.capacity_events.effective_weekly_limits`). ``capacity_overrides``
   apply to every week; dated ``capacity_events`` only to the weeks they overlap.
F. Optional carbon cap: ``total_carbon_kg <= baseline_carbon_kg * (1 - cut/100)``.

Objective (minimised)::

    w_cost   * total_cost      / cost_scale
  + w_carbon * total_carbon_kg / carbon_scale
  + tardiness_weight * total_days_late
  + lateness_penalty * late_orders

Scales are the sum over orders of the cheapest (respectively lowest-carbon)
eligible option. They depend only on the data, so every scenario shares them.
A non-positive sum falls back to 1.0, which is reported in ``diagnostics``.

``tardiness_weight`` is scaled so the whole term is at most 1.0. Among late
orders the solver therefore prefers fewer days late; without it, a late
order could be pushed to any later week at no cost.

Lateness penalty: by default ``max(DEFAULT_LATENESS_PENALTY, 2 + D)``, where
``D`` bounds the largest possible cost/carbon swing between two plans. Adding
1 for the tardiness term makes minimising the number of late orders strictly
lexicographic. ``weights["lateness"]`` overrides it verbatim, with no guarantee.

Partial mode (``allow_partial=True``): if not every order fits, the solver
first maximises the units assigned, then optimises the objective above with
that many units fixed. The result has status ``PARTIAL`` and lists the
``unassigned_orders`` (the Section 6.2 "closest option").

Carbon baseline (for ``carbon_cut_pct``): total carbon of the plan this module
produces with :data:`BASELINE_WEIGHTS`, no carbon cap, and the *same*
capacity, delivery and horizon settings. Pass ``baseline_carbon_kg`` to share
one baseline across several solves.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

from app.solver.capacity_events import effective_weekly_limits
from app.solver.cost_carbon import (
    calculate_candidate_metrics,
    capacity_limits,
    default_horizon_weeks,
    expand_to_weeks,
    kg_to_tonnes,
    nest_by_plant,
    parse_planning_date,
    plant_capacities,
    prepare_plants,
    to_json_compatible,
    validate_horizon_weeks,
)

# ============================================================================
# CONSTANTS
# ============================================================================

#: Default objective weights (cost dominant, carbon secondary).
DEFAULT_WEIGHTS: Dict[str, float] = {"cost": 0.7, "carbon": 0.3}

#: Weights defining the carbon-reduction baseline plan.
BASELINE_WEIGHTS: Dict[str, float] = dict(DEFAULT_WEIGHTS)

#: Minimum penalty per late order, in normalised objective units (cost and
#: carbon terms are each ~1.0 for a near-optimal plan).
DEFAULT_LATENESS_PENALTY: float = 100.0

#: Upper bound of the whole tardiness (days-late) term, in normalised units.
TARDINESS_TERM_MAX: float = 1.0

#: Relative slack on the carbon cap to absorb floating-point rounding.
CARBON_CAP_REL_TOLERANCE: float = 1e-9

#: A binary variable is treated as selected when its value exceeds this.
SELECTION_THRESHOLD: float = 0.5

SOLVER_NAME = "PuLP/CBC (PULP_CBC_CMD)"

STATUS_OPTIMAL = "OPTIMAL"
STATUS_FEASIBLE = "FEASIBLE"  # integer-feasible but optimality not proven (e.g. time limit)
STATUS_PARTIAL = "PARTIAL"  # allow_partial=True and not every order could be assigned
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_UNBOUNDED = "UNBOUNDED"
STATUS_NOT_SOLVED = "NOT_SOLVED"
STATUS_ERROR = "ERROR"

SUCCESS_STATUSES = frozenset({STATUS_OPTIMAL, STATUS_FEASIBLE})

_ALLOWED_WEIGHT_KEYS = frozenset({"cost", "carbon", "lateness"})


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def _is_real_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def validate_weights(weights: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, float], Optional[float]]:
    """Validate objective weights.

    Args:
        weights: ``{"cost": w_c, "carbon": w_e}`` with optional
            ``"lateness"`` (penalty per late order in normalised units).
            ``None`` selects :data:`DEFAULT_WEIGHTS`.

    Returns:
        ``(normalised {"cost", "carbon"} summing to 1, lateness override or None)``.

    Raises:
        ValueError: for non-mappings, unknown or missing keys, non-numeric,
            negative or non-finite values, or cost + carbon == 0.
    """
    if weights is None:
        return dict(DEFAULT_WEIGHTS), None
    if not isinstance(weights, Mapping):
        raise ValueError(f"weights must be a mapping like {DEFAULT_WEIGHTS}, got {type(weights).__name__}")
    unknown = set(weights) - _ALLOWED_WEIGHT_KEYS
    if unknown:
        raise ValueError(f"Unknown weight keys {sorted(map(str, unknown))}; allowed: {sorted(_ALLOWED_WEIGHT_KEYS)}")
    missing = {"cost", "carbon"} - set(weights)
    if missing:
        raise ValueError(f"weights must include {sorted(missing)}")
    for key, value in weights.items():
        if not _is_real_number(value):
            raise ValueError(f"weights[{key!r}] must be a finite number, got {value!r}")
        if float(value) < 0:
            raise ValueError(f"weights[{key!r}] must be non-negative, got {value!r}")
    total = float(weights["cost"]) + float(weights["carbon"])
    if total <= 0:
        raise ValueError("weights['cost'] + weights['carbon'] must be positive")
    normalised = {"cost": float(weights["cost"]) / total, "carbon": float(weights["carbon"]) / total}
    lateness = float(weights["lateness"]) if "lateness" in weights else None
    return normalised, lateness


def validate_carbon_cut_pct(carbon_cut_pct: Any) -> Optional[float]:
    """Validate a carbon-reduction percentage in ``[0, 100]`` (``None`` = no cap)."""
    if carbon_cut_pct is None:
        return None
    if not _is_real_number(carbon_cut_pct):
        raise ValueError(f"carbon_cut_pct must be a finite number in [0, 100], got {carbon_cut_pct!r}")
    value = float(carbon_cut_pct)
    if value < 0 or value > 100:
        raise ValueError(f"carbon_cut_pct must be within [0, 100], got {carbon_cut_pct!r}")
    return value


def _validate_time_limit(time_limit_seconds: Any) -> Optional[float]:
    if time_limit_seconds is None:
        return None
    if not _is_real_number(time_limit_seconds) or float(time_limit_seconds) <= 0:
        raise ValueError(f"time_limit_seconds must be a positive number, got {time_limit_seconds!r}")
    return float(time_limit_seconds)


def _validate_baseline(baseline_carbon_kg: Any) -> Optional[float]:
    if baseline_carbon_kg is None:
        return None
    if not _is_real_number(baseline_carbon_kg) or float(baseline_carbon_kg) < 0:
        raise ValueError(f"baseline_carbon_kg must be a non-negative finite number, got {baseline_carbon_kg!r}")
    return float(baseline_carbon_kg)


# ============================================================================
# OBJECTIVE SCALING
# ============================================================================

def compute_objective_scales(weekly: pd.DataFrame) -> Dict[str, Any]:
    """Return positive normalisation scales and the bounds used for the lateness penalty.

    ``cost_scale`` / ``carbon_scale`` are the sum over orders of the minimum
    total over that order's eligible (plant, week) rows. ``*_range_sum`` is
    the sum over orders of (max - min), which bounds how much the objective can
    change between two plans. Zero or negative sums fall back to 1.0 (flagged).
    """
    per_order = weekly.groupby("order_id", sort=False).agg(
        cost_min=("total_cost", "min"),
        cost_max=("total_cost", "max"),
        carbon_min=("total_carbon_kg", "min"),
        carbon_max=("total_carbon_kg", "max"),
    )
    raw_cost = math.fsum(per_order["cost_min"])
    raw_carbon = math.fsum(per_order["carbon_min"])
    return {
        "cost_scale": raw_cost if raw_cost > 0 else 1.0,
        "carbon_scale": raw_carbon if raw_carbon > 0 else 1.0,
        "cost_scale_fallback": not raw_cost > 0,
        "carbon_scale_fallback": not raw_carbon > 0,
        "cost_range_sum": math.fsum(per_order["cost_max"] - per_order["cost_min"]),
        "carbon_range_sum": math.fsum(per_order["carbon_max"] - per_order["carbon_min"]),
    }


def default_lateness_penalty(scales: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    """Per-late-order penalty that strictly dominates any cost/carbon/tardiness trade-off."""
    max_swing = (
        weights["cost"] * scales["cost_range_sum"] / scales["cost_scale"]
        + weights["carbon"] * scales["carbon_range_sum"] / scales["carbon_scale"]
    )
    return max(DEFAULT_LATENESS_PENALTY, 1.0 + max_swing + TARDINESS_TERM_MAX)


def _tardiness_weight(weekly: pd.DataFrame) -> float:
    """Per-day weight so that the whole days-late term is at most TARDINESS_TERM_MAX."""
    n_orders = weekly["order_id"].nunique()
    max_days = int(weekly["days_late"].max()) if len(weekly) else 0
    return TARDINESS_TERM_MAX / (max(n_orders, 1) * (max_days + 1))


# ============================================================================
# MODEL BUILD AND SOLVE
# ============================================================================

def _map_status(problem: pulp.LpProblem) -> str:
    if problem.sol_status == pulp.LpSolutionOptimal:
        return STATUS_OPTIMAL
    if problem.sol_status == pulp.LpSolutionIntegerFeasible:
        return STATUS_FEASIBLE
    if problem.status == pulp.LpStatusInfeasible or problem.sol_status == pulp.LpSolutionInfeasible:
        return STATUS_INFEASIBLE
    if problem.status == pulp.LpStatusUnbounded or problem.sol_status == pulp.LpSolutionUnbounded:
        return STATUS_UNBOUNDED
    return STATUS_NOT_SOLVED


def _solve_model(
    weekly: pd.DataFrame,
    ineligible_pairs: List[Tuple[str, str]],
    week_limits: Mapping[Tuple[str, str, int], float],
    weights: Mapping[str, float],
    lateness_penalty: float,
    scales: Mapping[str, Any],
    carbon_cap_kg: Optional[float],
    require_on_time: bool,
    time_limit_seconds: Optional[float],
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Build and solve the MILP; return status, chosen weekly row indices and model stats."""
    n = len(weekly)
    x = [pulp.LpVariable(f"x_{k}", cat=pulp.LpBinary) for k in range(n)]
    x_inel = [pulp.LpVariable(f"z_{k}", cat=pulp.LpBinary) for k in range(len(ineligible_pairs))]

    late = weekly["is_late"].to_numpy(dtype=bool)
    qty = weekly["qty"].to_numpy(dtype=float)
    cost = weekly["total_cost"].to_numpy(dtype=float)
    carbon = weekly["total_carbon_kg"].to_numpy(dtype=float)
    days_late = weekly["days_late"].to_numpy(dtype=float)
    allowed = ~late if require_on_time else np.ones(n, dtype=bool)
    w_cost = weights["cost"] / scales["cost_scale"]
    w_carbon = weights["carbon"] / scales["carbon_scale"]
    w_tardy = _tardiness_weight(weekly)

    order_rows = weekly.groupby("order_id", sort=False).indices
    inel_by_order: Dict[str, List[int]] = {}
    for k, (order_id, _plant) in enumerate(ineligible_pairs):
        inel_by_order.setdefault(order_id, []).append(k)

    def base_problem(name: str, sense: int) -> pulp.LpProblem:
        prob = pulp.LpProblem(name, sense)
        # A. each order assigned once (at most once in partial mode)
        for pos, (order_id, rows) in enumerate(order_rows.items()):
            expr = pulp.lpSum(x[k] for k in rows) + pulp.lpSum(x_inel[j] for j in inel_by_order.get(order_id, []))
            prob += (expr <= 1 if allow_partial else expr == 1), f"assign_{pos}"
        # C. ineligible pairs fixed to zero
        for j, var in enumerate(x_inel):
            prob += var == 0, f"ineligible_{j}"
        # D. late rows fixed to zero when on-time is required
        for k in np.flatnonzero(~allowed):
            prob += x[k] == 0, f"late_{k}"
        # B/E. weekly plant-product capacity
        groups = weekly.groupby(["plant", "product", "production_week"], sort=True).indices.items()
        for pos, (key, rows) in enumerate(groups):
            terms = [(x[k], qty[k]) for k in rows if allowed[k]]
            if terms:
                prob += pulp.LpAffineExpression(terms) <= week_limits[key], f"capacity_{pos}"
        # F. carbon cap
        if carbon_cap_kg is not None:
            prob += (
                pulp.LpAffineExpression([(x[k], carbon[k]) for k in range(n) if allowed[k]])
                <= carbon_cap_kg * (1.0 + CARBON_CAP_REL_TOLERANCE),
                "carbon_cap",
            )
        return prob

    try:
        import highspy
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit_seconds)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)

    start = time.perf_counter()
    units_assigned_max: Optional[float] = None
    if allow_partial:
        phase1 = base_problem("max_units", pulp.LpMaximize)
        phase1 += pulp.LpAffineExpression([(x[k], qty[k]) for k in range(n) if allowed[k]]), "units"
        try:
            phase1.solve(solver)
        except Exception:
            phase1.solve(pulp.LpSolverDefault)
        if _map_status(phase1) not in SUCCESS_STATUSES:
            solve_ms = (time.perf_counter() - start) * 1000.0
            return {"status": _map_status(phase1), "chosen": [], "unassigned": [], "extraction_problems": [], "objective_value": None,
                    "solve_ms": solve_ms, "pulp_status": pulp.LpStatus[phase1.status],
                    "pulp_solution_status": pulp.LpSolution[phase1.sol_status],
                    "num_variables": n + len(x_inel), "num_constraints": len(phase1.constraints)}
        units_assigned_max = float(pulp.value(phase1.objective) or 0.0)

    problem = base_problem("order_allocation", pulp.LpMinimize)
    problem += pulp.LpAffineExpression([
        (x[k], w_cost * cost[k] + w_carbon * carbon[k] + w_tardy * days_late[k] + (lateness_penalty if late[k] else 0.0))
        for k in range(n) if allowed[k]
    ]), "weighted_objective"
    if units_assigned_max is not None:
        # Integer units: keep the phase-1 maximum (0.5 absorbs float noise).
        problem += (
            pulp.LpAffineExpression([(x[k], qty[k]) for k in range(n) if allowed[k]]) >= units_assigned_max - 0.5,
            "keep_max_units",
        )
    try:
        problem.solve(solver)
    except Exception:
        problem.solve(pulp.LpSolverDefault)
    solve_ms = (time.perf_counter() - start) * 1000.0

    status = _map_status(problem)
    chosen: List[int] = []
    problems: List[str] = []
    unassigned: List[str] = []
    if status in SUCCESS_STATUSES:
        values = np.array([v.varValue if v.varValue is not None else 0.0 for v in x])
        for order_id, rows in order_rows.items():
            picked = [k for k in rows if values[k] > SELECTION_THRESHOLD]
            if len(picked) > 1 or (len(picked) == 0 and not allow_partial):
                problems.append(f"order {order_id} has {len(picked)} selected (plant, week) rows in the solver solution")
            elif picked:
                chosen.append(int(picked[0]))
            else:
                unassigned.append(order_id)
    return {
        "status": status,
        "chosen": chosen,
        "unassigned": unassigned,
        "extraction_problems": problems,
        "objective_value": pulp.value(problem.objective) if status in SUCCESS_STATUSES else None,
        "solve_ms": solve_ms,
        "pulp_status": pulp.LpStatus[problem.status],
        "pulp_solution_status": pulp.LpSolution[problem.sol_status],
        "num_variables": n + len(x_inel),
        "num_constraints": len(problem.constraints),
    }


def _post_solve_checks(
    chosen_rows: pd.DataFrame,
    week_limits: Mapping[Tuple[str, str, int], float],
    carbon_cap_kg: Optional[float],
    require_on_time: bool,
) -> List[str]:
    """Sanity-check a solver solution against hard rules before reporting success."""
    issues: List[str] = []
    loads = chosen_rows.groupby(["plant", "product", "production_week"])["qty"].sum()
    for key, load in loads.items():
        if load > week_limits[key] + 1e-6:
            issues.append(f"plant {key[0]} product {key[1]} week {key[2]} load {load} exceeds capacity {week_limits[key]}")
    if require_on_time and chosen_rows["is_late"].any():
        issues.append("solution contains late orders although require_on_time=True")
    if carbon_cap_kg is not None:
        total = math.fsum(chosen_rows["total_carbon_kg"])
        if total > carbon_cap_kg * (1.0 + CARBON_CAP_REL_TOLERANCE) + 1e-6:
            issues.append(f"solution carbon {total} exceeds cap {carbon_cap_kg}")
    return issues


def _infeasibility_hints(
    weekly: pd.DataFrame,
    orders: pd.DataFrame,
    week_limits: Mapping[Tuple[str, str, int], float],
    require_on_time: bool,
    carbon_cap_kg: Optional[float],
) -> List[str]:
    """Necessary-condition checks that explain common causes of infeasibility."""
    hints: List[str] = []
    allowed_rows = weekly.loc[~weekly["is_late"]] if require_on_time else weekly

    total_demand = int(orders["qty"].sum())
    total_cap = math.fsum(week_limits.values())
    if total_demand > total_cap:
        hints.append(f"Total demand {total_demand} units exceeds total effective capacity {total_cap:g} over the horizon")

    no_option = sorted(set(orders["order_id"]) - set(allowed_rows["order_id"]))
    if no_option:
        why = "eligible plant and week that delivers on time" if require_on_time else "eligible plant"
        hints.append(f"Orders with no {why}: {', '.join(no_option)}")

    caps = [week_limits[(p, q, w)] for p, q, w in zip(allowed_rows["plant"], allowed_rows["product"], allowed_rows["production_week"])]
    too_big = allowed_rows.assign(cap=caps).groupby("order_id").agg(qty=("qty", "first"), cap=("cap", "max"))
    for order_id, row in too_big.loc[too_big["qty"] > too_big["cap"]].iterrows():
        hints.append(f"Order {order_id} qty {int(row['qty'])} exceeds the largest weekly plant-product capacity {row['cap']:g} available to it")

    for product, group in allowed_rows.groupby("product"):
        demand = int(group.drop_duplicates("order_id")["qty"].sum())
        keys = group[["plant", "production_week"]].drop_duplicates()
        cap = math.fsum(week_limits[(p, product, w)] for p, w in keys.itertuples(index=False))
        if demand > cap:
            hints.append(f"Product {product}: demand {demand} exceeds its combined capacity {cap:g} over the horizon")

    if carbon_cap_kg is not None and not allowed_rows.empty:
        min_carbon = math.fsum(allowed_rows.groupby("order_id")["total_carbon_kg"].min())
        if min_carbon > carbon_cap_kg * (1.0 + CARBON_CAP_REL_TOLERANCE):
            hints.append(f"Carbon cap {carbon_cap_kg:.3f} kg is below the minimum achievable {min_carbon:.3f} kg even ignoring capacity")

    if not hints:
        hints.append("No single-cause bottleneck found; infeasibility arises from the interaction of capacity, eligibility, delivery and carbon constraints")
    return hints


# ============================================================================
# RESULT ASSEMBLY
# ============================================================================

ALLOCATION_COLUMNS = [
    "order_id", "plant", "product", "destination", "qty", "due",
    "production_week", "week_start", "unit_landed_cost", "total_cost",
    "unit_carbon_kg", "total_carbon_kg", "transit_days", "arrival_date", "is_late", "days_late",
]


def _allocation_records(chosen_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    records = chosen_rows[ALLOCATION_COLUMNS].rename(columns={"week_start": "production_week_start"})
    return [to_json_compatible(rec) for rec in records.to_dict(orient="records")]


def _ratio(load: float, cap: float) -> Optional[float]:
    return load / cap if cap > 0 else (0.0 if load == 0 else None)


def _utilisation(
    chosen_rows: pd.DataFrame,
    week_limits: Mapping[Tuple[str, str, int], float],
    week_starts: List[pd.Timestamp],
) -> Dict[str, Any]:
    """Utilisation over the horizon (plant and plant-product) and per plant per week."""
    cap_pp: Dict[Tuple[str, str], float] = {}
    cap_pw: Dict[Tuple[str, int], float] = {}
    for (plant, product, week), cap in week_limits.items():
        cap_pp[(plant, product)] = cap_pp.get((plant, product), 0.0) + cap
        cap_pw[(plant, week)] = cap_pw.get((plant, week), 0.0) + cap
    cap_p: Dict[str, float] = {}
    for (plant, _), cap in cap_pp.items():
        cap_p[plant] = cap_p.get(plant, 0.0) + cap

    load_pp = chosen_rows.groupby(["plant", "product"])["qty"].sum().to_dict()
    load_pw = chosen_rows.groupby(["plant", "production_week"])["qty"].sum().to_dict()
    load_p = chosen_rows.groupby("plant")["qty"].sum().to_dict()
    week_label = {w: ts.date().isoformat() for w, ts in enumerate(week_starts)}
    return {
        "plant_utilisation": {p: _ratio(float(load_p.get(p, 0)), c) for p, c in cap_p.items()},
        "plant_load_units": {p: float(load_p.get(p, 0)) for p in cap_p},
        "effective_capacity": cap_p,
        "load_by_product": nest_by_plant({k: float(load_pp.get(k, 0)) for k in cap_pp}),
        "effective_capacity_by_product": nest_by_plant(cap_pp),
        "utilisation_by_product": nest_by_plant({k: _ratio(float(load_pp.get(k, 0)), c) for k, c in cap_pp.items()}),
        "utilisation_by_week": nest_by_plant(
            {(p, week_label[w]): _ratio(float(load_pw.get((p, w), 0)), c) for (p, w), c in cap_pw.items()}
        ),
    }


def _unassigned_records(orders: pd.DataFrame, weekly: pd.DataFrame, unassigned: List[str], require_on_time: bool) -> List[Dict[str, Any]]:
    out = []
    for o in orders.loc[orders["order_id"].isin(unassigned)].itertuples(index=False):
        plants = sorted(weekly.loc[weekly["order_id"] == o.order_id, "plant"].unique())
        out.append(to_json_compatible({
            "order_id": o.order_id, "product": o.product, "qty": o.qty, "destination": o.destination, "due": o.due,
            "eligible_plants": plants,
            "reason": "no remaining weekly capacity for this product at any eligible plant within the horizon"
            + (" that delivers on time" if require_on_time else ""),
        }))
    return out


def _empty_result(status: str, weights_used: Mapping[str, float], assumptions: List[str], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return to_json_compatible({
        "status": status,
        "allocation": [],
        "unassigned_orders": [],
        "total_cost": None,
        "total_carbon_kg": None,
        "total_carbon_t": None,
        "late_orders": None,
        "late_order_ids": [],
        "plant_utilisation": {},
        "weights_used": dict(weights_used),
        "objective_value": None,
        "assumptions": assumptions,
        "diagnostics": diagnostics,
    })


# ============================================================================
# PUBLIC API
# ============================================================================

def solve_allocation(
    orders_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    planning_date: Any,
    weights: Optional[Mapping[str, Any]] = None,
    capacity_overrides: Optional[Mapping[str, Any]] = None,
    carbon_cut_pct: Optional[float] = None,
    require_on_time: bool = False,
    time_limit_seconds: Optional[float] = None,
    baseline_carbon_kg: Optional[float] = None,
    capacity_events: Optional[pd.DataFrame] = None,
    carbon_weekly: Optional[pd.DataFrame] = None,
    horizon_weeks: Optional[int] = None,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Allocate every order to a plant and production week by solving a binary MILP.

    Args:
        orders_df, plants_df, tariffs_df: Input tables (Annex B schema). Not mutated.
        planning_date: Start of week 0 of the planning horizon.
        weights: ``{"cost", "carbon"}`` (normalised to sum to 1) and optional
            ``"lateness"`` penalty per late order. Defaults to :data:`DEFAULT_WEIGHTS`.
        capacity_overrides: ``{plant: retained_fraction}`` for every week; ``{"B": 0.7}`` is a 30% loss.
        carbon_cut_pct: Require ``total_carbon_kg <= baseline * (1 - pct/100)``.
        require_on_time: Forbid late assignments (hard constraint).
        time_limit_seconds: Optional CBC time limit. A solution found but not
            proven optimal is returned with status ``"FEASIBLE"``.
        baseline_carbon_kg: Pre-computed carbon baseline so several solves share one.
        capacity_events: Dated ``capacity_events.csv`` rows; each reduces only the weeks it overlaps.
        carbon_weekly: Optional ``plant, week_start, carbon_kg_per_kwh`` table (Section 4.4).
        horizon_weeks: Number of planning weeks; default covers the latest due date.
        allow_partial: When not every order fits, return the closest plan
            (status ``PARTIAL``, ``unassigned_orders``) instead of ``INFEASIBLE``.

    Returns:
        JSON-compatible dict with ``status``, ``allocation`` (each entry with
        ``production_week`` and ``production_week_start``), ``unassigned_orders``,
        ``total_cost``, ``total_carbon_kg``, ``total_carbon_t``, ``late_orders``,
        ``late_order_ids``, ``plant_utilisation`` (load / capacity over the horizon),
        ``weights_used``, ``objective_value``, ``assumptions`` and ``diagnostics``.
        ``INFEASIBLE``, ``NOT_SOLVED`` and ``ERROR`` carry an empty allocation and ``None`` totals.

    Raises:
        DataValidationError: invalid input tables, events, horizon or capacity overrides.
        ValueError: invalid weights, carbon_cut_pct, time limit or baseline.
    """
    t_start = time.perf_counter()
    weights_used, lateness_override = validate_weights(weights)
    cut = validate_carbon_cut_pct(carbon_cut_pct)
    time_limit = _validate_time_limit(time_limit_seconds)
    baseline = _validate_baseline(baseline_carbon_kg)
    for name, flag in (("require_on_time", require_on_time), ("allow_partial", allow_partial)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be a bool, got {flag!r}")
    require_on_time, allow_partial = bool(require_on_time), bool(allow_partial)

    planning = parse_planning_date(planning_date)
    candidates = calculate_candidate_metrics(orders_df, plants_df, tariffs_df, planning)
    horizon = validate_horizon_weeks(horizon_weeks) if horizon_weeks is not None else default_horizon_weeks(candidates, planning)
    weekly = expand_to_weeks(candidates, planning, horizon, carbon_weekly)
    ineligible_pairs = list(candidates.loc[~candidates["eligible"], ["order_id", "plant"]].itertuples(index=False, name=None))
    orders = candidates.drop_duplicates("order_id")[["order_id", "product", "qty", "destination", "due"]]

    plants_clean = prepare_plants(plants_df)
    limits_info = effective_weekly_limits(capacity_limits(plants_clean), planning, horizon, capacity_overrides, capacity_events)
    week_limits = limits_info["limits"]
    week_starts = [planning + pd.Timedelta(days=7 * w) for w in range(horizon)]

    scales = compute_objective_scales(weekly)
    penalty = lateness_override if lateness_override is not None else default_lateness_penalty(scales, weights_used)
    preprocessing_ms = (time.perf_counter() - t_start) * 1000.0
    if lateness_override is not None:
        weights_used = {**weights_used, "lateness": lateness_override}

    assumptions = list(weekly.attrs.get("assumptions", [])) + [
        f"Planning horizon {horizon} week(s) from {planning.date().isoformat()} to "
        f"{(week_starts[-1] + pd.Timedelta(days=6)).date().isoformat()}",
        f"Weights {weights_used['cost']:.0%} cost, {weights_used['carbon']:.0%} carbon (normalised to sum to 1)",
        "Cost and carbon are normalised by the sum of each order's cheapest / lowest-carbon eligible option",
        (
            f"Lateness penalty {penalty:g} per late order (custom; no lexicographic guarantee)"
            if lateness_override is not None
            else f"Lateness penalty {penalty:g} per late order: fewest late orders first, then fewest days late, then cost and carbon"
        ),
        "capacity_overrides (no dates) apply to every week; dated capacity events apply only to the weeks they overlap",
    ]
    if limits_info["events"]:
        assumptions.extend(limits_info["events"]["assumptions"])
    if require_on_time:
        assumptions.append("All orders must arrive on time (hard constraint)")
    if allow_partial:
        assumptions.append("If not every order fits, the plan maximises units allocated first and lists unassigned orders")
    assumptions.extend(weekly.attrs.get("warnings", []))

    diagnostics: Dict[str, Any] = {
        "solver": SOLVER_NAME,
        "planning_date": planning,
        "horizon_weeks": horizon,
        "week_starts": week_starts,
        "energy_cost_included": bool(candidates.attrs.get("energy_cost_included")),
        "weekly_carbon_used": carbon_weekly is not None,
        "original_capacity": plant_capacities(plants_clean),
        "capacity_overrides": dict(capacity_overrides or {}),
        "capacity_events": limits_info["events"],
        "require_on_time": require_on_time,
        "carbon_cut_pct": cut,
        "lateness_penalty_per_order": penalty,
        **{k: scales[k] for k in ("cost_scale", "carbon_scale", "cost_scale_fallback", "carbon_scale_fallback")},
        "preprocessing_ms": preprocessing_ms,
    }

    def run(w: Mapping[str, float], pen: float, cap: Optional[float]) -> Dict[str, Any]:
        return _solve_model(weekly, ineligible_pairs, week_limits, w, pen, scales, cap, require_on_time, time_limit, allow_partial)

    carbon_cap: Optional[float] = None
    baseline_solve_ms = 0.0
    if cut is not None:
        if baseline is None:
            base_run = run(BASELINE_WEIGHTS, default_lateness_penalty(scales, BASELINE_WEIGHTS), None)
            baseline_solve_ms = base_run["solve_ms"]
            if base_run["status"] not in SUCCESS_STATUSES or base_run["extraction_problems"] or base_run["unassigned"]:
                diagnostics.update({
                    "baseline_status": STATUS_PARTIAL if base_run.get("unassigned") else base_run["status"],
                    "baseline_solve_ms": baseline_solve_ms,
                    "reason": "Carbon baseline plan could not assign every order, so the carbon cap is undefined",
                    "infeasibility_hints": _infeasibility_hints(weekly, orders, week_limits, require_on_time, None),
                })
                return _empty_result(STATUS_INFEASIBLE, weights_used, assumptions, diagnostics)
            baseline = math.fsum(weekly.loc[base_run["chosen"], "total_carbon_kg"])
            diagnostics["baseline_source"] = "solved internally with BASELINE_WEIGHTS"
        else:
            diagnostics["baseline_source"] = "provided by caller"
        carbon_cap = baseline * (1.0 - cut / 100.0)
        diagnostics.update({"baseline_carbon_kg": baseline, "carbon_cap_kg": carbon_cap, "baseline_solve_ms": baseline_solve_ms})
        assumptions.append(
            f"Carbon cut {cut:g}% vs baseline {baseline:.3f} kg (plan with baseline weights "
            f"{BASELINE_WEIGHTS['cost']:.0%}/{BASELINE_WEIGHTS['carbon']:.0%}, same capacity, delivery and horizon settings, no cap)"
        )

    result_run = run(weights_used, penalty, carbon_cap)
    diagnostics.update({k: result_run[k] for k in ("solve_ms", "pulp_status", "pulp_solution_status", "num_variables", "num_constraints")})
    diagnostics["total_ms"] = (time.perf_counter() - t_start) * 1000.0

    if result_run["status"] not in SUCCESS_STATUSES:
        if result_run["status"] == STATUS_INFEASIBLE:
            diagnostics["infeasibility_hints"] = _infeasibility_hints(weekly, orders, week_limits, require_on_time, carbon_cap)
        return _empty_result(result_run["status"], weights_used, assumptions, diagnostics)

    chosen_rows = weekly.loc[result_run["chosen"]]
    issues = result_run["extraction_problems"] or _post_solve_checks(chosen_rows, week_limits, carbon_cap, require_on_time)
    if issues:
        diagnostics["errors"] = issues
        return _empty_result(STATUS_ERROR, weights_used, assumptions, diagnostics)

    status = result_run["status"]
    unassigned = _unassigned_records(orders, weekly, result_run["unassigned"], require_on_time)
    if unassigned:
        status = STATUS_PARTIAL
        diagnostics["infeasibility_hints"] = _infeasibility_hints(weekly, orders, week_limits, require_on_time, carbon_cap)

    usage = _utilisation(chosen_rows, week_limits, week_starts)
    diagnostics.update({k: v for k, v in usage.items() if k != "plant_utilisation"})
    late_rows = chosen_rows.loc[chosen_rows["is_late"]]
    total_carbon = math.fsum(chosen_rows["total_carbon_kg"])

    return to_json_compatible({
        "status": status,
        "allocation": _allocation_records(chosen_rows),
        "unassigned_orders": unassigned,
        "total_cost": math.fsum(chosen_rows["total_cost"]),
        "total_carbon_kg": total_carbon,
        "total_carbon_t": kg_to_tonnes(total_carbon),
        "late_orders": int(len(late_rows)),
        "late_order_ids": late_rows["order_id"].tolist(),
        "plant_utilisation": usage["plant_utilisation"],
        "weights_used": weights_used,
        "objective_value": result_run["objective_value"],
        "assumptions": assumptions,
        "diagnostics": diagnostics,
    })
