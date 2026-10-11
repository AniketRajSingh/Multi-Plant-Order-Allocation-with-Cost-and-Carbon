"""Binary MILP order-to-plant allocation using PuLP and the bundled CBC solver.

Model
-----
Decision variables: ``x[i, p] in {0, 1}`` for every order ``i`` and every
plant ``p`` listed in ``plants_df`` (one variable per candidate row returned
by :func:`app.solver.cost_carbon.calculate_candidate_metrics`).

Constraints:

A. Exactly one assignment per order: ``sum_p x[i, p] == 1``.
B. Plant-wide capacity: ``sum_i qty[i] * x[i, p] <= effective_capacity[p]``
   (one constraint per plant, never one per plant-product row).
C. Eligibility: ``x[i, p] == 0`` when plant ``p`` cannot make the product.
D. Optional hard delivery: with ``require_on_time=True``, ``x[i, p] == 0``
   for every late candidate.
E. Capacity overrides: ``effective_capacity = weekly_capacity * retained_fraction``.
F. Optional carbon cap: ``total_carbon_kg <= baseline_carbon_kg * (1 - cut/100)``.

Objective (minimised)::

    w_cost   * total_cost      / cost_scale
  + w_carbon * total_carbon_kg / carbon_scale
  + lateness_penalty * late_orders

Normalisation scales are the sum over orders of the cheapest (respectively
lowest-carbon) *eligible* candidate, i.e. a lower bound on any plan's total.
They depend only on the input data (not on overrides, caps or weights), so
all scenarios and trade-off points share one scale. A non-positive sum (e.g.
every candidate has zero carbon) falls back to a scale of 1.0, which is
reported in ``diagnostics``.

Lateness penalty: by default the per-late-order penalty is
``max(DEFAULT_LATENESS_PENALTY, 1 + D)`` where ``D`` is the largest possible
difference in the weighted normalised cost+carbon terms between any two
plans. That makes minimising the number of late orders strictly lexicographic
over cost/carbon: the solver never accepts an extra late order to save money
or carbon. A caller may override it with ``weights["lateness"]`` (used
verbatim, no lexicographic guarantee).

Planning period: one week starting at ``planning_date``. ``capacity_overrides``
are applied to the whole week; date-ranged capacity events must be converted
to retained fractions by the caller (``1 - capacity_loss_pct / 100``).

Carbon baseline (for ``carbon_cut_pct``): total carbon of the plan this module
produces with :data:`BASELINE_WEIGHTS`, no carbon cap, and the *same*
capacity overrides and ``require_on_time`` setting. Callers that solve
several points (e.g. a trade-off curve) can pass ``baseline_carbon_kg`` so
every point uses the identical baseline.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

from app.solver.cost_carbon import (
    COST_MODEL_ASSUMPTIONS,
    calculate_candidate_metrics,
    effective_capacities,
    kg_to_tonnes,
    parse_planning_date,
    plant_capacities,
    prepare_plants,
    to_json_compatible,
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

#: Relative slack on the carbon cap to absorb floating-point rounding.
CARBON_CAP_REL_TOLERANCE: float = 1e-9

#: A binary variable is treated as selected when its value exceeds this.
SELECTION_THRESHOLD: float = 0.5

SOLVER_NAME = "PuLP/CBC (PULP_CBC_CMD)"

STATUS_OPTIMAL = "OPTIMAL"
STATUS_FEASIBLE = "FEASIBLE"  # integer-feasible but optimality not proven (e.g. time limit)
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

def compute_objective_scales(candidates: pd.DataFrame) -> Dict[str, Any]:
    """Return positive normalisation scales and the lexicographic lateness bound.

    ``cost_scale`` / ``carbon_scale`` are the sum over orders of the minimum
    eligible total. ``*_range_sum`` is the sum over orders of (max - min)
    eligible totals, used to bound how much the objective can change between
    any two plans. Zero or negative sums fall back to 1.0 (flagged).
    """
    eligible = candidates.loc[candidates["eligible"]]
    per_order = eligible.groupby("order_id", sort=False).agg(
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
    """Per-late-order penalty that strictly dominates any cost/carbon trade-off."""
    max_swing = (
        weights["cost"] * scales["cost_range_sum"] / scales["cost_scale"]
        + weights["carbon"] * scales["carbon_range_sum"] / scales["carbon_scale"]
    )
    return max(DEFAULT_LATENESS_PENALTY, 1.0 + max_swing)


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
    candidates: pd.DataFrame,
    eff_caps: Mapping[str, float],
    weights: Mapping[str, float],
    lateness_penalty: float,
    scales: Mapping[str, Any],
    carbon_cap_kg: Optional[float],
    require_on_time: bool,
    time_limit_seconds: Optional[float],
) -> Dict[str, Any]:
    """Build and solve the MILP; return status, chosen candidate row indices and model stats."""
    problem = pulp.LpProblem("order_allocation", pulp.LpMinimize)
    n = len(candidates)
    x = [pulp.LpVariable(f"x_{k}", cat=pulp.LpBinary) for k in range(n)]

    eligible = candidates["eligible"].to_numpy(dtype=bool)
    late = candidates["is_late"].fillna(False).to_numpy(dtype=bool)
    qty = candidates["qty"].to_numpy(dtype=float)
    cost = np.where(eligible, candidates["total_cost"].to_numpy(dtype=float), 0.0)
    carbon = np.where(eligible, candidates["total_carbon_kg"].to_numpy(dtype=float), 0.0)
    w_cost = weights["cost"] / scales["cost_scale"]
    w_carbon = weights["carbon"] / scales["carbon_scale"]

    allowed = eligible.copy()
    if require_on_time:
        allowed &= ~late

    objective_terms = [
        (x[k], w_cost * cost[k] + w_carbon * carbon[k] + (lateness_penalty if late[k] else 0.0))
        for k in range(n)
        if allowed[k]
    ]
    problem += pulp.LpAffineExpression(objective_terms), "weighted_objective"

    # A. exactly one plant per order (over all candidate plants)
    for pos, rows in enumerate(candidates.groupby("order_id", sort=False).indices.values()):
        problem += pulp.lpSum(x[k] for k in rows) == 1, f"assign_{pos}"
    # C/D. ineligible (and, if required, late) candidates fixed to zero
    for k in np.flatnonzero(~allowed):
        reason = "ineligible" if not eligible[k] else "late"
        problem += x[k] == 0, f"{reason}_{k}"
    # B/E. one plant-wide capacity constraint per plant
    for plant, rows in candidates.groupby("plant", sort=True).indices.items():
        problem += (
            pulp.LpAffineExpression([(x[k], qty[k]) for k in rows if allowed[k]]) <= eff_caps[plant],
            f"capacity_{plant}",
        )
    # F. carbon cap
    if carbon_cap_kg is not None:
        cap = carbon_cap_kg * (1.0 + CARBON_CAP_REL_TOLERANCE)
        problem += (
            pulp.LpAffineExpression([(x[k], carbon[k]) for k in range(n) if allowed[k]]) <= cap,
            "carbon_cap",
        )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    start = time.perf_counter()
    problem.solve(solver)
    solve_ms = (time.perf_counter() - start) * 1000.0

    status = _map_status(problem)
    chosen: List[int] = []
    problems: List[str] = []
    if status in SUCCESS_STATUSES:
        values = np.array([v.varValue if v.varValue is not None else 0.0 for v in x])
        for order_id, rows in candidates.groupby("order_id", sort=False).indices.items():
            picked = [k for k in rows if values[k] > SELECTION_THRESHOLD]
            if len(picked) != 1:
                problems.append(f"order {order_id} has {len(picked)} selected plants in the solver solution")
            else:
                chosen.append(int(picked[0]))
    objective = pulp.value(problem.objective) if status in SUCCESS_STATUSES else None
    return {
        "status": status,
        "chosen": chosen,
        "extraction_problems": problems,
        "objective_value": objective,
        "solve_ms": solve_ms,
        "pulp_status": pulp.LpStatus[problem.status],
        "pulp_solution_status": pulp.LpSolution[problem.sol_status],
        "num_variables": n,
        "num_constraints": len(problem.constraints),
    }


def _post_solve_checks(
    chosen_rows: pd.DataFrame,
    eff_caps: Mapping[str, float],
    carbon_cap_kg: Optional[float],
    require_on_time: bool,
) -> List[str]:
    """Sanity-check a solver solution against hard rules before reporting success."""
    issues: List[str] = []
    if not chosen_rows["eligible"].all():
        issues.append("solution assigns an order to an ineligible plant")
    loads = chosen_rows.groupby("plant")["qty"].sum()
    for plant, load in loads.items():
        if load > eff_caps[plant] + 1e-6:
            issues.append(f"plant {plant} load {load} exceeds effective capacity {eff_caps[plant]}")
    if require_on_time and chosen_rows["is_late"].fillna(False).any():
        issues.append("solution contains late orders although require_on_time=True")
    if carbon_cap_kg is not None:
        total = math.fsum(chosen_rows["total_carbon_kg"])
        if total > carbon_cap_kg * (1.0 + CARBON_CAP_REL_TOLERANCE) + 1e-6:
            issues.append(f"solution carbon {total} exceeds cap {carbon_cap_kg}")
    return issues


def _infeasibility_hints(
    candidates: pd.DataFrame,
    eff_caps: Mapping[str, float],
    require_on_time: bool,
    carbon_cap_kg: Optional[float],
) -> List[str]:
    """Necessary-condition checks that explain common causes of infeasibility."""
    hints: List[str] = []
    allowed = candidates["eligible"].copy()
    if require_on_time:
        allowed &= ~candidates["is_late"].fillna(False).astype(bool)
    orders = candidates.drop_duplicates("order_id")

    total_demand = int(orders["qty"].sum())
    total_cap = math.fsum(eff_caps.values())
    if total_demand > total_cap:
        hints.append(f"Total demand {total_demand} units exceeds total effective capacity {total_cap:g}")

    allowed_rows = candidates.loc[allowed]
    no_option = sorted(set(orders["order_id"]) - set(allowed_rows["order_id"]))
    if no_option:
        why = "eligible on-time plant" if require_on_time else "eligible plant"
        hints.append(f"Orders with no {why}: {', '.join(no_option)}")

    allowed_cap = allowed_rows["plant"].map(eff_caps)
    too_big = allowed_rows.assign(cap=allowed_cap).groupby("order_id").agg(qty=("qty", "first"), cap=("cap", "max"))
    too_big = too_big.loc[too_big["qty"] > too_big["cap"]]
    for order_id, row in too_big.iterrows():
        hints.append(f"Order {order_id} qty {int(row['qty'])} exceeds the largest allowed plant capacity {row['cap']:g}")

    for product, group in allowed_rows.groupby("product"):
        demand = int(group.drop_duplicates("order_id")["qty"].sum())
        cap = math.fsum(eff_caps[p] for p in group["plant"].unique())
        if demand > cap:
            hints.append(f"Product {product}: demand {demand} exceeds combined capacity {cap:g} of its allowed plants")

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

def _allocation_records(chosen_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "order_id", "plant", "product", "destination", "qty", "due",
        "unit_landed_cost", "total_cost", "unit_carbon_kg", "total_carbon_kg",
        "transit_days", "arrival_date", "is_late",
    ]
    return [to_json_compatible(rec) for rec in chosen_rows[cols].to_dict(orient="records")]


def _utilisation(chosen_rows: pd.DataFrame, eff_caps: Mapping[str, float]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    loads = chosen_rows.groupby("plant")["qty"].sum().to_dict()
    load_by_plant = {p: float(loads.get(p, 0)) for p in eff_caps}
    util: Dict[str, Any] = {}
    for plant, cap in eff_caps.items():
        load = load_by_plant[plant]
        util[plant] = load / cap if cap > 0 else (0.0 if load == 0 else None)
    return util, load_by_plant


def _empty_result(status: str, weights_used: Mapping[str, float], assumptions: List[str], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return to_json_compatible({
        "status": status,
        "allocation": [],
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
) -> Dict[str, Any]:
    """Allocate every order to exactly one plant by solving a binary MILP.

    Args:
        orders_df, plants_df, tariffs_df: Input tables (Annex B schema). Not mutated.
        planning_date: Start of the one-week planning period.
        weights: ``{"cost", "carbon"}`` (normalised to sum to 1) and optional
            ``"lateness"`` penalty per late order. Defaults to :data:`DEFAULT_WEIGHTS`.
        capacity_overrides: ``{plant: retained_fraction}``; ``{"B": 0.7}`` is a 30% loss.
        carbon_cut_pct: Require ``total_carbon_kg <= baseline * (1 - pct/100)``.
        require_on_time: Forbid late assignments (hard constraint).
        time_limit_seconds: Optional CBC time limit. A solution found but not
            proven optimal is returned with status ``"FEASIBLE"``.
        baseline_carbon_kg: Extension to the base interface. Pre-computed
            carbon baseline so several solves share one baseline; when omitted
            and ``carbon_cut_pct`` is set, the baseline is solved internally
            (see module docstring).

    Returns:
        JSON-compatible dict with ``status``, ``allocation``, ``total_cost``,
        ``total_carbon_kg``, ``total_carbon_t``, ``late_orders``,
        ``late_order_ids``, ``plant_utilisation`` (load / effective capacity),
        ``weights_used``, ``objective_value``, ``assumptions`` and
        ``diagnostics``. Non-success statuses (``INFEASIBLE``, ``NOT_SOLVED``,
        ``ERROR``...) carry an empty allocation and ``None`` totals.

    Raises:
        DataValidationError: invalid input tables or capacity overrides.
        ValueError: invalid weights, carbon_cut_pct, time limit or baseline.
    """
    t_start = time.perf_counter()
    weights_used, lateness_override = validate_weights(weights)
    cut = validate_carbon_cut_pct(carbon_cut_pct)
    time_limit = _validate_time_limit(time_limit_seconds)
    baseline = _validate_baseline(baseline_carbon_kg)
    if not isinstance(require_on_time, (bool, np.bool_)):
        raise ValueError(f"require_on_time must be a bool, got {require_on_time!r}")

    candidates = calculate_candidate_metrics(orders_df, plants_df, tariffs_df, planning_date)
    capacities = plant_capacities(prepare_plants(plants_df))
    eff_caps = effective_capacities(capacities, capacity_overrides)
    scales = compute_objective_scales(candidates)
    if lateness_override is not None:
        penalty = lateness_override
    else:
        penalty = default_lateness_penalty(scales, weights_used)
    preprocessing_ms = (time.perf_counter() - t_start) * 1000.0

    if lateness_override is not None:
        weights_used = {**weights_used, "lateness": lateness_override}

    assumptions = list(COST_MODEL_ASSUMPTIONS) + [
        f"Weights {weights_used['cost']:.0%} cost, {weights_used['carbon']:.0%} carbon (normalised to sum to 1)",
        "Cost and carbon are normalised by the sum of each order's cheapest / lowest-carbon eligible option",
        (
            f"Lateness penalty {penalty:g} per late order (custom; no lexicographic guarantee)"
            if lateness_override is not None
            else f"Lateness penalty {penalty:g} per late order: minimising late orders takes priority over cost and carbon"
        ),
        "Capacity overrides are retained fractions applied to the whole planning week",
    ]
    if require_on_time:
        assumptions.append("All orders must arrive on time (hard constraint)")
    assumptions.extend(candidates.attrs.get("warnings", []))

    diagnostics: Dict[str, Any] = {
        "solver": SOLVER_NAME,
        "planning_date": parse_planning_date(planning_date),
        "energy_cost_included": False,
        "original_capacity": capacities,
        "effective_capacity": eff_caps,
        "capacity_overrides": dict(capacity_overrides or {}),
        "require_on_time": bool(require_on_time),
        "carbon_cut_pct": cut,
        "lateness_penalty_per_order": penalty,
        **{k: scales[k] for k in ("cost_scale", "carbon_scale", "cost_scale_fallback", "carbon_scale_fallback")},
        "preprocessing_ms": preprocessing_ms,
    }

    carbon_cap: Optional[float] = None
    baseline_solve_ms = 0.0
    if cut is not None:
        if baseline is None:
            base_penalty = default_lateness_penalty(scales, BASELINE_WEIGHTS)
            base_run = _solve_model(candidates, eff_caps, BASELINE_WEIGHTS, base_penalty, scales, None, bool(require_on_time), time_limit)
            baseline_solve_ms = base_run["solve_ms"]
            if base_run["status"] not in SUCCESS_STATUSES or base_run["extraction_problems"]:
                diagnostics.update({
                    "baseline_status": base_run["status"],
                    "baseline_solve_ms": baseline_solve_ms,
                    "reason": "Carbon baseline plan could not be solved, so the carbon cap is undefined",
                    "infeasibility_hints": _infeasibility_hints(candidates, eff_caps, bool(require_on_time), None),
                })
                status = STATUS_INFEASIBLE if base_run["status"] == STATUS_INFEASIBLE else base_run["status"]
                return _empty_result(status, weights_used, assumptions, diagnostics)
            baseline = math.fsum(candidates.loc[base_run["chosen"], "total_carbon_kg"])
            diagnostics["baseline_source"] = "solved internally with BASELINE_WEIGHTS"
        else:
            diagnostics["baseline_source"] = "provided by caller"
        carbon_cap = baseline * (1.0 - cut / 100.0)
        diagnostics.update({"baseline_carbon_kg": baseline, "carbon_cap_kg": carbon_cap, "baseline_solve_ms": baseline_solve_ms})
        assumptions.append(
            f"Carbon cut {cut:g}% vs baseline {baseline:.3f} kg (plan with baseline weights "
            f"{BASELINE_WEIGHTS['cost']:.0%}/{BASELINE_WEIGHTS['carbon']:.0%}, same capacity and delivery settings, no cap)"
        )

    run = _solve_model(candidates, eff_caps, weights_used, penalty, scales, carbon_cap, bool(require_on_time), time_limit)
    diagnostics.update({k: run[k] for k in ("solve_ms", "pulp_status", "pulp_solution_status", "num_variables", "num_constraints")})
    diagnostics["total_ms"] = (time.perf_counter() - t_start) * 1000.0

    if run["status"] not in SUCCESS_STATUSES:
        if run["status"] == STATUS_INFEASIBLE:
            diagnostics["infeasibility_hints"] = _infeasibility_hints(candidates, eff_caps, bool(require_on_time), carbon_cap)
        return _empty_result(run["status"], weights_used, assumptions, diagnostics)

    chosen_rows = candidates.loc[run["chosen"]]
    issues = run["extraction_problems"] or _post_solve_checks(chosen_rows, eff_caps, carbon_cap, bool(require_on_time))
    if issues:
        diagnostics["errors"] = issues
        return _empty_result(STATUS_ERROR, weights_used, assumptions, diagnostics)

    utilisation, loads = _utilisation(chosen_rows, eff_caps)
    diagnostics["plant_load_units"] = loads
    late_rows = chosen_rows.loc[chosen_rows["is_late"].fillna(False).astype(bool)]
    total_carbon = math.fsum(chosen_rows["total_carbon_kg"])

    return to_json_compatible({
        "status": run["status"],
        "allocation": _allocation_records(chosen_rows),
        "total_cost": math.fsum(chosen_rows["total_cost"]),
        "total_carbon_kg": total_carbon,
        "total_carbon_t": kg_to_tonnes(total_carbon),
        "late_orders": int(len(late_rows)),
        "late_order_ids": late_rows["order_id"].tolist(),
        "plant_utilisation": utilisation,
        "weights_used": weights_used,
        "objective_value": run["objective_value"],
        "assumptions": assumptions,
        "diagnostics": diagnostics,
    })
