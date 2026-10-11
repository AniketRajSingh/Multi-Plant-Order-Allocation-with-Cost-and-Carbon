"""Cost / carbon / delivery trade-off curves built on :func:`app.solver.milp.solve_allocation`.

Two views are provided, both re-using the single MILP model (no duplicate
optimisation logic):

* :func:`generate_tradeoff_curve` - epsilon-constraint sweep over carbon-cut
  targets (default 0, 5, 10, 15, 20 %). The carbon baseline is solved
  **once** (see ``app.solver.milp`` for its definition) and passed to every
  point, so all targets are measured against the identical baseline.
* :func:`generate_weight_sweep` - weighted-sum sweep of the carbon weight
  from 0.0 to 1.0 (cost weight = 1 - carbon weight).

Limitation: a weighted-sum sweep only finds *supported* Pareto points (those
on the convex hull of the cost/carbon frontier). Non-supported Pareto-optimal
plans can lie between sweep points and are never returned, and several
weights may map to the same plan. The carbon-cut (epsilon-constraint) curve
does not have this limitation for the targets it evaluates.

Infeasible or unsolved points are returned explicitly with their solver
status and ``None`` for cost and emissions - no values are invented.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from app.solver.milp import (
    BASELINE_WEIGHTS,
    SUCCESS_STATUSES,
    solve_allocation,
    validate_carbon_cut_pct,
)

#: Default carbon-reduction targets (percent of baseline carbon).
DEFAULT_CARBON_CUT_TARGETS: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0)

#: Default carbon weights for the weighted-sum sweep (0.0 .. 1.0 in steps of 0.1).
DEFAULT_CARBON_WEIGHTS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))


def _validate_sequence(values: Any, name: str) -> List[Any]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of numbers, got {values!r}")
    return list(values)


def _point_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract reportable metrics; non-success points carry ``None`` values."""
    ok = result["status"] in SUCCESS_STATUSES
    return {
        "solver_status": result["status"],
        "cost": result["total_cost"] if ok else None,
        "total_carbon_kg": result["total_carbon_kg"] if ok else None,
        "total_carbon_t": result["total_carbon_t"] if ok else None,
        "late_orders": result["late_orders"] if ok else None,
        "late_order_ids": result["late_order_ids"] if ok else [],
        "plant_utilisation": result["plant_utilisation"] if ok else {},
        "solve_ms": result["diagnostics"].get("solve_ms"),
    }


def generate_tradeoff_curve(
    orders_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    planning_date: Any,
    carbon_cut_targets: Sequence[float] = DEFAULT_CARBON_CUT_TARGETS,
    weights: Optional[Mapping[str, Any]] = None,
    capacity_overrides: Optional[Mapping[str, Any]] = None,
    require_on_time: bool = False,
    time_limit_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Solve one plan per carbon-cut target against a shared baseline.

    Args:
        orders_df, plants_df, tariffs_df, planning_date: As for ``solve_allocation``.
        carbon_cut_targets: Percent reductions in ``[0, 100]``.
        weights: Objective weights used at every point (default 0.7 / 0.3).
        capacity_overrides, require_on_time, time_limit_seconds: Passed through
            unchanged to every solve, including the baseline.

    Returns:
        One dict per target, in input order, with ``carbon_cut_pct``,
        ``solver_status``, ``cost``, ``total_carbon_kg``, ``total_carbon_t``,
        ``late_orders``, ``late_order_ids``, ``plant_utilisation``,
        ``solve_ms``, ``baseline_carbon_kg``, ``carbon_cap_kg``,
        ``baseline_cost`` and ``extra_cost_vs_baseline``. Infeasible points
        have ``None`` metrics. If the baseline itself cannot be solved, every
        point reports the baseline's status and a ``reason``.

    Raises:
        ValueError: on invalid targets or weights.
        DataValidationError: on invalid input tables or capacity overrides.
    """
    targets = [validate_carbon_cut_pct(t) for t in _validate_sequence(carbon_cut_targets, "carbon_cut_targets")]
    if any(t is None for t in targets):
        raise ValueError("carbon_cut_targets must not contain None")

    common = dict(
        capacity_overrides=capacity_overrides,
        require_on_time=require_on_time,
        time_limit_seconds=time_limit_seconds,
    )
    baseline = solve_allocation(orders_df, plants_df, tariffs_df, planning_date, weights=BASELINE_WEIGHTS, **common)

    if baseline["status"] not in SUCCESS_STATUSES:
        return [
            {
                "carbon_cut_pct": t,
                **_point_metrics(baseline),
                "solve_ms": None,
                "baseline_carbon_kg": None,
                "carbon_cap_kg": None,
                "baseline_cost": None,
                "extra_cost_vs_baseline": None,
                "reason": f"Baseline plan status {baseline['status']}: carbon cap undefined",
            }
            for t in targets
        ]

    baseline_carbon = baseline["total_carbon_kg"]
    baseline_cost = baseline["total_cost"]
    points: List[Dict[str, Any]] = []
    for target in targets:
        result = solve_allocation(
            orders_df, plants_df, tariffs_df, planning_date,
            weights=weights, carbon_cut_pct=target, baseline_carbon_kg=baseline_carbon, **common,
        )
        metrics = _point_metrics(result)
        cost = metrics["cost"]
        points.append({
            "carbon_cut_pct": target,
            **metrics,
            "baseline_carbon_kg": baseline_carbon,
            "carbon_cap_kg": baseline_carbon * (1.0 - target / 100.0),
            "baseline_cost": baseline_cost,
            "extra_cost_vs_baseline": None if cost is None else cost - baseline_cost,
        })
    return points


def generate_weight_sweep(
    orders_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    planning_date: Any,
    carbon_weights: Sequence[float] = DEFAULT_CARBON_WEIGHTS,
    capacity_overrides: Optional[Mapping[str, Any]] = None,
    require_on_time: bool = False,
    time_limit_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Weighted-sum sweep: solve once per carbon weight ``w`` with cost weight ``1 - w``.

    See the module docstring: this does not necessarily find every
    Pareto-optimal plan.

    Returns:
        One dict per weight with ``carbon_weight``, ``cost_weight``,
        ``solver_status``, ``cost``, ``total_carbon_kg``, ``total_carbon_t``,
        ``late_orders``, ``late_order_ids``, ``plant_utilisation`` and ``solve_ms``.

    Raises:
        ValueError: if any carbon weight is not a finite number in ``[0, 1]``.
    """
    sweep = _validate_sequence(carbon_weights, "carbon_weights")
    clean: List[float] = []
    for w in sweep:
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(float(w)) or not 0.0 <= float(w) <= 1.0:
            raise ValueError(f"carbon weights must be finite numbers in [0, 1], got {w!r}")
        clean.append(float(w))

    points: List[Dict[str, Any]] = []
    for w in clean:
        result = solve_allocation(
            orders_df, plants_df, tariffs_df, planning_date,
            weights={"cost": 1.0 - w, "carbon": w},
            capacity_overrides=capacity_overrides,
            require_on_time=require_on_time,
            time_limit_seconds=time_limit_seconds,
        )
        points.append({"carbon_weight": w, "cost_weight": 1.0 - w, **_point_metrics(result)})
    return points
