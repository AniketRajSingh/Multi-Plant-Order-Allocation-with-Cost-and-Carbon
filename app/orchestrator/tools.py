"""Deterministic tools called by the LangGraph nodes.

Every number these tools return is computed from the tables in DuckDB
(app.db.database) by the optimisation package (app.solver). Nothing here is
hard-coded or produced by an LLM.

Tools
-----
- tool_solve_allocation: MILP plan (app.solver.milp) + carbon-cut trade-off curve.
- tool_validate_plan:    independent Annex A check (app.solver.validator).
- tool_compare_plans:    re-solves the base plan and diffs it against the scenario plan.
- tool_diagnose_order:   per-plant cost / carbon / delivery / capacity reasons for one order.
"""

from __future__ import annotations

import math
import os
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import pulp

from app.db.database import Database, get_db
from app.solver.cost_carbon import (
    DataValidationError,
    calculate_candidate_metrics,
    capacity_limits,
    effective_capacities,
    effective_capacity_limits,
    nest_by_plant,
    plant_capacities,
    prepare_plants,
    to_json_compatible,
)
from app.solver.milp import SOLVER_NAME, SUCCESS_STATUSES, solve_allocation
from app.solver.tradeoff import DEFAULT_CARBON_CUT_TARGETS, generate_tradeoff_curve
from app.solver.validator import validate_plan


STATUS_DATA_ERROR = "DATA_ERROR"
STATUS_PARTIAL = "PARTIAL"  # infeasible as asked; best subset of orders allocated
PLANNING_HORIZON_DAYS = 7  # the solver plans one week


#: Known model limits, reported with every plan (Section 6.1 "limitations").
MODEL_LIMITATIONS: Tuple[str, ...] = (
    "Energy cost is excluded from landed cost: plants.csv has no energy price per kWh.",
    "One-week planning horizon; a capacity event overlapping the week applies to the whole week.",
    "Carbon intensity is constant over the planning week.",
    "Arrival = planning date + transit days; production lead time is not modelled.",
    "Duty rates and transport costs are held constant for the period.",
)


class ToolDataError(Exception):
    """Input data is missing or unusable; the message is safe to show the user."""


# ============================================================================
# INPUTS
# ============================================================================

def planning_date() -> date:
    value = os.getenv("PLANNING_START")
    return date.fromisoformat(value) if value else date.today()


def load_inputs(db: Optional[Database] = None) -> Dict[str, pd.DataFrame]:
    data = (db or get_db()).load_dataset()
    empty = [t for t in ("orders", "plants", "transport_duties") if data[t].empty]
    if empty:
        raise ToolDataError(
            f"No data loaded for {empty}. Load the CSVs with scripts/load_data.py or POST /ingest first."
        )
    return data


def resolve_capacity_overrides(
    scenario: Optional[Mapping[str, Any]],
    events: pd.DataFrame,
    start: date,
    use_loaded_events: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Turn a scenario and/or loaded capacity events into {plant: retained_fraction}.

    - scenario with plant + capacity_loss_pct: applied as given.
    - scenario with only event_id: that row of capacity_events.
    - use_loaded_events: every capacity event overlapping the planning week.
    Several losses on one plant keep the largest loss.
    """
    losses: Dict[str, float] = {}
    applied: List[Dict[str, Any]] = []

    def add(plant: str, pct: float, source: str, **extra: Any) -> None:
        plant = str(plant).strip().upper()
        losses[plant] = max(losses.get(plant, 0.0), float(pct))
        applied.append({"plant": plant, "capacity_loss_pct": float(pct), "source": source, **extra})

    scenario = scenario or {}
    if scenario.get("plant") and scenario.get("capacity_loss_pct") is not None:
        add(scenario["plant"], scenario["capacity_loss_pct"], "request")
    elif scenario.get("event_id"):
        match = events.loc[events["event_id"].astype(str) == str(scenario["event_id"])] if not events.empty else events
        if match.empty:
            raise ToolDataError(f"capacity event '{scenario['event_id']}' is not in the loaded capacity_events table")
        row = match.iloc[0]
        add(row["plant"], row["capacity_loss_pct"], "capacity_events", event_id=row["event_id"],
            from_date=row["from_date"], to_date=row["to_date"])

    if use_loaded_events and not events.empty:
        end = start + timedelta(days=PLANNING_HORIZON_DAYS - 1)
        for row in events.itertuples(index=False):
            if pd.Timestamp(row.from_date).date() <= end and pd.Timestamp(row.to_date).date() >= start:
                add(row.plant, row.capacity_loss_pct, "capacity_events", event_id=row.event_id,
                    from_date=row.from_date, to_date=row.to_date)

    overrides = {p: round(1.0 - pct / 100.0, 9) for p, pct in losses.items()}
    return overrides, to_json_compatible(applied)


def _solver_weights(weights: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    # The API's lateness weight (0-1) has no direct meaning in the MILP, which
    # minimises late orders first (lexicographic penalty). It is reported, not used.
    w = dict(weights or {})
    return {"cost": float(w.get("cost", 0.7)), "carbon": float(w.get("carbon", 0.3))}


# ============================================================================
# SOLVE
# ============================================================================

def tool_solve_allocation(
    weights: Optional[Mapping[str, Any]] = None,
    scenario: Optional[Mapping[str, Any]] = None,
    use_loaded_events: bool = False,
    include_tradeoff: bool = True,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """Solve the allocation on the loaded data and return it in the orchestrator's format."""
    data = load_inputs(db)
    start = planning_date()
    scenario = dict(scenario or {})
    overrides, applied = resolve_capacity_overrides(scenario, data["capacity_events"], start, use_loaded_events)
    solver_weights = _solver_weights(weights)
    require_on_time = bool(scenario.get("require_on_time", False))
    carbon_cut = scenario.get("carbon_cut_pct")

    tables = (data["orders"], data["plants"], data["transport_duties"])
    result = solve_allocation(
        *tables, start,
        weights=solver_weights,
        capacity_overrides=overrides or None,
        carbon_cut_pct=carbon_cut,
        require_on_time=require_on_time,
    )

    status = result["status"]
    unallocated: List[Dict[str, Any]] = []
    if status not in SUCCESS_STATUSES:
        hints = result["diagnostics"].get("infeasibility_hints", [])
        partial = solve_partial_allocation(*tables, start, solver_weights, overrides, require_on_time)
        if partial is None:
            return _no_plan(status, hints, result, applied, overrides, start, data)
        result, unallocated = partial
        result["diagnostics"]["infeasibility_hints"] = hints
        status = STATUS_PARTIAL

    tradeoff: List[Dict[str, Any]] = []
    if include_tradeoff and status in SUCCESS_STATUSES:
        curve = generate_tradeoff_curve(
            *tables, start, carbon_cut_targets=DEFAULT_CARBON_CUT_TARGETS,
            weights=solver_weights, capacity_overrides=overrides or None, require_on_time=require_on_time,
        )
        tradeoff = [
            {
                "carbon_cut_pct": p["carbon_cut_pct"],
                "cost": _r(p["cost"], 2),
                "total_carbon_t": _r(p["total_carbon_t"], 3),
                "late_orders": p["late_orders"],
                "extra_cost_vs_baseline": _r(p["extra_cost_vs_baseline"], 2),
                "solver_status": p["solver_status"],
            }
            for p in curve
        ]

    return {
        "status": status,
        "kpis": _kpis(result),
        "allocations": [_allocation_item(a) for a in result["allocation"]],
        "tradeoff": tradeoff,
        "unallocated_orders": unallocated,
        "infeasibility_hints": result["diagnostics"].get("infeasibility_hints", []),
        "capacity_overrides": overrides,
        "capacity_events_applied": applied,
        "planning_date": start.isoformat(),
        "weights_used": result["weights_used"],
        "assumptions": result["assumptions"],
        "limitations": list(MODEL_LIMITATIONS),
        "data_rows": {t: int(len(df)) for t, df in data.items()},
        "solve_time_ms": _r(result["diagnostics"].get("total_ms"), 1),
        "solver_name": SOLVER_NAME,
    }


def solve_partial_allocation(
    orders: pd.DataFrame,
    plants: pd.DataFrame,
    tariffs: pd.DataFrame,
    start: date,
    weights: Mapping[str, float],
    overrides: Mapping[str, float],
    require_on_time: bool,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Closest feasible plan when the full problem is infeasible.

    Each order may stay unassigned. The MILP first maximises the units
    allocated, then minimises late orders, then weighted cost and carbon.
    Returns (result in solve_allocation format, unallocated orders) or None.
    """
    try:
        cand = calculate_candidate_metrics(orders, plants, tariffs, start)
        plants_clean = prepare_plants(plants)
        caps = effective_capacities(plant_capacities(plants_clean), overrides or None)  # plant totals, reporting
        limits = effective_capacity_limits(capacity_limits(plants_clean), overrides or None)
    except DataValidationError:
        return None

    allowed = cand["eligible"].to_numpy(bool)
    late = cand["is_late"].fillna(False).to_numpy(bool)
    if require_on_time:
        allowed &= ~late
    rows = [k for k in range(len(cand)) if allowed[k]]
    if not rows:
        return None

    cost_scale = max(cand.loc[allowed, "total_cost"].sum(), 1.0)
    carbon_scale = max(cand.loc[allowed, "total_carbon_kg"].sum(), 1.0)
    total_qty = float(cand.drop_duplicates("order_id")["qty"].sum())
    n_orders = cand["order_id"].nunique()

    prob = pulp.LpProblem("partial_allocation", pulp.LpMinimize)
    x = {k: pulp.LpVariable(f"x_{k}", lowBound=0, upBound=1, cat="Binary") for k in rows}
    # Weights keep the priorities strictly ordered: units > lateness > cost/carbon.
    unit_w = (n_orders + 2) * 10.0 / max(total_qty, 1.0)
    prob += pulp.lpSum(
        x[k] * (
            -unit_w * float(cand.at[k, "qty"])
            + (1.0 if late[k] else 0.0)
            + (weights["cost"] * float(cand.at[k, "total_cost"]) / cost_scale
               + weights["carbon"] * float(cand.at[k, "total_carbon_kg"]) / carbon_scale) / 2.0
        )
        for k in rows
    )
    for order_id, idx in cand.groupby("order_id", sort=False).indices.items():
        prob += pulp.lpSum(x[k] for k in idx if k in x) <= 1
    # One limit per plant-product row (same capacity rule as solve_allocation).
    for key, idx in cand.groupby(["plant", "product"]).indices.items():
        terms = [float(cand.at[k, "qty"]) * x[k] for k in idx if k in x]
        if terms:
            prob += pulp.lpSum(terms) <= limits[key]

    try:
        import highspy
        solver = pulp.HiGHS(msg=False)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False)

    try:
        prob.solve(solver)
    except Exception:
        prob.solve(pulp.LpSolverDefault)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    chosen = [k for k in rows if (x[k].varValue or 0) > 0.5]
    chosen_rows = cand.loc[chosen]
    assigned = set(chosen_rows["order_id"])
    left = cand.drop_duplicates("order_id").loc[lambda d: ~d["order_id"].isin(assigned)]

    unallocated = []
    for o in left.itertuples(index=False):
        opts = cand.loc[(cand["order_id"] == o.order_id) & cand["eligible"]]
        unallocated.append(to_json_compatible({
            "order_id": o.order_id, "product": o.product, "qty": o.qty, "destination": o.destination,
            "due": o.due, "eligible_plants": opts["plant"].tolist(),
            "reason": "no remaining capacity for this product at any eligible plant"
            + (" that delivers on time" if require_on_time else ""),
        }))

    loads = chosen_rows.groupby("plant")["qty"].sum().to_dict()
    product_loads = chosen_rows.groupby(["plant", "product"])["qty"].sum().to_dict()
    late_rows = chosen_rows.loc[chosen_rows["is_late"].fillna(False).astype(bool)]
    total_carbon = math.fsum(chosen_rows["total_carbon_kg"])
    cols = ["order_id", "plant", "product", "destination", "qty", "due", "unit_landed_cost", "total_cost",
            "unit_carbon_kg", "total_carbon_kg", "transit_days", "arrival_date", "is_late"]
    result = to_json_compatible({
        "status": STATUS_PARTIAL,
        "allocation": chosen_rows[cols].to_dict(orient="records"),
        "total_cost": math.fsum(chosen_rows["total_cost"]),
        "total_carbon_kg": total_carbon,
        "total_carbon_t": total_carbon / 1000.0,
        "late_orders": int(len(late_rows)),
        "late_order_ids": late_rows["order_id"].tolist(),
        "plant_utilisation": {p: (loads.get(p, 0) / c if c > 0 else None) for p, c in caps.items()},
        "weights_used": dict(weights),
        "assumptions": cand.attrs.get("assumptions", []) + [
            "Full plan infeasible: closest plan maximises units allocated, then minimises late orders, then cost/carbon"
        ],
        "diagnostics": {
            "plant_load_units": loads,
            "effective_capacity": caps,
            "load_by_product": nest_by_plant({k: float(product_loads.get(k, 0)) for k in limits}),
            "effective_capacity_by_product": nest_by_plant(limits),
        },
    })
    return result, unallocated


def _no_plan(status, hints, result, applied, overrides, start, data) -> Dict[str, Any]:
    return {
        "status": status,
        "kpis": {},
        "allocations": [],
        "tradeoff": [],
        "unallocated_orders": [],
        "infeasibility_hints": hints,
        "capacity_overrides": overrides,
        "capacity_events_applied": applied,
        "planning_date": start.isoformat(),
        "weights_used": result.get("weights_used", {}),
        "assumptions": result.get("assumptions", []),
        "limitations": list(MODEL_LIMITATIONS),
        "data_rows": {t: int(len(df)) for t, df in data.items()},
        "solver_name": SOLVER_NAME,
    }


def _kpis(result: Mapping[str, Any]) -> Dict[str, Any]:
    diag = result.get("diagnostics", {})
    return {
        "total_cost": _r(result["total_cost"], 2),
        "total_carbon_t": _r(result["total_carbon_t"], 3),
        "total_carbon_kg": _r(result["total_carbon_kg"], 1),
        "late_orders": result["late_orders"],
        "late_order_ids": result["late_order_ids"],
        "orders_allocated": len(result["allocation"]),
        "plant_utilisation": {p: _r(u, 4) for p, u in result["plant_utilisation"].items()},
        "plant_load_units": diag.get("plant_load_units", {}),
        "effective_capacity": diag.get("effective_capacity", {}),
        "load_by_product": diag.get("load_by_product", {}),
        "effective_capacity_by_product": diag.get("effective_capacity_by_product", {}),
    }


def _allocation_item(a: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "order_id": a["order_id"],
        "plant": a["plant"],
        "product": a["product"],
        "qty": a["qty"],
        "destination": a["destination"],
        "due": a["due"],
        "cost": a["total_cost"],
        "carbon_kg": a["total_carbon_kg"],
        "arrival": a["arrival_date"],
        "on_time": not bool(a["is_late"]),
    }


def _r(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# ============================================================================
# VALIDATE
# ============================================================================

def tool_validate_plan(raw_solver_output: Mapping[str, Any], db: Optional[Database] = None) -> Dict[str, Any]:
    """Re-check the plan against Annex A from the raw tables, without trusting the solver."""
    allocations = raw_solver_output.get("allocations") or []
    if not allocations:
        return {
            "capacity_ok": None, "eligibility_ok": None, "delivery_ok": None, "is_valid": False,
            "violations": [], "warnings": [f"No plan to validate (solver status {raw_solver_output.get('status')})"],
        }
    data = load_inputs(db)
    entries = [
        {"order_id": a["order_id"], "plant": a["plant"], "qty": a["qty"], "total_cost": a["cost"],
         "total_carbon_kg": a["carbon_kg"], "arrival_date": a["arrival"], "is_late": not a["on_time"]}
        for a in allocations
    ]
    res = validate_plan(
        entries, data["orders"], data["plants"], data["transport_duties"],
        raw_solver_output.get("planning_date") or planning_date(),
        capacity_overrides=raw_solver_output.get("capacity_overrides") or None,
    )
    unallocated = {u["order_id"] for u in raw_solver_output.get("unallocated_orders") or []}
    violations, warnings = [], []
    for v in res["violations"]:
        if v["type"] == "late_delivery":
            warnings.append(v["message"])
        elif v["type"] == "missing_assignment" and v.get("order_id") in unallocated:
            warnings.append(f"{v['message']} (flagged infeasible: no capacity left)")
        else:
            violations.append(v["message"])
    return {
        "capacity_ok": res["capacity_ok"],
        "eligibility_ok": res["eligibility_ok"],
        "delivery_ok": res["delivery_ok"],
        "assignment_ok": res["assignment_ok"],
        "consistency_ok": res["consistency_ok"],
        "is_valid": res["is_valid"],
        "late_orders_count": res["late_orders_count"],
        "violations": violations,
        "warnings": warnings,
    }


# ============================================================================
# COMPARE (what-if)
# ============================================================================

def tool_compare_plans(
    scenario_output: Mapping[str, Any],
    weights: Optional[Mapping[str, Any]] = None,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """Solve the base plan (no capacity changes) and diff the scenario plan against it."""
    base = tool_solve_allocation(weights, scenario=None, include_tradeoff=False, db=db)
    applied = scenario_output.get("capacity_events_applied") or []
    if not base.get("allocations") or not scenario_output.get("allocations"):
        return {
            "status": "NOT_COMPARABLE",
            "base_status": base["status"],
            "scenario_status": scenario_output.get("status"),
            "capacity_changes": applied,
            "summary": "No comparison: one of the plans could not be solved.",
        }

    before = {a["order_id"]: a for a in base["allocations"]}
    after = {a["order_id"]: a for a in scenario_output["allocations"]}
    moved = []
    for oid, a in after.items():
        b = before.get(oid)
        if b and b["plant"] != a["plant"]:
            moved.append({
                "order_id": oid, "from_plant": b["plant"], "to_plant": a["plant"],
                "cost_delta": round(a["cost"] - b["cost"], 2),
                "carbon_delta_kg": round(a["carbon_kg"] - b["carbon_kg"], 1),
                "now_late": not a["on_time"] and b["on_time"],
            })
    dropped = sorted(set(before) - set(after))

    bk, sk = base["kpis"], scenario_output["kpis"]
    extra_cost = round(sk["total_cost"] - bk["total_cost"], 2)
    extra_carbon_t = round(sk["total_carbon_t"] - bk["total_carbon_t"], 3)
    late_delta = sk["late_orders"] - bk["late_orders"]
    changes = ", ".join(f"Plant {c['plant']} -{c['capacity_loss_pct']:g}%" for c in applied) or "none"
    return {
        "status": "OK",
        "capacity_changes": applied,
        "extra_cost": extra_cost,
        "extra_carbon_t": extra_carbon_t,
        "late_delta": late_delta,
        "orders_moved": len(moved),
        "orders_unallocated": dropped,
        "moved_order_details": moved,
        "base_kpis": {k: bk[k] for k in ("total_cost", "total_carbon_t", "late_orders")},
        "scenario_kpis": {k: sk[k] for k in ("total_cost", "total_carbon_t", "late_orders")},
        "plant_util_shifts": {
            p: {"before": bk["plant_utilisation"].get(p), "after": sk["plant_utilisation"].get(p)}
            for p in sorted(set(bk["plant_utilisation"]) | set(sk["plant_utilisation"]))
        },
        "summary": (
            f"Capacity change ({changes}) moved {len(moved)} orders; cost {extra_cost:+,.2f}, "
            f"carbon {extra_carbon_t:+.3f} t, late orders {late_delta:+d}"
            + (f", {len(dropped)} orders cannot be met" if dropped else "") + "."
        ),
    }


# ============================================================================
# DIAGNOSE (why was order X built in plant Y?)
# ============================================================================

def resolve_order_id(ref: Optional[str], order_ids: List[str]) -> Optional[str]:
    """Match 'O-077', 'o-077', '77' or '077' to a loaded order id."""
    if not ref:
        return None
    ref = str(ref).strip()
    by_upper = {o.upper(): o for o in order_ids}
    if ref.upper() in by_upper:
        return by_upper[ref.upper()]
    digits = re.sub(r"\D", "", ref)
    if digits:
        hits = [o for o in order_ids if re.sub(r"\D", "", o) and int(re.sub(r"\D", "", o)) == int(digits)]
        if len(hits) == 1:
            return hits[0]
    return None


def tool_diagnose_order(
    order_ref: Optional[str],
    plan: Mapping[str, Any],
    asked_plant: Optional[str] = None,
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """Explain one order's placement from per-plant cost, carbon, delivery and capacity."""
    data = load_inputs(db)
    order_id = resolve_order_id(order_ref, data["orders"]["order_id"].astype(str).tolist())
    if order_id is None:
        return {"status": "ORDER_NOT_FOUND", "order_ref": order_ref,
                "reasons": [f"Order '{order_ref}' is not in the loaded orders table."]}

    start = planning_date()
    cand = calculate_candidate_metrics(data["orders"], data["plants"], data["transport_duties"], start)
    rows = cand.loc[cand["order_id"] == order_id]
    assigned = next((a for a in plan.get("allocations") or [] if a["order_id"] == order_id), None)
    kpis = plan.get("kpis") or {}
    # Capacity is limited per plant-product, so spare capacity is for this order's product.
    loads = kpis.get("load_by_product") or {}
    caps = kpis.get("effective_capacity_by_product") or {}

    options = []
    for r in rows.itertuples(index=False):
        opt = {"plant": r.plant, "eligible": bool(r.eligible)}
        if r.eligible:
            days_late = (r.arrival_date - r.due).days
            spare = None
            cap = caps.get(r.plant, {}).get(r.product)
            if cap is not None:
                used = loads.get(r.plant, {}).get(r.product, 0)
                spare = cap - used + (r.qty if assigned and assigned["plant"] == r.plant else 0)
            opt.update({
                "cost": round(float(r.total_cost), 2), "carbon_kg": round(float(r.total_carbon_kg), 1),
                "unit_landed_cost": round(float(r.unit_landed_cost), 4), "duty_pct": float(r.duty_pct),
                "transit_days": int(r.transit_days), "arrival": r.arrival_date.date().isoformat(),
                "on_time": not bool(r.is_late), "days_late": max(days_late, 0),
                "spare_capacity_units": None if spare is None else round(float(spare), 1),
                "fits_in_spare_capacity": None if spare is None else bool(spare >= r.qty),
            })
        options.append(opt)

    eligible = [o for o in options if o["eligible"]]
    first = rows.iloc[0]
    reasons: List[str] = []
    cheapest = min(eligible, key=lambda o: o["cost"]) if eligible else None
    greenest = min(eligible, key=lambda o: o["carbon_kg"]) if eligible else None
    if len(eligible) == 1:
        reasons.append(f"Only Plant {eligible[0]['plant']} can make product {first['product']}.")
    chosen = next((o for o in eligible if assigned and o["plant"] == assigned["plant"]), None)
    if chosen:
        for o in eligible:
            if o is chosen:
                continue
            why = []
            if not o["on_time"] and chosen["on_time"]:
                why.append(f"arrives {o['days_late']} day(s) late")
            if o["fits_in_spare_capacity"] is False:
                why.append(f"has only {o['spare_capacity_units']:g} spare units for {first['qty']} ordered")
            if o["cost"] < chosen["cost"]:
                why.append(f"is cheaper by {chosen['cost'] - o['cost']:,.2f}")
            else:
                why.append(f"costs {o['cost'] - chosen['cost']:,.2f} more")
            if o["carbon_kg"] < chosen["carbon_kg"]:
                why.append(f"emits {chosen['carbon_kg'] - o['carbon_kg']:,.1f} kg less carbon")
            else:
                why.append(f"emits {o['carbon_kg'] - chosen['carbon_kg']:,.1f} kg more carbon")
            if o["duty_pct"] > chosen["duty_pct"]:
                why.append(f"pays {o['duty_pct']:g}% import duty")
            reasons.append(f"Plant {o['plant']} " + ", ".join(why) + ".")
        if chosen is cheapest and chosen is greenest:
            reasons.insert(0, f"Plant {chosen['plant']} is both the cheapest and the lowest-carbon option.")
        elif chosen is cheapest:
            reasons.insert(0, f"Plant {chosen['plant']} is the cheapest option.")
        elif chosen is greenest:
            reasons.insert(0, f"Plant {chosen['plant']} is the lowest-carbon option at the chosen weights.")
        if not chosen["on_time"]:
            reasons.append(f"No eligible plant with capacity left delivers on time; order is {chosen['days_late']} day(s) late.")
    if asked_plant and assigned and asked_plant.upper() != assigned["plant"]:
        reasons.insert(0, f"Note: the current plan builds {order_id} in Plant {assigned['plant']}, not Plant {asked_plant.upper()}.")

    return to_json_compatible({
        "status": "OK",
        "order_id": order_id,
        "product": first["product"], "qty": first["qty"], "destination": first["destination"], "due": first["due"],
        "assigned_plant": assigned["plant"] if assigned else None,
        "cheapest_plant": cheapest["plant"] if cheapest else None,
        "lowest_carbon_plant": greenest["plant"] if greenest else None,
        "options": options,
        "reasons": reasons or ["Order is not in the current plan (see unallocated orders)."],
    })
