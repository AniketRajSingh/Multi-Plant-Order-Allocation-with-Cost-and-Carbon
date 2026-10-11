"""Independent plan validator (Annex A rules).

The validator never trusts solver output or status. It recomputes every
quantity it checks from the input tables using the shared formulas and
capacity rules in :mod:`app.solver.cost_carbon`:

* every input order is assigned exactly once (no missing, duplicate or
  unknown order IDs; no malformed entries);
* each assigned plant exists and can manufacture the order's product;
* assigned quantity per plant-product does not exceed that row's effective
  capacity (``weekly_capacity[plant, product] * retained_fraction[plant]``);
* arrival dates and lateness are recomputed from ``planning_date`` and
  ``transit_days``;
* reported ``qty``, ``total_cost``, ``total_carbon_kg``, ``arrival_date``
  and ``is_late`` (when present on an entry) match recomputed values.

Late deliveries are reported (``delivery_ok``, ``late_order_ids`` and
``late_delivery`` violations) but are kept separate from the hard rules in
``is_valid``, because Annex A allows an unavoidable late order to be flagged
rather than forbidden.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from app.solver.cost_carbon import (
    calculate_candidate_metrics,
    capacity_limits,
    effective_capacities,
    effective_capacity_limits,
    nest_by_plant,
    plant_capacities,
    prepare_plants,
    to_json_compatible,
)

#: Relative / absolute tolerance when comparing reported cost and carbon figures.
VALUE_REL_TOLERANCE: float = 1e-6
VALUE_ABS_TOLERANCE: float = 1e-6

#: Absolute tolerance on capacity loads (units).
CAPACITY_TOLERANCE: float = 1e-6


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=VALUE_REL_TOLERANCE, abs_tol=VALUE_ABS_TOLERANCE)


def _as_entries(allocation: Any) -> List[Mapping[str, Any]]:
    """Normalise a solver result, list of dicts or DataFrame into a list of entries."""
    if isinstance(allocation, pd.DataFrame):
        return allocation.to_dict(orient="records")
    if isinstance(allocation, Mapping) and "allocation" in allocation:
        allocation = allocation["allocation"]
    if not isinstance(allocation, Iterable) or isinstance(allocation, (str, bytes, Mapping)):
        raise TypeError(
            "allocation must be a list of {order_id, plant, ...} dicts, a DataFrame, or a solver result dict"
        )
    return list(allocation)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_plan(
    allocation: Any,
    orders_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    planning_date: Any,
    capacity_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate an allocation independently of whichever tool produced it.

    Args:
        allocation: List of entries with at least ``order_id`` and ``plant``
            (optionally ``qty``, ``total_cost``, ``total_carbon_kg``,
            ``arrival_date``, ``is_late`` for consistency checks), a DataFrame
            of such rows, or a :func:`app.solver.milp.solve_allocation` result.
        orders_df, plants_df, tariffs_df: Input tables (not mutated).
        planning_date: Planning start date used to recompute arrivals.
        capacity_overrides: ``{plant: retained_fraction}``, same meaning as the solver.

    Returns:
        JSON-compatible dict with ``is_valid``, ``assignment_ok``,
        ``capacity_ok``, ``eligibility_ok``, ``delivery_ok``,
        ``consistency_ok``, ``late_orders_count``, ``late_order_ids``,
        ``plant_load_units``, ``effective_capacity`` and ``plant_utilisation``
        (plant totals: load / sum of the plant's product limits),
        ``load_by_product``, ``effective_capacity_by_product``
        and ``violations`` (each a dict with ``type``, ``message`` and details).

    Raises:
        DataValidationError: if the input tables or capacity overrides are invalid.
        TypeError: if ``allocation`` has an unsupported shape.
    """
    entries = _as_entries(allocation)
    candidates = calculate_candidate_metrics(orders_df, plants_df, tariffs_df, planning_date)
    plants_clean = prepare_plants(plants_df)
    eff_limits = effective_capacity_limits(capacity_limits(plants_clean), capacity_overrides)
    eff_caps = effective_capacities(plant_capacities(plants_clean), capacity_overrides)

    by_pair = candidates.set_index(["order_id", "plant"])
    orders = candidates.drop_duplicates("order_id").set_index("order_id")
    violations: List[Dict[str, Any]] = []
    flags = {"assignment": True, "eligibility": True, "capacity": True, "consistency": True}

    def add(kind: str, flag: str, message: str, **details: Any) -> None:
        if flag:
            flags[flag] = False
        violations.append({"type": kind, "message": message, **details})

    seen: Dict[str, int] = {}
    accepted: List[Dict[str, Any]] = []
    for pos, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or _missing(entry.get("order_id")) or _missing(entry.get("plant")):
            add("malformed_entry", "assignment", f"Allocation entry {pos} lacks order_id or plant", entry_index=pos)
            continue
        order_id, plant = str(entry["order_id"]).strip(), str(entry["plant"]).strip()

        if order_id not in orders.index:
            add("unknown_order", "assignment", f"Order {order_id} is not in the input orders", order_id=order_id, plant=plant)
            continue
        seen[order_id] = seen.get(order_id, 0) + 1
        if seen[order_id] > 1:
            add("duplicate_assignment", "assignment", f"Order {order_id} is assigned more than once", order_id=order_id, plant=plant)
            continue
        if plant not in eff_caps:
            add("unknown_plant", "eligibility", f"Order {order_id} assigned to unknown plant {plant}", order_id=order_id, plant=plant)
            continue
        row = by_pair.loc[(order_id, plant)]
        if not bool(row["eligible"]):
            add(
                "ineligible_plant", "eligibility",
                f"Plant {plant} cannot manufacture product {row['product']} for order {order_id}",
                order_id=order_id, plant=plant, product=row["product"],
            )
            continue
        accepted.append({**row.to_dict(), "order_id": order_id, "plant": plant})

        if not _missing(entry.get("qty")) and float(entry["qty"]) != float(row["qty"]):
            add("qty_mismatch", "consistency", f"Order {order_id} reports qty {entry['qty']} but ordered {row['qty']}",
                order_id=order_id, plant=plant, reported=entry["qty"], expected=row["qty"])
        for field in ("total_cost", "total_carbon_kg"):
            if not _missing(entry.get(field)) and not _close(float(entry[field]), float(row[field])):
                add(f"{field}_mismatch", "consistency",
                    f"Order {order_id} at plant {plant} reports {field} {entry[field]} but recomputed {row[field]}",
                    order_id=order_id, plant=plant, reported=entry[field], expected=row[field])
        if not _missing(entry.get("arrival_date")):
            reported = pd.Timestamp(entry["arrival_date"]).normalize()
            if reported != row["arrival_date"]:
                add("arrival_mismatch", "consistency",
                    f"Order {order_id} reports arrival {reported.date()} but recomputed {row['arrival_date'].date()}",
                    order_id=order_id, plant=plant, reported=reported, expected=row["arrival_date"])
        if not _missing(entry.get("is_late")) and bool(entry["is_late"]) != bool(row["is_late"]):
            add("lateness_mismatch", "consistency",
                f"Order {order_id} reports is_late={bool(entry['is_late'])} but recomputed {bool(row['is_late'])}",
                order_id=order_id, plant=plant)

    for order_id in orders.index:
        if order_id not in seen:
            add("missing_assignment", "assignment", f"Order {order_id} is not assigned to any plant", order_id=order_id)

    assigned = pd.DataFrame(accepted)
    loads = {p: 0.0 for p in eff_caps}
    product_loads = {key: 0.0 for key in eff_limits}
    late_ids: List[str] = []
    if not assigned.empty:
        for (plant, product), qty in assigned.groupby(["plant", "product"])["qty"].sum().items():
            product_loads[(plant, product)] = float(qty)
            loads[plant] += float(qty)
        for row in assigned.loc[assigned["is_late"].astype(bool)].itertuples(index=False):
            late_ids.append(row.order_id)
            violations.append({
                "type": "late_delivery",
                "message": f"Order {row.order_id} at plant {row.plant} arrives {row.arrival_date.date()} after due {row.due.date()}",
                "order_id": row.order_id, "plant": row.plant, "due": row.due, "arrival_date": row.arrival_date,
                "days_late": int((row.arrival_date - row.due).days),
            })

    for (plant, product), load in product_loads.items():
        cap = eff_limits[(plant, product)]
        if load > cap + CAPACITY_TOLERANCE:
            add("capacity_exceeded", "capacity",
                f"Plant {plant} product {product} load {load:g} exceeds effective capacity {cap:g}",
                plant=plant, product=product, load=load, effective_capacity=cap, excess=load - cap)

    utilisation = {
        p: (loads[p] / cap if cap > 0 else (0.0 if loads[p] == 0 else None)) for p, cap in eff_caps.items()
    }
    delivery_ok = not late_ids
    return to_json_compatible({
        "is_valid": all(flags.values()),
        "assignment_ok": flags["assignment"],
        "capacity_ok": flags["capacity"],
        "eligibility_ok": flags["eligibility"],
        "delivery_ok": delivery_ok,
        "consistency_ok": flags["consistency"],
        "late_orders_count": len(late_ids),
        "late_order_ids": late_ids,
        "plant_load_units": loads,
        "effective_capacity": eff_caps,
        "plant_utilisation": utilisation,
        "load_by_product": nest_by_plant(product_loads),
        "effective_capacity_by_product": nest_by_plant(eff_limits),
        "violations": violations,
    })
