"""Unit tests for the optimisation subsystem (app.solver), using small synthetic data only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.solver.cost_carbon import (
    DataValidationError,
    calculate_candidate_metrics,
    compute_unit_landed_cost,
    effective_capacities,
)
from app.solver.milp import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    solve_allocation,
    validate_weights,
)
from app.solver.validator import validate_plan

PLANNING_DATE = "2026-10-12"


# ============================================================================
# FIXTURES / BUILDERS
# ============================================================================

def make_orders(rows):
    return pd.DataFrame(rows, columns=["order_id", "product", "qty", "destination", "due"])


def make_plants(rows):
    return pd.DataFrame(
        rows,
        columns=["plant", "product", "weekly_capacity", "unit_cost", "energy_kwh_per_unit", "carbon_kg_per_kwh"],
    )


def make_tariffs(rows):
    return pd.DataFrame(
        rows,
        columns=["plant", "destination", "transport_cost_per_unit", "transit_days", "duty_pct", "carbon_kg_per_unit"],
    )


@pytest.fixture
def o077_data():
    """O-077: 500 units to DE. Plant A 17,400 but 3 days late; Plant C 18,200 on time.

    Plant A is also the lower-carbon option, so only the lateness rule can push
    the order to Plant C.
    """
    orders = make_orders([["O-077", "P1", 500, "DE", "2026-10-19"]])
    plants = make_plants([
        ["A", "P1", 1000, 30.0, 1.0, 0.2],
        ["C", "P1", 1000, 30.0, 2.0, 0.4],
    ])
    tariffs = make_tariffs([
        ["A", "DE", 4.8, 10, 0.0, 0.5],  # arrives 2026-10-22, due 2026-10-19 -> 3 days late
        ["C", "DE", 6.4, 3, 0.0, 0.5],
    ])
    return orders, plants, tariffs


def allocation_map(result):
    return {a["order_id"]: a["plant"] for a in result["allocation"]}


# ============================================================================
# cost_carbon: formulas
# ============================================================================

def test_landed_cost_formula():
    orders = make_orders([["O-1", "P1", 10, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 20.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 5.0, 2, 10.0, 0.0]])
    row = calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE).iloc[0]
    assert row["unit_landed_cost"] == pytest.approx((20.0 + 5.0) * 1.10)
    assert row["total_cost"] == pytest.approx(10 * 27.5)


def test_landed_cost_energy_extension_point_is_opt_in():
    assert compute_unit_landed_cost(20.0, 5.0, 10.0) == pytest.approx(27.5)
    assert compute_unit_landed_cost(20.0, 5.0, 10.0, energy_cost_per_unit=5.0) == pytest.approx(33.0)


def test_energy_and_transport_carbon():
    orders = make_orders([["O-1", "P1", 100, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 1000, 1.0, 3.0, 0.5]])
    tariffs = make_tariffs([["A", "DE", 1.0, 2, 50.0, 0.25]])
    row = calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE).iloc[0]
    assert row["energy_carbon_kg_per_unit"] == pytest.approx(1.5)
    assert row["carbon_kg_per_unit"] == pytest.approx(0.25)
    assert row["unit_carbon_kg"] == pytest.approx(1.75)  # duty not applied to carbon
    assert row["total_carbon_kg"] == pytest.approx(175.0)


def test_arrival_date_and_lateness(o077_data):
    df = calculate_candidate_metrics(*o077_data, PLANNING_DATE).set_index("plant")
    assert df.loc["A", "arrival_date"] == pd.Timestamp("2026-10-22")
    assert df.loc["C", "arrival_date"] == pd.Timestamp("2026-10-15")
    assert bool(df.loc["A", "is_late"]) is True
    assert bool(df.loc["C", "is_late"]) is False
    assert df.loc["A", "total_cost"] == pytest.approx(17_400)
    assert df.loc["C", "total_cost"] == pytest.approx(18_200)


def test_arrival_on_due_date_is_not_late():
    orders = make_orders([["O-1", "P1", 1, "DE", "2026-10-14"]])
    plants = make_plants([["A", "P1", 10, 1.0, 0.0, 0.0]])
    tariffs = make_tariffs([["A", "DE", 0.0, 2, 0.0, 0.0]])
    row = calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE).iloc[0]
    assert bool(row["is_late"]) is False


def test_product_eligibility_flags():
    orders = make_orders([["O-1", "P1", 5, "DE", "2026-10-30"], ["O-2", "P2", 5, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 1.0, 1.0, 1.0], ["B", "P2", 100, 1.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 1, 0.0, 0.0], ["B", "DE", 1.0, 1, 0.0, 0.0]])
    df = calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE)
    assert len(df) == 4  # one row per order x plant
    eligible = {(r.order_id, r.plant) for r in df.itertuples() if r.eligible}
    assert eligible == {("O-1", "A"), ("O-2", "B")}
    assert df.loc[~df["eligible"], "total_cost"].isna().all()


def test_inputs_not_mutated(o077_data):
    copies = [df.copy(deep=True) for df in o077_data]
    calculate_candidate_metrics(*o077_data, PLANNING_DATE)
    solve_allocation(*o077_data, PLANNING_DATE)
    for original, copy in zip(o077_data, copies):
        pd.testing.assert_frame_equal(original, copy)


# ============================================================================
# cost_carbon: validation
# ============================================================================

def _base():
    orders = make_orders([["O-1", "P1", 5, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 1.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 1, 0.0, 0.0]])
    return orders, plants, tariffs


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda o, p, t: (o.drop(columns=["due"]), p, t), "missing required columns"),
        (lambda o, p, t: (o.assign(order_id=[None]), p, t), "missing values"),
        (lambda o, p, t: (pd.concat([o, o]), p, t), "duplicate order_id"),
        (lambda o, p, t: (o.assign(qty=[0]), p, t), "qty must be > 0"),
        (lambda o, p, t: (o.assign(qty=[2.5]), p, t), "whole numbers"),
        (lambda o, p, t: (o.assign(due=["not-a-date"]), p, t), "invalid dates"),
        (lambda o, p, t: (o, p.assign(unit_cost=[-1.0]), t), "unit_cost must be >= 0"),
        (lambda o, p, t: (o, p.assign(energy_kwh_per_unit=[-1.0]), t), "energy_kwh_per_unit"),
        (lambda o, p, t: (o, p.assign(carbon_kg_per_kwh=[None]), t), "missing values"),
        (lambda o, p, t: (o, pd.concat([p, p]), t), "duplicate plant-product"),
        (lambda o, p, t: (o, p, t.assign(transit_days=[-1])), "transit_days must be >= 0"),
        (lambda o, p, t: (o, p, t.assign(carbon_kg_per_unit=[-0.1])), "carbon_kg_per_unit"),
        (lambda o, p, t: (o, p, pd.concat([t, t])), "duplicate plant-destination"),
        (lambda o, p, t: (o.assign(product=["P9"]), p, t), "no plant can manufacture"),
    ],
)
def test_input_validation_errors(mutate, match):
    with pytest.raises(DataValidationError, match=match):
        calculate_candidate_metrics(*mutate(*_base()), PLANNING_DATE)


def test_missing_transport_combination():
    orders = make_orders([["O-1", "P1", 5, "FR", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 1.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 1, 0.0, 0.0]])
    with pytest.raises(DataValidationError, match="A -> FR"):
        calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE)


def test_inconsistent_plant_capacity_rejected():
    orders = make_orders([["O-1", "P1", 5, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 1.0, 1.0, 1.0], ["A", "P2", 200, 1.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 1, 0.0, 0.0]])
    with pytest.raises(DataValidationError, match="inconsistent weekly_capacity"):
        calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE)


def test_invalid_planning_date():
    with pytest.raises(DataValidationError, match="planning_date"):
        calculate_candidate_metrics(*_base(), "2026-13-45")


@pytest.mark.parametrize("bad", [{"A": 1.5}, {"A": -0.1}, {"A": "0.7"}, {"A": True}, {"Z": 0.5}, [("A", 0.5)]])
def test_invalid_capacity_overrides(bad):
    with pytest.raises(DataValidationError):
        effective_capacities({"A": 100}, bad)


def test_capacity_override_is_retained_fraction():
    assert effective_capacities({"A": 1000, "B": 1000}, {"B": 0.70}) == {"A": 1000.0, "B": 700.0}


# ============================================================================
# milp: weights and parameters
# ============================================================================

@pytest.mark.parametrize(
    "bad",
    [
        {"cost": -0.1, "carbon": 1.1},
        {"cost": 0.0, "carbon": 0.0},
        {"cost": "0.7", "carbon": 0.3},
        {"cost": True, "carbon": 0.3},
        {"cost": float("nan"), "carbon": 0.3},
        {"cost": 0.7},
        {"cost": 0.7, "carbon": 0.3, "speed": 1.0},
        [0.7, 0.3],
    ],
)
def test_invalid_weights_rejected(bad):
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_weights_default_and_normalisation():
    assert validate_weights(None) == ({"cost": 0.7, "carbon": 0.3}, None)
    assert validate_weights({"cost": 7, "carbon": 3}) == ({"cost": 0.7, "carbon": 0.3}, None)


@pytest.mark.parametrize("bad", [-1, 100.1, "10", float("inf")])
def test_invalid_carbon_cut_rejected(o077_data, bad):
    with pytest.raises(ValueError, match="carbon_cut_pct"):
        solve_allocation(*o077_data, PLANNING_DATE, carbon_cut_pct=bad)


# ============================================================================
# milp: model behaviour
# ============================================================================

def test_o077_selects_plant_c_under_default_settings(o077_data):
    result = solve_allocation(*o077_data, PLANNING_DATE)
    assert result["status"] == STATUS_OPTIMAL
    assert allocation_map(result) == {"O-077": "C"}
    assert result["total_cost"] == pytest.approx(18_200)
    assert result["late_orders"] == 0
    assert result["allocation"][0]["arrival_date"] == "2026-10-15"


def test_o077_cheaper_plant_wins_only_without_lateness_penalty(o077_data):
    result = solve_allocation(*o077_data, PLANNING_DATE, weights={"cost": 0.7, "carbon": 0.3, "lateness": 0.0})
    assert allocation_map(result) == {"O-077": "A"}
    assert result["late_order_ids"] == ["O-077"]


def test_import_duty_reverses_cost_ranking():
    # Plant A has cheaper production (10 vs 12) but a 50% duty into DE.
    orders = make_orders([["O-1", "P1", 100, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 1000, 10.0, 1.0, 1.0], ["B", "P1", 1000, 12.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 2.0, 2, 50.0, 0.0], ["B", "DE", 2.0, 2, 0.0, 0.0]])
    df = calculate_candidate_metrics(orders, plants, tariffs, PLANNING_DATE).set_index("plant")
    assert df.loc["A", "unit_cost"] < df.loc["B", "unit_cost"]
    assert df.loc["A", "unit_landed_cost"] == pytest.approx(18.0)
    assert df.loc["B", "unit_landed_cost"] == pytest.approx(14.0)
    result = solve_allocation(orders, plants, tariffs, PLANNING_DATE, weights={"cost": 1.0, "carbon": 0.0})
    assert allocation_map(result) == {"O-1": "B"}


def _two_plant_capacity_data():
    orders = make_orders([[f"O-{i}", "P1", 100, "DE", "2026-10-30"] for i in range(5)])
    plants = make_plants([
        ["A", "P1", 300, 10.0, 1.0, 1.0],
        ["A", "P2", 300, 10.0, 1.0, 1.0],  # second product row must not double plant A's capacity
        ["B", "P1", 500, 20.0, 1.0, 1.0],
    ])
    tariffs = make_tariffs([["A", "DE", 1.0, 2, 0.0, 0.0], ["B", "DE", 1.0, 2, 0.0, 0.0]])
    return orders, plants, tariffs


def test_exactly_one_assignment_per_order():
    data = _two_plant_capacity_data()
    result = solve_allocation(*data, PLANNING_DATE)
    ids = [a["order_id"] for a in result["allocation"]]
    assert sorted(ids) == sorted(data[0]["order_id"])
    assert len(ids) == len(set(ids))


def test_plant_wide_capacity_not_duplicated_across_product_rows():
    result = solve_allocation(*_two_plant_capacity_data(), PLANNING_DATE, weights={"cost": 1.0, "carbon": 0.0})
    counts = pd.Series(allocation_map(result)).value_counts().to_dict()
    assert counts == {"A": 3, "B": 2}  # cheap A filled to 300, not 600
    assert result["plant_utilisation"]["A"] == pytest.approx(1.0)
    assert result["plant_utilisation"]["B"] == pytest.approx(200 / 500)


def test_capacity_override_30pct_loss_replans():
    result = solve_allocation(
        *_two_plant_capacity_data(), PLANNING_DATE,
        weights={"cost": 1.0, "carbon": 0.0}, capacity_overrides={"A": 0.70},
    )
    assert result["status"] == STATUS_OPTIMAL
    assert result["diagnostics"]["effective_capacity"]["A"] == pytest.approx(210.0)
    counts = pd.Series(allocation_map(result)).value_counts().to_dict()
    assert counts == {"A": 2, "B": 3}


def test_capacity_loss_can_be_explicitly_infeasible():
    result = solve_allocation(*_two_plant_capacity_data(), PLANNING_DATE, capacity_overrides={"A": 0.0, "B": 0.5})
    assert result["status"] == STATUS_INFEASIBLE
    assert result["allocation"] == []
    assert result["total_cost"] is None and result["total_carbon_kg"] is None
    assert any("exceeds total effective capacity" in h for h in result["diagnostics"]["infeasibility_hints"])


def test_require_on_time_hard_constraint_infeasible():
    orders = make_orders([["O-1", "P1", 10, "DE", "2026-10-13"]])
    plants = make_plants([["A", "P1", 100, 1.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 5, 0.0, 0.0]])
    soft = solve_allocation(orders, plants, tariffs, PLANNING_DATE)
    assert soft["status"] == STATUS_OPTIMAL and soft["late_order_ids"] == ["O-1"]
    hard = solve_allocation(orders, plants, tariffs, PLANNING_DATE, require_on_time=True)
    assert hard["status"] == STATUS_INFEASIBLE
    assert hard["allocation"] == []


def _carbon_tradeoff_data():
    # Per order: A = cost 1000 / carbon 100 kg; B = cost 2000 / carbon 50 kg.
    # Under default weights A is preferred, so the baseline is all-A (400 kg).
    orders = make_orders([[f"O-{i}", "P1", 100, "DE", "2026-10-30"] for i in range(4)])
    plants = make_plants([["A", "P1", 1000, 9.0, 1.0, 1.0], ["B", "P1", 1000, 19.0, 0.5, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 2, 0.0, 0.0], ["B", "DE", 1.0, 2, 0.0, 0.0]])
    return orders, plants, tariffs


def test_carbon_cut_constraint_enforced():
    data = _carbon_tradeoff_data()
    base = solve_allocation(*data, PLANNING_DATE)
    assert base["total_carbon_kg"] == pytest.approx(400.0)  # all on A
    cut = solve_allocation(*data, PLANNING_DATE, carbon_cut_pct=20)
    assert cut["status"] == STATUS_OPTIMAL
    assert cut["diagnostics"]["baseline_carbon_kg"] == pytest.approx(400.0)
    assert cut["total_carbon_kg"] <= 400.0 * 0.80 + 1e-6
    assert cut["total_cost"] > base["total_cost"]


def test_carbon_cut_unreachable_is_infeasible():
    result = solve_allocation(*_carbon_tradeoff_data(), PLANNING_DATE, carbon_cut_pct=60)  # min is 50%
    assert result["status"] == STATUS_INFEASIBLE
    assert result["total_cost"] is None


def test_zero_cost_and_zero_carbon_normalisation():
    orders = make_orders([["O-1", "P1", 10, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 0.0, 0.0, 0.0], ["B", "P1", 100, 0.0, 0.0, 0.0]])
    tariffs = make_tariffs([["A", "DE", 0.0, 1, 0.0, 0.0], ["B", "DE", 0.0, 1, 0.0, 0.0]])
    result = solve_allocation(orders, plants, tariffs, PLANNING_DATE)
    assert result["status"] == STATUS_OPTIMAL
    assert result["total_cost"] == 0.0 and result["total_carbon_kg"] == 0.0
    diag = result["diagnostics"]
    assert diag["cost_scale"] == 1.0 and diag["cost_scale_fallback"] is True
    assert diag["carbon_scale"] == 1.0 and diag["carbon_scale_fallback"] is True


def test_result_is_json_compatible_and_complete(o077_data):
    result = solve_allocation(*o077_data, PLANNING_DATE)
    json.dumps(result)
    for key in (
        "status", "allocation", "total_cost", "total_carbon_kg", "total_carbon_t", "late_orders",
        "late_order_ids", "plant_utilisation", "weights_used", "objective_value", "assumptions", "diagnostics",
    ):
        assert key in result
    assert result["total_carbon_t"] == pytest.approx(result["total_carbon_kg"] / 1000)
    assert any("Energy monetary cost is NOT included" in a for a in result["assumptions"])
    for key in ("order_id", "plant", "qty", "total_cost", "total_carbon_kg", "arrival_date", "is_late"):
        assert key in result["allocation"][0]


# ============================================================================
# validator
# ============================================================================

def _violation_types(report):
    return {v["type"] for v in report["violations"]}


def test_validator_accepts_solver_plan():
    data = _two_plant_capacity_data()
    result = solve_allocation(*data, PLANNING_DATE)
    report = validate_plan(result["allocation"], *data, PLANNING_DATE)
    assert report["is_valid"] and report["capacity_ok"] and report["eligibility_ok"]
    assert report["delivery_ok"] and report["consistency_ok"]
    assert report["violations"] == []
    json.dumps(report)


def test_validator_accepts_full_solver_result_and_overrides():
    data = _two_plant_capacity_data()
    result = solve_allocation(*data, PLANNING_DATE, capacity_overrides={"A": 0.70})
    report = validate_plan(result, *data, PLANNING_DATE, capacity_overrides={"A": 0.70})
    assert report["is_valid"]
    assert report["effective_capacity"]["A"] == pytest.approx(210.0)


def test_validator_detects_capacity_violation_independently_of_solver():
    data = _two_plant_capacity_data()
    plan = [{"order_id": f"O-{i}", "plant": "A"} for i in range(5)]  # 500 units into a 300-unit plant
    report = validate_plan(plan, *data, PLANNING_DATE)
    assert not report["capacity_ok"] and not report["is_valid"]
    excess = [v for v in report["violations"] if v["type"] == "capacity_exceeded"][0]
    assert excess["plant"] == "A" and excess["excess"] == pytest.approx(200.0)


def test_validator_uses_capacity_overrides():
    data = _two_plant_capacity_data()
    plan = [{"order_id": f"O-{i}", "plant": "A" if i < 3 else "B"} for i in range(5)]
    assert validate_plan(plan, *data, PLANNING_DATE)["capacity_ok"]
    assert not validate_plan(plan, *data, PLANNING_DATE, capacity_overrides={"A": 0.70})["capacity_ok"]


def test_validator_detects_ineligible_and_unknown_plants():
    orders = make_orders([["O-1", "P1", 5, "DE", "2026-10-30"], ["O-2", "P2", 5, "DE", "2026-10-30"]])
    plants = make_plants([["A", "P1", 100, 1.0, 1.0, 1.0], ["B", "P2", 100, 1.0, 1.0, 1.0]])
    tariffs = make_tariffs([["A", "DE", 1.0, 1, 0.0, 0.0], ["B", "DE", 1.0, 1, 0.0, 0.0]])
    plan = [{"order_id": "O-1", "plant": "B"}, {"order_id": "O-2", "plant": "Z"}]
    report = validate_plan(plan, orders, plants, tariffs, PLANNING_DATE)
    assert not report["eligibility_ok"]
    assert {"ineligible_plant", "unknown_plant"} <= _violation_types(report)


def test_validator_detects_missing_duplicate_and_unknown_orders():
    data = _two_plant_capacity_data()
    plan = [
        {"order_id": "O-0", "plant": "A"},
        {"order_id": "O-0", "plant": "B"},
        {"order_id": "O-1", "plant": "A"},
        {"order_id": "O-999", "plant": "A"},
        {"plant": "A"},
    ]
    report = validate_plan(plan, *data, PLANNING_DATE)
    assert not report["assignment_ok"] and not report["is_valid"]
    types = _violation_types(report)
    assert {"duplicate_assignment", "unknown_order", "missing_assignment", "malformed_entry"} <= types
    missing = sorted(v["order_id"] for v in report["violations"] if v["type"] == "missing_assignment")
    assert missing == ["O-2", "O-3", "O-4"]


def test_validator_recomputes_delivery(o077_data):
    report = validate_plan([{"order_id": "O-077", "plant": "A"}], *o077_data, PLANNING_DATE)
    assert report["is_valid"]  # hard rules hold; lateness is flagged separately
    assert not report["delivery_ok"]
    assert report["late_orders_count"] == 1 and report["late_order_ids"] == ["O-077"]
    late = [v for v in report["violations"] if v["type"] == "late_delivery"][0]
    assert late["arrival_date"] == "2026-10-22" and late["due"] == "2026-10-19" and late["days_late"] == 3


def test_validator_flags_inconsistent_reported_values(o077_data):
    plan = [{"order_id": "O-077", "plant": "C", "qty": 400, "total_cost": 17_400.0,
             "total_carbon_kg": 1.0, "arrival_date": "2026-10-14", "is_late": True}]
    report = validate_plan(plan, *o077_data, PLANNING_DATE)
    assert not report["consistency_ok"]
    assert {"qty_mismatch", "total_cost_mismatch", "total_carbon_kg_mismatch",
            "arrival_mismatch", "lateness_mismatch"} <= _violation_types(report)


def test_validator_rejects_bad_override(o077_data):
    with pytest.raises(DataValidationError):
        validate_plan([], *o077_data, PLANNING_DATE, capacity_overrides={"C": 1.2})
