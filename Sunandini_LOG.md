# Activity Log: Sunandini (Optimisation / Solver Lead)

**Project:** Multi-Plant Order Allocation with Cost and Carbon
**Hackathon:** HCLTech Future Ready AI Engineer Hackathon, 2026-10-11
**Role:** Optimisation subsystem: cost/carbon model, MILP solver, independent validator, trade-off curves
**Built with:** Claude Code (AI coding assistant). Every result below was checked by running the code or tests; nothing is reported as passing without being run.

---

## 1. Summary

I built the deterministic optimisation layer in `app/solver/`. Every number the system reports (allocation, cost, carbon, lateness, trade-offs) comes from these modules, not from the LLM.

| Module | Purpose |
|---|---|
| `app/solver/cost_carbon.py` | Input validation, landed-cost and carbon formulas, arrival dates and lateness, capacity overrides, JSON conversion |
| `app/solver/milp.py` | Binary MILP allocation with PuLP and the bundled CBC solver |
| `app/solver/validator.py` | Independent Annex A plan checker; recomputes everything and never trusts solver status |
| `app/solver/tradeoff.py` | Carbon-cut curve (0–20%) and cost/carbon weight sweep |
| `tests/test_solver.py` | 79 pytest cases on small synthetic data |

---

## 2. Design decisions

### 2.1 Cost model
```
unit_landed_cost = (unit_cost + transport_cost_per_unit) * (1 + duty_pct / 100)
total_cost       = qty * unit_landed_cost
```
- **Energy monetary cost is not included.** The Annex B schema has no energy price per kWh, so no tariff is invented. This is documented in the module docstring and returned in every result's `assumptions`.
- `compute_unit_landed_cost(..., energy_cost_per_unit=None)` is an extension point: an energy component can be added once an authoritative price exists.
- Duty applies to production plus transport. It is never applied to carbon.

### 2.2 Carbon model
```
energy_carbon_kg_per_unit = energy_kwh_per_unit * carbon_kg_per_kwh
unit_carbon_kg            = energy_carbon_kg_per_unit + carbon_kg_per_unit
total_carbon_kg           = qty * unit_carbon_kg
```
All carbon is calculated in kg. Tonnes (`kg / 1000`) are used only for reporting.

### 2.3 Delivery
`arrival_date = planning_date + transit_days` and `is_late = arrival_date > due`. Arriving on the due date counts as on time.

### 2.4 MILP model (`milp.py`)
- **Variables:** binary `x[order, plant]`, one for every order × plant.
- **Constraints:**
  - each order is assigned to exactly one plant
  - ineligible plant–product pairs are fixed to 0
  - **one plant-wide capacity constraint per plant**, never one per product row
  - optional hard on-time rule (`require_on_time=True`)
  - optional carbon cap
- **Capacity overrides:** `{"B": 0.70}` means Plant B *keeps* 70% of its capacity (a 30% loss). Values must be in [0, 1].
- **Objective:** `w_cost * cost/cost_scale + w_carbon * carbon/carbon_scale + lateness_penalty * late_orders`
  - Default weights: **0.7 cost / 0.3 carbon**. Weights are validated (non-negative, finite, known keys) and normalised to sum to 1.
  - Scales = the sum of each order's cheapest (or lowest-carbon) eligible option. They depend only on the data, so every scenario shares the same scale. Zero sums fall back to 1.0, and this is flagged in `diagnostics`.
  - **Lateness penalty:** set automatically above the largest possible cost/carbon swing. The solver therefore never accepts an extra late order to save money or carbon. It can be overridden with `weights["lateness"]`.
- **Carbon baseline:** the plan solved with the default 0.7/0.3 weights, the same capacity and on-time settings, and no carbon cap. A carbon cut of X% means `total_carbon_kg <= baseline * (1 - X/100)`.
- **Results:** the status is `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `NOT_SOLVED` or `ERROR`. Failed runs return an empty allocation, `None` totals and `infeasibility_hints`, never invented numbers. After solving, the solution is re-checked against capacity, eligibility, the on-time rule and the carbon cap before it is reported as a success.
- **Additions to the requested interface:**
  - optional `baseline_carbon_kg` keyword, so a trade-off curve shares one baseline
  - optional `lateness` weight key, matching the API contract's `{cost, carbon, lateness}` weights

### 2.5 Validator (`validator.py`)
Recomputes everything from the input tables:
- each order is assigned exactly once (missing, duplicate, unknown and malformed entries are all detected)
- the plant exists and can make the product
- plant-wide load is within effective capacity (the excess is reported)
- arrival dates and lateness are recomputed
- reported qty, cost, carbon, arrival and late flag match the recomputed values

Late orders set `delivery_ok=False` and are listed with due date, arrival date and days late. They do not make `is_valid` false, because Annex A says to *flag* an unavoidable late order.

### 2.6 Trade-off (`tradeoff.py`)
- `generate_tradeoff_curve`: targets 0/5/10/15/20%, with **one baseline solved once** and shared by every point. Infeasible targets are returned explicitly with `None` values.
- `generate_weight_sweep`: carbon weight from 0.0 to 1.0. As documented in the module, a weighted-sum sweep only finds supported Pareto points.
- Neither function duplicates the model; both call `solve_allocation`.

### 2.7 Input validation
Informative `DataValidationError` for:
- missing columns or values
- null or duplicate order IDs
- non-positive or non-integer quantities
- negative costs, energy or carbon
- invalid dates or transit days
- duplicate plant–product or plant–destination rows
- missing transport routes for eligible plants
- products with no eligible plant
- **inconsistent repeated capacity for the same plant**

The caller's DataFrames are never modified, and no orders are silently dropped.

---

## 3. Chronological log

### Step 1: Repository inspection
- The repo had Aniket's orchestrator with **hard-coded stub tools** in `graph.py` (cost 1,840,000; extra cost 71,000; O-077). There was no `app/solver/` and no CSV data.
- Read the participant guide. Found a conflict: §2 and §4.1 mention an "energy price", but Annex B (the fixed schema) has none. Followed the instruction to exclude energy cost and to document the gap.
- Environment: Python 3.12, **PuLP 2.9.0** (CBC available), pandas 3.0.6, pytest 8.4.2.

### Step 2: `cost_carbon.py`
- Wrote the formulas, validation and candidate metrics. A smoke test gave O-077's numbers exactly: Plant A 17,400 (3 days late) and Plant C 18,200 (on time).
- **Commit `fa3ffb8`**, on branch `feature/solver-subsystem`.

### Step 3: `milp.py` + first tests
- O-077 data on its own: Plant C is chosen under default settings. Plant A is chosen only when the lateness penalty is set to 0, which shows the model drives the choice and nothing is hard-coded.
- Tests: 56 passed.

### Step 4: `validator.py`
- Tests: 65 solver tests passed, 78 in total with the orchestrator tests.
- **Commit `883c976`.** Fast-forward merged into `main`.

### Step 5: Push problem: "it's not committing"
- **Cause:** the commits existed locally, but Aniket had pushed 2 commits (`f38f88c`, `ae55b04`) after I cloned, so the push was rejected as non-fast-forward. Authentication was not the problem.
- **Fix:**
  - discarded my local `.pyc` changes (Aniket had removed those files from tracking and added a `.gitignore`)
  - merged `origin/main` (merge commit `0716e90`, no conflicts)
  - ran the tests: 81 passed
  - pushed
- **Lesson for the team:** always `git pull origin main` before `git push`.

### Step 6: `tradeoff.py` + benchmark
- Tests: **79 solver tests**, **95 total, all passing.**
- **Commit `1721eae`**, pushed to `origin/main`.

### Step 7: Benchmark (≈150 orders, 3 plants, 12 destinations)
The script was kept outside the repo, in the session scratchpad. The data was synthetic, seed 42. Times are measured after warm-up.

| Scenario | Status | Preprocessing p50 / p95 | Solve p50 / p95 |
|---|---|---|---|
| Default weights | OPTIMAL | 46 / 69 ms | 67 / 97 ms |
| Plant B loses 30% | OPTIMAL | 51 / 61 ms | 79 / 95 ms |
| 10% carbon cut (includes baseline solve) | OPTIMAL | 48 / 71 ms | 143 / 181 ms |
| All orders must be on time | INFEASIBLE | 45 / 61 ms | 50 / 69 ms |

- **End to end:** about 120–230 ms per call at the median, **so we cannot claim 50 ms.** The first call in each process takes another 300–500 ms while CBC starts.
- **Variation with data:** with tighter capacity, the default solve took a median of 134 ms and the weight sweep took 5.7 s (1.5 s with roomier capacity). The trade-off curve takes about 1.2 s.
- **Validity:** every successful plan passed the independent validator.

Behaviour worth knowing for Q&A:
- **Carbon cuts can increase late orders.** The cap is hard, while lateness is only a large penalty. In the benchmark, a 10% cut went from 10 to 18 late orders, and the curve reports this. With `require_on_time=True`, the cut returns infeasible instead.
- **Capacity loss can lower cost.** In one run, Plant B's loss made the plan 14,000 cheaper but 6 t dirtier. The weighted objective still got worse (1001.164 → 1001.184), as it should.
- **"Infeasible" carbon targets were genuine.** On tight data, the lowest-carbon plan possible was only 2.8% below the baseline, so cuts of 5% or more cannot be reached.

### Step 8: Orchestrator wiring (started, **not committed**, superseded)
- Started replacing the stub tools in `graph.py` with the real solver. Also added `app/orchestrator/data_source.py` (a CSV reader) and `app/solver/capacity_events.py`, which turns dated events into weekly retained fractions, pro-rated by days of overlap with concurrent losses added and capped at 100%. Edited `prompts.py` so a why-order brief no longer prints a fake "disturbance".
- Stopped partway. Shagun has since pushed the same integration (`app/orchestrator/tools.py`, DuckDB, loader, API), so **my uncommitted `graph.py`, `prompts.py` and `data_source.py` changes are redundant and would conflict.** Plan: stash them and adopt Shagun's version. `capacity_events.py` may still be useful if pro-rating partial weeks is needed.

### Step 9: Review against the problem statement
| Area | Status |
|---|---|
| R3 carbon, R4 solver, R5 delivery, R6 trade-off, Annex A validator | Done and tested |
| §4.4 traps: cheapest plant late (O-077), duty reversal, single-plant product, low-carbon small plant | Done by the model and tested |
| R2 energy in landed cost | Excluded (no price column in Annex B). Needs revisiting if the data has a price column |
| §6.2 "infeasible with closest option" | My solver returns no plan when infeasible. Shagun added a partial-plan fallback in `tools.py` |
| R7 defend default weights | 0.7/0.3 still needs a written rationale; the weight sweep gives the evidence |

### Step 10: Where are the CSV files?
- **None are in the shared repo or its history.** The solver tests build their own synthetic data in code. Before Shagun's work, the orchestrator "ran" only because the stubs returned fixed numbers.
- Found a dataset at `Downloads\new\data\`. A copy is at `data_syn/` in the repo folder, **untracked**. Its data card: `mpoa-synthetic` v1.0, seed 42, planning start 2026-10-12, 150 orders, 11 plant rows, 36 routes, 24 weekly carbon rows, 2 capacity events.
- **Three conflicts with the solver as built:**
  1. **Per-product capacity:** `plants.csv` has different `weekly_capacity` values per product for the same plant (A: 2600/2200/1800/1400). The solver rejects this with `inconsistent weekly_capacity`. **This needs a team decision:** is capacity per plant–product, per plant, or both?
  2. **Energy price:** `plants.csv` includes `energy_price_per_kwh`, so landed cost can include energy (R2).
  3. **Weekly carbon intensity:** `plant_carbon_weekly.csv` gives carbon intensity per plant per week (§4.4), and it differs slightly from `plants.csv`. We need to decide which source to use.

### Step 11: Built the DuckDB database file for Shagun
- Shagun's system reads a **DuckDB** file (default `data/allocation.duckdb`, configurable with `DUCKDB_PATH`), not a `.db` file.
- Installed `duckdb==1.5.6`, the version pinned in Shagun's `requirements.txt`, so the file format matches.
- Loaded `data_syn/` with Shagun's own loader from a clean checkout of `origin/main` (`94418bc`), so my uncommitted edits were not involved:
  ```
  python scripts/load_data.py --dir data_syn --capacity-events data_syn/drop/capacity_events.csv \
      --db data/allocation.duckdb --planning-start 2026-10-12 \
      --expect orders=150 --expect plants=11 --expect transport_duties=36 --expect capacity_events=2
  ```
- **Result: PASS.** orders 150, plants 11, transport_duties 36, capacity_events 2, with 0 rows rejected. All §4.5 checks pass. Informational: P5 can be made only in Plant C.
- Output: `data/allocation.duckdb` (about 3 MB). It is gitignored in Shagun's setup, so it has to be shared directly, not through Git.
- **Solving on it fails:** `tool_solve_allocation` raises `DataValidationError: inconsistent weekly_capacity` because of the per-product capacities (open item 1). The data loads correctly, but no plan can be produced until the capacity rule is decided and the solver updated.

### Step 12: Pulled Shagun's work, fixed a Windows bug
- Stashed my redundant orchestrator edits (`stash@{0}`, recoverable with `git stash pop`) and moved `data_source.py` out of the repo.
- Fast-forwarded to Shagun's `94418bc`.
- **9 tests errored on Windows.** `app/db/database.py` built ingest IDs from the clock (`ing-<timestamp>`). Windows' clock is coarse, so two quick ingests got the same ID and broke the primary key. Shagun's machine did not show this. **Fix (one line):** add a 6-character random suffix to the ID. This could also have hit `/ingest` during the judges' live data loading. After the fix: **115 passed.**

### Step 13: Capacity is now per plant-product (team decision)
- **Decision:** each `plants.csv` row's `weekly_capacity` limits that plant–product pair. Different products of the same plant may have different limits. Overrides such as `{"B": 0.7}` apply to every product row of that plant.
- `cost_carbon.py`:
  - new `capacity_limits()` returns one limit per (plant, product)
  - new `effective_capacity_limits()` and `nest_by_plant()`
  - `plant_capacities()` now returns plant **totals** (sum of the product limits), used only for reporting
  - the "inconsistent weekly_capacity" error is removed
  - `weekly_capacity` in the candidate table is now the row's own limit (NA for ineligible rows)
- `milp.py`: one capacity constraint per plant–product. `plant_utilisation` = plant load / plant total. New diagnostics: `effective_capacity_by_product`, `load_by_product` and `utilisation_by_product`.
- `validator.py`: capacity checked per plant–product. Violations now name the product.
- `app/orchestrator/tools.py` (Shagun's):
  - the partial "closest option" plan uses the same per-product limits
  - the why-order "spare capacity" explanation now uses the order's own product limit, not the plant total
  - per-product loads and limits are passed through the KPIs
- Tests: replaced the "inconsistent capacity rejected" test, updated 3 tests that assumed plant-wide limits, and added a test where products of one plant have different limits (enforced by the solver and confirmed by the validator). **116 passed.**

### Step 14: First end-to-end run on `data_syn` (DuckDB + full pipeline, `MOCK_LLM=true`)
The solve now works, but **only 49 of 150 orders are allocated**. The status is `PARTIAL` (the closest-option plan) and every plant is 100% utilised.
- **Root cause, a model/data mismatch rather than a bug:**
  - `data_syn` orders are due from **24 Oct to 21 Dec** (about 10 weeks), with total demand of **63,490 units**
  - the solver plans **one week**, where all plants together have **18,000 units/week**
  - the weekly capacity is therefore used once for 10 weeks of orders
- The same data also has weekly carbon intensity (`plant_carbon_weekly.csv`) and capacity events lasting 2–3 weeks, which suggests it was designed for **multi-week planning**.
- **Side effect to flag to Shagun:** `compare_plans` on two partial plans reported "extra cost −71,967.55, extra carbon −8.012 t, 0 orders moved" for Plant B losing 30%. The cost fell only because fewer orders were allocated (45 vs 49), so this comparison is misleading when the plans cover different order sets.
- Run times: allocation about 3.9 s; capacity-loss re-plan about 23.5 s (several partial-plan solves).
- **Decision needed:**
  - (a) multi-week model: assign each order to a plant *and* a production week, with capacity per week
  - (b) plan only the orders due in a horizon
  - (c) scale capacity by the number of weeks

---

## 4. Git history (my commits)

| Commit | Description |
|---|---|
| `fa3ffb8` | Add solver package and cost/carbon candidate metrics module |
| `883c976` | Add MILP allocation solver, independent plan validator and tests |
| `0716e90` | Merge `origin/main` (Aniket's orchestrator/router updates) |
| `1721eae` | Add cost/carbon trade-off curve and weight sweep with tests |

All are pushed to `origin/main`. The branch `feature/solver-subsystem` still exists locally.

---

## 5. Current state (at the time of writing)

- Local `main` = Shagun's `94418bc` plus my **uncommitted** changes:
  - `app/solver/cost_carbon.py`, `milp.py`, `validator.py` and `tests/test_solver.py`: per plant-product capacity
  - `app/orchestrator/tools.py`: per-product capacity in the partial plan and in the spare-capacity explanation
  - `app/db/database.py`: Windows ingest-ID fix
  - `Sunandini_LOG.md` (this file)
- **Not committed, not used by the pipeline:** `app/solver/capacity_events.py`, my converter from dated events to weekly fractions. Shagun's `tools.py` has its own event handling.
- **Untracked:** `data_syn/`. **Gitignored:** `data/allocation.duckdb`.
- **Tests: 116 passed.**
- **End to end on `data_syn`:** the pipeline runs, but only 49 of 150 orders are allocated, because 10 weeks of demand meets a one-week capacity (Step 14).

---

## 6. Open items

1. **Decide the planning horizon** for multi-week data like `data_syn` (Step 14). This is the main blocker for useful results.
2. **Add energy cost** to landed cost when `energy_price_per_kwh` is present (R2).
3. **Decide how weekly carbon intensity** (`plant_carbon_weekly.csv`) is used. This is tied to item 1.
4. **Pin `pulp==2.9.0`** in `requirements.txt`. Shagun found that 19 solver tests fail on PuLP 4.0.
5. **Make `compare_plans` fair for partial plans.** Comparing two plans that cover different order sets makes cost look lower when fewer orders are served.
6. **Write the rationale for the default 0.7/0.3 weights** (R7, 10 points), backed by weight-sweep results.
7. **Commit `data_syn/` or share it otherwise.** Shagun can rebuild the database with `python scripts/load_data.py --dir data_syn --capacity-events data_syn/drop/capacity_events.csv --planning-start 2026-10-12`.
8. **Raise with Aniket** (his files):
   - `router.py` falls back to `"O-077"` when no order ID is found
   - weights written in the request text are never applied
   - the writer records hard-coded token counts and hard-codes models outside the §5.1 list

---

## 7. WIP NOTE (committed unfinished, untested) - what to change and do next

**Done, not tested:**
- `app/solver/milp.py` rewritten as a **multi-week** model: `x[order, plant, week]`, weekly per plant-product capacity. New `solve_allocation` arguments: `capacity_events`, `carbon_weekly`, `horizon_weeks` and `allow_partial`. The partial mode returns status `PARTIAL` with `unassigned_orders`.
- `app/solver/cost_carbon.py`:
  - energy cost is included when `energy_price_per_kwh` is present (R2)
  - new `expand_to_weeks()`, `default_horizon_weeks()` and `prepare_carbon_weekly()`
  - the assumptions are built per run
- `app/solver/capacity_events.py`: `weekly_retained_fractions()` and `effective_weekly_limits()`, the shared capacity rule for the solver and the validator.

**Still to do:**
1. **`validator.py`:** use `effective_weekly_limits` and `expand_to_weeks`. Read `production_week` from each entry (default 0), check capacity per (plant, product, week), and accept `capacity_events`, `carbon_weekly` and `horizon_weeks`.
2. **`tradeoff.py`:** pass `capacity_events`, `carbon_weekly` and `horizon_weeks` through to `solve_allocation`.
3. **`app/orchestrator/tools.py`** (Shagun's):
   - pass dated events as `capacity_events`; keep `capacity_overrides` only for scenarios without dates
   - call `solve_allocation(..., allow_partial=True)` and **delete `solve_partial_allocation`**, which duplicates the model
   - add `production_week` and `production_week_start` to `_allocation_item` and to the validator entries
   - pass `plant_carbon_weekly` if the database has that table
4. **Database and loader (Shagun):** add an optional `plant_carbon_weekly` table so the weekly carbon intensity in §4.4 reaches the solver.
5. **`tests/test_solver.py`:** update for the multi-week output. Add tests for:
   - weekly capacity
   - an event covering only some weeks
   - weekly carbon intensity
   - energy cost
   - partial mode
6. **Run** `python -m pytest -q`, fix the failures, then re-run `data_syn` end to end (`DUCKDB_PATH=data/allocation.duckdb PLANNING_START=2026-10-12 MOCK_LLM=true`).
7. **Pin `pulp==2.9.0`** in `requirements.txt`.
