# Work log: Claude Code changes

Every change Claude Code makes in this repo is listed here, newest first, with what changed, why, and how it was checked.

---

## 2026-10-11: Replace hard-coded orchestrator tools with Sunandini's solver + DuckDB data

**Asked for:** fetch Sunandini's commit and use the database instead of the hard-coded values.

### Pulled
- Merged `origin/main` (`90a39a3`). It brings 4 commits from Sunandini: `app/solver/` with `cost_carbon.py` (landed-cost and carbon formulas), `milp.py` (PuLP/CBC MILP), `validator.py` (independent Annex A checker) and `tradeoff.py` (carbon-cut curve and weight sweep), plus `tests/test_solver.py`.
- **Her commit has the solver but no data CSVs.** `data/allocation.duckdb` is still empty, so `/analyze` now answers "Cannot plan: no data loaded…" until data is loaded.

### Environment finding
- With PuLP 4.0, 19 of Sunandini's 79 tests fail: `LpVariable(cat=...)` was removed in 4.0. With **PuLP 2.9.0** all 79 pass. I installed 2.9.0 locally.
- **TODO (not committed):** add `pulp==2.9.0` to `requirements.txt`. The user asked me not to commit configuration files, so this is left for the team. The Docker image will not have the solver until it is added.

### Files changed
| File | Change |
|---|---|
| `app/orchestrator/tools.py` (new) | Real tools. `tool_solve_allocation` loads orders, plants and transport_duties from DuckDB and calls `solve_allocation` and `generate_tradeoff_curve` (0/5/10/15/20% carbon cut). It turns scenario capacity losses into retained fractions; a "re-plan" with no plant given applies the loaded `capacity_events` that overlap the planning week. `tool_validate_plan` calls Sunandini's validator on the raw tables. `tool_compare_plans` re-solves the base plan and diffs it (extra cost/carbon, moved orders, utilisation shifts). `tool_diagnose_order` gives per-plant cost, carbon, arrival, duty and spare capacity for "why order X in plant Y" questions; it accepts `O-077`, `77` or `077`. `solve_partial_allocation` is the closest-option plan when the full plan is infeasible: it maximises units allocated, then minimises late orders, then cost and carbon, and lists the unmet orders. |
| `app/orchestrator/graph.py` | Removed the 4 hard-coded tool functions (1,840,000 / 71,000 / O-077 values). The nodes now call `tools.py`. The trace now records the data rows used, capacity overrides, planning date, KPIs and solve time. Added a `no_plan` node: when data is missing or no plan exists, it returns `clarification_needed` or `infeasible` instead of a brief full of zeros. The diagnosis node now solves the plan and uses the order id the router extracted. |
| `app/orchestrator/prompts.py` | Rewrote the deterministic brief. It now handles scenario, why-order and infeasible answers separately; before, a why-order answer printed "extra cost $0.00". Signed deltas, moved-order list, per-plant options. Removed the claim "carbon factors based on regional grid averages", which the data does not support. |
| `app/api/schemas.py` | Added `why_order` to `AnswerType`. Aniket's router emits it, so without this `/analyze` would fail validation on "why" questions. |
| `app/api/routes.py` | The summary for why-order answers is built from the diagnosis reasons. |
| `app/db/database.py` | Added `set_db()` so tests can point the tools at a fixture database. |
| `tests/fixtures.py` (new) | Small synthetic pack with the Section 4.4 traps: cheapest plant late (O-001, O-007), single-plant product (O-003/P2), duty trap (A→US 40%), low-carbon low-capacity Plant C, and capacity event E-1 (B −30%). |

### Verification
- `python -m pytest tests -q`: **108 passed, 2 failed.**
- The 2 failures are `test_orchestrator.py::TestGraphPipeline::test_run_scenario_pipeline` and `test_run_why_order_pipeline`. These are Aniket's tests: they assert the old hard-coded output (`extra_cost == 71000`) and run against the empty database. **Next step:** point them at `tests/fixtures.demo_db()` with `set_db()` and `PLANNING_START=2026-10-12`, and assert real computed values. The user stopped this step and asked for a commit.
- Not yet checked end to end through `/analyze` with loaded data.

---

## 2026-10-11: Commit and pull from origin/main

- Committed the API/DB/loader/Docker work as 7 commits (`8e4867f`..`61e4c32`).
- Pulled `origin/main`, which had 2 new commits from the team (`f38f88c` orchestrator/router update, `ae55b04` ACTIVITY_LOG.md and traces). **No data CSVs were on the remote yet**, so there is nothing to load into DuckDB.
- Merge conflict in `.gitignore` (both sides added it). Resolved by keeping both sets of rules.
- After the merge: `python -m unittest tests.test_api tests.test_orchestrator` passes **31/31**. The test run rewrote some `data/traces/*.json` files; I reset them to their committed state.

---

## 2026-10-11: API, DuckDB store, loader, Docker

**Asked for:** FastAPI app for the Section 6 contract, Pydantic v2 schemas, DuckDB persistence, a CLI loader for judges, Docker packaging, and an initialised database.

### Files added
| File | What it does |
|---|---|
| `app/api/schemas.py` | Pydantic v2 models. API side: `AnalyzeRequest/Response`, `Weights`, `Scenario`, `IngestResponse`, `HealthResponse`, `TraceResponse`, `Recommendation`. Data side: one row model per Annex B table (`OrderRow`, `PlantRow`, `TransportDutyRow`, `CapacityEventRow`) that every CSV row is checked against. |
| `app/db/database.py` | DuckDB layer. Tables: the 4 Annex B tables, plus `ingest_log`, `rejected_rows`, `runs`, `traces`. Ingest modes are `replace`, `append` and `upsert`. Also runs the Section 4.5 cross-table validation checks. |
| `app/api/routes.py` | `POST /analyze` (calls `run_pipeline`, maps state to the 6.1 response, saves the run and trace), `POST /ingest` (multipart CSV plus JSON `{"table", "mode"}`), `GET /health` (API, data store, LLM), `GET /trace/{run_id}`. |
| `app/main.py` | FastAPI app. Creates the DB schema on startup and sets up CORS. |
| `scripts/load_data.py` | Judge loader (Section 4.7). Takes `--dir` or per-table flags, plus `--mode`, `--api-url`, `--planning-start`, `--expect orders=150`, `--init-only`, `--check-only` and `--json`. Prints the validation report. Exits 1 on failure. |
| `Dockerfile.api`, `docker-compose.yml` | API container on port 8000 with a health check. `./data` is mounted so the DuckDB file persists. `host.docker.internal` reaches Ollama on the host. The UI service is commented out until `Dockerfile.ui` exists. |
| `requirements.txt`, `.env.example`, `.gitignore`, `.dockerignore` | Pinned deps; env template (`LLM_PROVIDER`, `LLM_MODEL`, `MOCK_LLM`, keys, `PLANNING_START`); ignores `.env`, `__pycache__` and `*.duckdb`. |
| `tests/test_api.py`, `tests/__init__.py`, `scripts/__init__.py` | 16 tests covering ingest rejects, aliases, append/upsert, validation checks, the loader CLI and the API contract. The `__init__.py` files were needed because an installed `tests` package was shadowing the folder. |

### Design decisions
- **Short-lived DuckDB connections.** DuckDB allows one writer process per file. Without this, the CLI loader would fail with a lock error while the API is running.
- **Bad rows are rejected, not dropped silently.** Each rejected row gets a row number and a reason. These are returned by `/ingest` and stored in `rejected_rows`.
- **A failed file never wipes a table.** If an upload has no valid rows in `replace` mode, the existing data stays.
- **Extra CSV columns are kept** (Annex B allows them). Common header variants are mapped to the schema names (`quantity→qty`, `due_date→due`, …) so a late-drop schema change still loads.
- **Routes do no maths.** Every number in `/analyze` is copied from the orchestrator's tool outputs.

### Verification
- `python -m unittest tests.test_api tests.test_orchestrator`: **28/28 pass**.
- `python scripts/load_data.py --init-only` created `data/allocation.duckdb` (gitignored) with empty tables.
- Live smoke test with `uvicorn` and `MOCK_LLM=true`: `/health` returned `degraded` (no data loaded yet, as expected) and the `/analyze` scenario request returned `answer_type=scenario`.
- `docker compose config` is valid. **The image was not built:** Docker Desktop was not running.
- **Not finished:** a check that the CLI loader can write while the API is running. The user stopped that step.

### Open issues for the team (not changed: these files belong to others)
1. **Disqualification risk:** the tools in `app/orchestrator/graph.py` (`tool_solve_allocation`, `tool_compare_plans`, `tool_diagnose_order`) return hard-coded numbers (1,840,000 / 71,000 / O-077). They do not read the DuckDB tables yet. Section 10 lists hard-coded answers as a disqualification condition. The solver should read its tables with `get_db().load_dataset()`.
2. `trace_logger.seed_default_sample_traces()` inserts fabricated sample traces. Replace them with real run traces.
3. `router_node` never applies weights written in the request text (e.g. "50% cost 50% carbon"). `create_initial_state` always fills in default weights, so the `not state.get("weights")` check is always false.
4. `writer_node` ignores `LLM_PROVIDER`/`LLM_MODEL` and hard-codes Gemini or OpenAI models. The token counts it records are also hard-coded.
5. Running the orchestrator tests rewrites files under `data/traces/`.
