# Work log: Claude Code changes

Every change Claude Code makes in this repo is listed here, newest first, with what changed, why, and how it was checked.

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
