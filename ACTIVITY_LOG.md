# Multi-Plant Order Allocation with Cost and Carbon: Unified Team Activity Log

**Hackathon:** HCLTech Future Ready AI Engineer Hackathon (October 2026)  
**Project:** Multi-Plant Order Allocation with Cost and Carbon  
**Core Architecture:** Neuro-Symbolic Optimization Engine (PuLP MILP + LangGraph Multi-Agent Orchestrator + FastAPI + SQLite + Streamlit)

---

## Team Overview & Attribution Index

| Team Member | Subsystem Role | Key Deliverables & Modules |
|---|---|---|
| **Member 1 (Sunandini)** | **Optimization & Mathematical Solver Lead** | `app/solver/cost_carbon.py`, `app/solver/milp.py`, `app/solver/validator.py`, `app/solver/tradeoff.py`, `app/solver/capacity_events.py`, `tests/test_solver.py` |
| **Member 2 (Shagun)** | **Backend API & Database Lead** | `app/main.py`, `app/api/routes.py`, `app/api/schemas.py`, `app/db/database.py`, `scripts/load_data.py`, `tests/fixtures.py`, `tests/test_api.py`, `Dockerfile.api` |
| **Member 3 (Aniket Raj Singh)** | **GenAI & Multi-Agent Orchestration Lead** | `app/orchestrator/state.py`, `app/orchestrator/router.py`, `app/orchestrator/prompts.py`, `app/orchestrator/graph.py`, `app/orchestrator/llm.py`, `app/tracer/trace_logger.py`, SQLite persistence migration, HiGHS solver integration |
| **Member 4 (QA & Frontend Lead)** | **Frontend UI, Evaluation & QA Lead** | `frontend/app.py`, `Dockerfile.frontend`, `scripts/validate_data.py`, `scripts/run_eval.py`, `eval_report.md`, `DEMO_SCRIPT.md` |

---

## Section 1: Member 1 — Sunandini (Optimization & Solver Lead)

### 1.1 Objective & Summary
Built the deterministic optimization subsystem in `app/solver/`. Every physical allocation, cost, carbon calculation, lateness tracking, and trade-off curve originates from these modules rather than an LLM.

### 1.2 Mathematical Formulation & Design Decisions
1. **Landed Cost Model (`app/solver/cost_carbon.py`):**
   $$\text{unit\_landed\_cost} = (\text{unit\_cost} + \text{transport\_cost\_per\_unit}) \times (1 + \text{duty\_pct}/100)$$
   $$\text{total\_cost} = \sum (\text{qty} \times \text{unit\_landed\_cost})$$
   - *Energy monetary cost:* Excluded because Annex B schema omits `energy_price_per_kwh`. This assumption is explicitly documented and reported in all result outputs.
   - Tariffs apply to production plus transport, never to carbon.
2. **Carbon Emission Model:**
   $$\text{energy\_carbon} = \text{energy\_kwh\_per\_unit} \times \text{carbon\_kg\_per\_kwh}$$
   $$\text{unit\_carbon\_kg} = \text{energy\_carbon} + \text{transport\_carbon\_kg\_per\_unit}$$
   Calculated strictly in kg; reported in metric tonnes ($\text{kg} / 1000$).
3. **Delivery Feasibility:**
   $$\text{arrival\_date} = \text{planning\_date} + \text{transit\_days}, \quad \text{is\_late} = (\text{arrival\_date} > \text{due})$$
4. **Binary MILP Allocation (`app/solver/milp.py`):**
   - **Decision variables:** Binary $x[i, p] \in \{0, 1\}$ for each order $i$ and candidate plant $p$.
   - **Constraints:**
     - Exactly one plant assignment per order ($\sum_p x[i, p] = 1$, or $\le 1$ in partial mode).
     - Ineligible plant-product combinations set to 0.
     - Plant-product weekly capacity limits: $\sum_i \text{qty}[i] \cdot x[i, p] \le \text{effective\_capacity}[p, \text{prod}]$.
     - Optional hard deadline rule (`require_on_time=True`).
     - Optional carbon cap: $\text{total\_carbon\_kg} \le \text{baseline} \times (1 - \text{cut}/100)$.
   - **Objective Function:**
     $$\min \left( w_{\text{cost}} \frac{\text{Cost}}{\text{CostScale}} + w_{\text{carbon}} \frac{\text{Carbon}}{\text{CarbonScale}} + \text{LatenessPenalty} \cdot \text{LateOrders} + w_{\text{tardy}} \cdot \text{DaysLate} \right)$$
   - Objective scales correspond to lower bounds (cheapest / lowest-carbon eligible options), ensuring scale invariance across scenarios.
5. **Independent Annex A Validator (`app/solver/validator.py`):**
   - Recomputes all quantities from raw input tables independently. Never trusts solver status.
   - Verifies assignment uniqueness, product capability eligibility, plant physical capacity, and arrival date accuracy.
6. **Pareto Trade-Off Engine (`app/solver/tradeoff.py`):**
   - Generates 0%, 5%, 10%, 15%, 20% carbon-cut targets relative to a single, consistent baseline.

### 1.3 Step-by-Step Chronological Execution
- **Step 1 — Cost & Carbon Core:** Built `app/solver/cost_carbon.py` with strict schema validation and tolerance handling.
- **Step 2 — PuLP MILP Formulation:** Built `app/solver/milp.py` with normalized objective terms, lateness penalty lexicographic priority, and capacity overrides.
- **Step 3 — Annex A Validator:** Implemented `app/solver/validator.py` checking capacity, product eligibility, and arrival dates with separate `delivery_ok` reporting.
- **Step 4 — Multi-Objective Trade-Off:** Implemented `app/solver/tradeoff.py` computing Pareto frontier points with single-baseline consistency.
- **Step 5 — Unit Test Suite:** Developed `tests/test_solver.py` with 80 comprehensive test cases verifying Section 4.4 traps (O-077 plant switch, single-plant products, duty traps).
- **Step 6 — Multi-Week & Dated Events:** Added `app/solver/capacity_events.py` converting dated calendar disruptions into weekly retained-capacity fractions.

---

## Section 2: Member 2 — Shagun (Backend API & Database Lead)

### 2.1 Objective & Summary
Engineered the persistence, data ingestion, and REST API layers to satisfy the competition Section 6 contract: `POST /analyze`, `POST /ingest`, `GET /trace/{run_id}`, and `GET /health`.

### 2.2 System Architecture & Design Decisions
1. **FastAPI Contract Layer (`app/api/routes.py`, `app/api/schemas.py`):**
   - Implemented Pydantic v2 schemas for all request and response structures (`AnalyzeRequest`, `AnalyzeResponse`, `IngestResponse`, `HealthResponse`).
   - Integrated recommendation generator building actionable items with `action`, `rationale`, `evidence_refs`, `confidence`, and `requires_approval`.
2. **Persistence Architecture (`app/db/database.py`):**
   - Built thread-safe connection management with WAL mode on SQLite to prevent file locks between worker threads and CLI scripts.
   - Tables: `orders`, `plants`, `transport_duties`, `capacity_events`, `ingest_log`, `rejected_rows`, `runs`, `traces`.
   - Supports 3 ingestion modes: `replace` (atomic table swap), `append` (with duplicate key rejection), and `upsert` (natural key update).
3. **Section 4.5 Pre-Analysis Validation Suite:**
   - Embedded cross-table relational integrity checks into the database: verifying product coverage, transit lane completeness, positive capacity (`units/week`), due-date ordering, and grid carbon unit ranges.
4. **CLI Loader (`scripts/load_data.py`):**
   - Standalone judge loader supporting direct DB ingestion or remote ingestion via `POST /ingest`.

### 2.3 Step-by-Step Chronological Execution
- **Step 1 — Database Layer:** Designed SQLite DDL schemas, natural keys, and validation checks in `app/db/database.py`.
- **Step 2 — FastAPI Application:** Built `app/main.py` and `app/api/routes.py` with CORS middleware and lifecycle management.
- **Step 3 — CLI Loader Implementation:** Authored `scripts/load_data.py` supporting directory loading, mode flags, and text/JSON reports.
- **Step 4 — Section 4.4 Synthetic Fixtures:** Created `tests/fixtures.py` providing `demo_db()` with known business traps for offline testing.
- **Step 5 — Orchestrator Bridge:** Built `app/orchestrator/tools.py` bridging the orchestrator directly to the database layer and PuLP solver.
- **Step 6 — Section 6 Contract Compliance:** Added recommendation generation, solver health checks, and API integration tests (`tests/test_api.py`).

---

## Section 3: Member 3 — Aniket Raj Singh (GenAI & Orchestration Lead)

### 3.1 Objective & Summary
Architected the multi-agent orchestration layer, intent classification, zero-hallucination prompt generator, local Ollama integration, and Annex C audit logging.

### 3.2 Key Technical Achievements & Implementations
1. **LangGraph State Machine (`app/orchestrator/graph.py`):**
   - Built deterministic state graph: `router` $\rightarrow$ (`optimiser` / `diagnosis` / `clarification` / `refusal`) $\rightarrow$ `checker` $\rightarrow$ `writer` $\rightarrow$ `END`.
   - Bound Member 1 tools into the execution graph: `solve_allocation`, `validate_plan`, `compare_plans`, and `explain_order`.
   - Added `no_plan` node to gracefully handle empty data or systemic infeasibility.
2. **Intent & Answer Type Classifier (`app/orchestrator/router.py`):**
   - Regex-based intent classification routing queries into `allocation`, `scenario`, `why_order`, `brief`, `infeasible`, `clarification_needed`, or `refused`.
   - Entity extraction capturing target plants, disruption percentages, and order identifiers (`O-077`, `077`, `77`).
   - Prompt-injection defense intercepting constraint-bypass attempts (e.g. *"ignore capacity and force 300%"*) and routing to `refusal_node`.
3. **Zero-Hallucination Prompt Templates (`app/orchestrator/prompts.py`):**
   - Strict read-only executive brief template enforcing the triad: **Action**, **Rationale**, and **KPI Impact**.
   - Built-in verification audit (`verify_brief_grounding`) checking every numerical value against solver outputs, automatically falling back to verified deterministic text if discrepancies occur.
4. **Local Ollama LLM Client (`app/orchestrator/llm.py`):**
   - Integrated local endpoint (`http://192.168.3.173:11434`, model `jev-9b:latest`).
   - Implemented regex stripping of `<think>` reasoning tokens to ensure clean executive output.
   - Clean fallback to mock generator when running in offline or CI/CD test modes (`MOCK_LLM=true`).
5. **Annex C Trace Audit Logger (`app/tracer/trace_logger.py`):**
   - In-memory and persistent trace auditing matching the Annex C JSON schema for `GET /trace/{run_id}`.
   - Tracks agent name, tool invoked, input parameters, output data, latency (ms), and token breakdown.
6. **Cross-Platform Solver Compatibility:**
   - Resolved macOS arm64 CBC executable crash (`[Errno 86] Bad CPU type`) by integrating `highspy` / `pulp.HiGHS(msg=False)` with automatic CBC fallback.
7. **SQLite Persistence Architecture:**
   - Standard library `sqlite3` integration with zero external binary dependencies, thread-safe locking, and WAL mode.

---

## Section 4: Member 4 (Frontend UI, Evaluation & QA Lead)

### 4.1 Objective & Summary
Delivered the interactive user experience, automated evaluation framework, pre-ingestion validation tool, and hackathon presentation deliverables.

### 4.2 Key Technical Achievements & Implementations
1. **Streamlit Frontend Dashboard (`frontend/app.py`, `Dockerfile.frontend`):**
   - **Header:** 6 responsive KPI metric cards (Total Cost, Carbon, On-Time %, Plant A/B/C utilisation).
   - **Allocation Controls:** Natural language query input, cost/carbon weighting sliders, and run trigger.
   - **What-If Disruption Panel:** Real-time Plant B capacity loss slider (0%–50%) displaying side-by-side metric deltas and moved orders.
   - **Trade-Off Scatter Plot:** Interactive Plotly curve displaying landed cost vs. carbon reduction percentage.
   - **Interactive Order Table:** Filterable by Plant assignment and on-time status.
   - **Executive Brief Viewer:** Formatted markdown display of Action, Rationale, and KPI Impact recommendations.
   - **Trace Viewer:** Accordion interface pulling `GET /trace/{run_id}` showing agent execution steps and token counts.
   - **Data Ingest Tab:** Multipart CSV uploader supporting `replace`, `append`, and `upsert` modes with rejection logs.
2. **Automated 24-Case Evaluation Suite (`scripts/run_eval.py`):**
   - Evaluates 24 benchmark test cases across 14 categories: deadline conflicts, capacity losses, weight sensitivity, prompt injection defense, eligibility compliance, tariff penalties, and trace compliance.
   - Outputs comprehensive validity %, latency percentiles (p50/p95), and grounding accuracy.
3. **Standalone CSV Pre-Ingestion Validator (`scripts/validate_data.py`):**
   - Zero-dependency CLI validating files against Annex B schemas before database loading.
   - Performs 29 distinct checks across orders, plants, transport duties, and capacity events.
4. **Hackathon Presentation Deliverables:**
   - **`DEMO_SCRIPT.md`:** 10-minute minute-by-minute presentation plan covering problem statement, data ingestion, normal solve, deadline trap, capacity disruption, trade-off analysis, prompt injection defense, and trace inspector.
   - **`eval_report.md`:** Detailed benchmark methodology report.

---

## Section 5: Verification & System Health Summary

All subsystems have been integrated, verified, and cross-tested:
- `python3 -m unittest discover tests`: **49/49 PASSED**
- `python3 -m pytest tests/test_solver.py`: **80/80 PASSED**
- `python3 scripts/validate_data.py --dir data/sample`: **29/29 PASSED**
- **Total Test Suite:** **129 / 129 tests passing**
- SQLite database (`data/allocation.db`) active with 100% data integrity and passing all checks.
