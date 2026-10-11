# Evaluation Report — Multi-Plant Order Allocation with Cost and Carbon

**Hackathon:** HCLTech Future Ready AI Engineer Hackathon  
**Role:** Frontend UI, Evaluation & QA Lead (Member 4)  
**Date:** 2026-10-11  
**Evaluation Script:** `scripts/run_eval.py`  

---

## 1. Evaluation Methodology

All tests are run against the live FastAPI backend via HTTP requests using the `requests` library. Each test sends a specific request to `POST /analyze`, `POST /ingest`, `GET /health`, or `GET /trace/{run_id}` and validates the response against expected business rules.

- **Test data:** The synthetic fixtures from `tests/fixtures.py` (8 orders, 3 plants, 9 transport routes, 1 capacity event) are loaded before testing.
- **Solver mode:** `MOCK_LLM=true` for deterministic brief generation; the solver (PuLP/CBC) always runs for real.
- **Comparison baseline:** The cost-focused allocation (weights: cost=1.0, carbon=0.0) serves as the cost baseline. The balanced allocation (cost=0.5, carbon=0.5) and carbon-focused allocation (cost=0.0, carbon=1.0) are compared against it.
- **Float tolerance:** Cost and carbon comparisons use a tolerance of 0.01.

---

## 2. Test Environment and Configuration

| Item | Value |
|---|---|
| Backend | FastAPI + Uvicorn |
| Solver | PuLP/CBC (MILP) |
| Database | DuckDB (in-memory for tests) |
| LLM Mode | `MOCK_LLM=true` (deterministic) |
| Python | 3.12 |
| API URL | `http://localhost:8000` |
| Planning Start | 2026-10-12 |

---

## 3. Number of Test Cases

| Category | Count |
|---|---|
| Deadline Conflict | 3 |
| Capacity Loss | 3 |
| Infeasible Requests | 2 |
| Weight Sensitivity | 3 |
| Prompt Injection | 2 |
| Valid Input | 1 |
| Invalid Input | 1 |
| Ineligible Plant | 1 |
| Capacity Validation | 1 |
| Missed Deadline | 1 |
| High Import Duty | 1 |
| Normal Allocation | 1 |
| Management Brief | 1 |
| Why-Order Diagnosis | 1 |
| Health Check | 1 |
| Trace Retrieval | 1 |
| **Total** | **24** |

---

## 4. Plan Validity

**Not measured** — run `python scripts/run_eval.py --api-url http://localhost:8000` to populate.

Plan validity measures the percentage of allocation responses where the validator confirms:
- `capacity_ok = True` (no plant exceeds its weekly capacity)
- `eligibility_ok = True` (every order is assigned to a plant that can make its product)
- `delivery_ok = True` (no orders are late, or late orders are explicitly acknowledged)

---

## 5. Cost and Carbon Comparison

| Configuration | Total Cost | Total Carbon (t) | Late Orders |
|---|---|---|---|
| Cost-focused (cost=1.0, carbon=0.0) | Not measured | Not measured | Not measured |
| Balanced (cost=0.5, carbon=0.5) | Not measured | Not measured | Not measured |
| Carbon-focused (cost=0.0, carbon=1.0) | Not measured | Not measured | Not measured |

**Expected pattern:** Cost-focused should have the lowest cost but highest carbon. Carbon-focused should have the lowest carbon but highest cost. Balanced should be in between.

---

## 6. Late-Order Count

| Metric | Value |
|---|---|
| Late orders (normal allocation) | Not measured |
| Late-order rate | Not measured |
| Orders with deadline conflicts | O-001, O-007 (Plant A → DE is 9 days transit) |

---

## 7. Re-Planning Quality

After Plant B loses 30% capacity (scenario test T-04):

| Metric | Value |
|---|---|
| Extra cost vs base plan | Not measured |
| Extra carbon vs base plan | Not measured |
| Change in late orders | Not measured |
| Orders reassigned to different plants | Not measured |

---

## 8. Brief Faithfulness

| Metric | Value |
|---|---|
| Verified numerical claims | Not measured |
| Mismatched claims | Not measured |
| Unverifiable claims | Not measured |

**Method:** The prompt injection tests (T-12, T-13) check whether the LLM uses solver-computed KPI values or adopts user-injected fake numbers. A brief that contains the injected value ($999.99 or 0.0t carbon) instead of the solver's value is a mismatch.

---

## 9. p50 and p95 Latency

| Metric | Value |
|---|---|
| p50 (median) latency | Not measured |
| p95 latency | Not measured |
| Number of requests measured | Not measured |

---

## 10. LLM Calls and Token Usage

| Metric | Value |
|---|---|
| Average LLM calls per request | Not measured |
| Average tokens per request | Not measured |

---

## 11. Failed Tests and Limitations

**Not measured** — will be populated after running the evaluation script.

### Known Limitations
1. Prompt injection detection is best-effort: we check whether the injected number appears in KPIs, but cannot verify every sentence in the brief.
2. Brief faithfulness checks only numerical values, not qualitative claims.
3. Latency measurements include network overhead when testing against a remote API.
4. The `MOCK_LLM=true` mode uses a deterministic brief writer, so prompt injection tests may behave differently with a real LLM.

---

## 12. Comparison of Configurations

| Dimension | Cost-Focused | Balanced | Carbon-Focused |
|---|---|---|---|
| Total cost | Not measured | Not measured | Not measured |
| Total carbon (t) | Not measured | Not measured | Not measured |
| Late orders | Not measured | Not measured | Not measured |
| Best for | Minimising cost | Trade-off | Minimising emissions |

---

## 13. Final Configuration Choice

**Not measured** — the final configuration will be chosen based on measured results.

**Decision criteria:**
1. Zero late orders (hard requirement)
2. Lowest total cost among configurations with zero late orders
3. Carbon emissions within acceptable range
4. All validator checks pass (capacity, eligibility, delivery)

> **Note:** All results marked "Not measured" will be populated when the evaluation script is run against a live backend. We never invent benchmark numbers.

---

## How to Run

```bash
# Start the backend
MOCK_LLM=true uvicorn app.main:app --host 0.0.0.0 --port 8000

# Load test data
python scripts/load_data.py --dir data/

# Run evaluation
python scripts/run_eval.py --api-url http://localhost:8000
```
