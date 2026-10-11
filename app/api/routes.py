"""Section 6 API contract: POST /analyze, POST /ingest, GET /health, GET /trace/{run_id}.

The routes are deliberately thin. All numbers in an /analyze response are copied
from the orchestrator's tool outputs (solver, validator, comparator); this layer
only reshapes them into the contract and persists the run and its trace.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ComponentHealth,
    DataStoreHealth,
    ErrorResponse,
    HealthResponse,
    IngestMetadata,
    IngestResponse,
    LLMHealth,
    Recommendation,
    TraceResponse,
)
from app.db.database import Database, get_db
from app.orchestrator.graph import run_pipeline
from app.tracer.trace_logger import trace_logger


logger = logging.getLogger("api")
router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "openai": "gpt-6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.8-flash",
}
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


# ============================================================================
# POST /analyze
# ============================================================================

@router.post("/analyze", response_model=AnalyzeResponse, responses={500: {"model": ErrorResponse}})
def analyze(body: AnalyzeRequest, db: Database = Depends(get_db)) -> AnalyzeResponse:
    weights = body.weights.model_dump(exclude_none=True) if body.weights else None
    scenario = body.scenario.model_dump(exclude_none=True, mode="json") if body.scenario else None

    started = time.perf_counter()
    try:
        state = run_pipeline(body.request, weights=weights or None, scenario=scenario or None)
    except Exception as exc:
        logger.exception("pipeline failed for request %r", body.request)
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    trace = trace_logger.get_trace(state["run_id"])
    response = build_analyze_response(state, trace, latency_ms)

    try:
        db.save_run(response.run_id, body.request, response.answer_type, response.model_dump(mode="json"), latency_ms)
        if trace:
            db.save_trace(trace)
    except Exception:
        # The caller still gets the answer; the trace also lives in data/traces/.
        logger.exception("could not persist run %s", response.run_id)

    return response


def build_analyze_response(state: Dict[str, Any], trace: Optional[Dict[str, Any]], latency_ms: float) -> AnalyzeResponse:
    """Map the final LangGraph state onto the Section 6.1 response."""
    solver = state.get("raw_solver_output") or {}
    kpis = solver.get("kpis")
    diff = state.get("diff_output")
    unallocated = solver.get("unallocated_orders") or []
    weights = state.get("weights") or {}

    answer_type = state.get("answer_type") or "allocation"
    if answer_type == "allocation" and (solver.get("status") == "INFEASIBLE" or unallocated):
        answer_type = "infeasible"

    brief = state.get("brief_text") or None
    usage = state.get("token_usage") or {}

    return AnalyzeResponse(
        run_id=state["run_id"],
        answer_type=answer_type,
        summary=_summary(answer_type, kpis, diff, unallocated, brief),
        allocation=solver.get("allocations") or [],
        kpis=kpis,
        tradeoff=solver.get("tradeoff") or [],
        weights_used={k: float(v) for k, v in weights.items() if v is not None},
        validator=state.get("validation_result"),
        assumptions=_assumptions(weights, state.get("scenario")) + list(state.get("assumptions") or []),
        limitations=list(state.get("limitations") or []),
        recommendations=_recommendations(state.get("recommendations") or []),
        brief=brief,
        scenario_comparison=diff,
        unallocated_orders=unallocated,
        llm_calls=(trace or {}).get("llm_calls", usage.get("llm_calls", 0)),
        tokens=(trace or {}).get("tokens", usage.get("total_tokens", 0)),
        latency_ms=latency_ms,
    )


def _summary(
    answer_type: str,
    kpis: Optional[Dict[str, Any]],
    diff: Optional[Dict[str, Any]],
    unallocated: List[Dict[str, Any]],
    brief: Optional[str],
) -> str:
    if answer_type in ("refused", "clarification_needed") or not kpis:
        return (brief or "No result produced.").strip().split("\n\n")[0]
    if diff and "reasons" in diff:
        head = f"Order {diff['order_id']} is built in Plant {diff.get('assigned_plant')}." if diff.get("order_id") else ""
        return " ".join([head, *diff["reasons"]]).strip()

    parts = []
    cost, carbon, late = kpis.get("total_cost"), kpis.get("total_carbon_t"), kpis.get("late_orders")
    if cost is not None:
        parts.append(f"cost {cost:,.0f}")
    if carbon is not None:
        parts.append(f"carbon {carbon:g} t")
    if late is not None:
        parts.append(f"{late} late orders")
    text = "Plan: " + ", ".join(parts) + "." if parts else "Plan computed."

    if diff and "extra_cost" in diff:
        text += f" Versus base: {diff.get('extra_cost', 0):+,.0f} cost, {diff.get('extra_carbon_t', 0):+g} t carbon"
        if diff.get("orders_moved") is not None:
            text += f", {diff['orders_moved']} orders moved"
        text += "."
    if unallocated:
        text += f" {len(unallocated)} orders cannot be met."
    return text


def _assumptions(weights: Dict[str, Any], scenario: Optional[Dict[str, Any]]) -> List[str]:
    out = []
    if weights:
        out.append("Weights " + ", ".join(f"{float(v) * 100:g}% {k}" for k, v in weights.items() if v is not None))
    if scenario and scenario.get("plant"):
        text = f"Scenario: Plant {scenario['plant']} loses {scenario.get('capacity_loss_pct', '?')}% capacity"
        if scenario.get("from_date") or scenario.get("to_date"):
            text += f" from {scenario.get('from_date', 'start')} to {scenario.get('to_date', 'end of horizon')}"
        out.append(text)
    return out


def _recommendations(raw: List[Dict[str, Any]]) -> List[Recommendation]:
    out = []
    for item in raw:
        try:
            out.append(Recommendation.model_validate(item))
        except ValidationError as exc:
            logger.warning("dropping recommendation missing contract fields: %s", exc.errors())
    return out


# ============================================================================
# POST /ingest
# ============================================================================

@router.post("/ingest", response_model=IngestResponse, responses={422: {"model": IngestResponse}})
async def ingest(
    file: UploadFile = File(..., description="CSV in the Annex B schema"),
    metadata: Optional[str] = Form(
        None, description='JSON naming the table, e.g. {"table": "orders", "mode": "replace"}'
    ),
    table: Optional[str] = Form(None, description="Alternative to metadata.table"),
    mode: Optional[str] = Form(None, description="replace (default), append or upsert"),
    db: Database = Depends(get_db),
):
    try:
        meta_dict = json.loads(metadata) if metadata else {}
        if not isinstance(meta_dict, dict):
            raise ValueError("metadata must be a JSON object")
        if table:
            meta_dict.setdefault("table", table)
        if mode:
            meta_dict.setdefault("mode", mode)
        meta = IngestMetadata.model_validate(meta_dict)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid ingest metadata: {exc}") from exc

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    result = db.ingest_csv(meta.table, content, mode=meta.mode, source_name=file.filename or "upload")
    body = IngestResponse(**result.to_dict())
    status_code = 422 if body.status == "failed" else 200
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


# ============================================================================
# GET /health
# ============================================================================

@router.get("/health", response_model=HealthResponse)
def health(db: Database = Depends(get_db)) -> HealthResponse:
    try:
        counts = db.table_counts()
        empty = [t for t in ("orders", "plants", "transport_duties") if counts.get(t, 0) == 0]
        data_store = DataStoreHealth(
            status="degraded" if empty else "ok",
            detail=f"no rows in {empty}; load data with scripts/load_data.py or POST /ingest" if empty else None,
            path=db.path,
            tables=counts,
        )
    except Exception as exc:
        data_store = DataStoreHealth(status="down", detail=str(exc), path=db.path)

    llm = _llm_health()
    statuses = {data_store.status, llm.status}
    overall = "down" if "down" in statuses else "degraded" if "degraded" in statuses else "ok"
    return HealthResponse(status=overall, api=ComponentHealth(status="ok"), data_store=data_store, llm=llm)


def _llm_health() -> LLMHealth:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "unknown")
    mock = os.getenv("MOCK_LLM", "false").lower() == "true"

    if mock:
        return LLMHealth(status="ok", provider=provider, model=model, mock=True,
                         detail="MOCK_LLM=true: briefs use the deterministic grounded writer")
    if provider in PROVIDER_KEYS:
        has_key = bool(os.getenv(PROVIDER_KEYS[provider]))
        return LLMHealth(status="ok" if has_key else "degraded", provider=provider, model=model, mock=False,
                         detail=None if has_key else f"{PROVIDER_KEYS[provider]} not set")
    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=1.5) as resp:
                ok = resp.status == 200
            return LLMHealth(status="ok" if ok else "degraded", provider=provider, model=model, mock=False,
                             detail=f"reachable at {base}")
        except Exception as exc:
            return LLMHealth(status="degraded", provider=provider, model=model, mock=False,
                             detail=f"Ollama not reachable at {base}: {exc}")
    return LLMHealth(status="degraded", provider=provider, model=model, mock=False,
                     detail=f"unknown LLM_PROVIDER '{provider}'")


# ============================================================================
# GET /trace/{run_id}
# ============================================================================

@router.get(
    "/trace/{run_id}",
    response_model=TraceResponse,
    response_model_exclude_none=True,
    responses={404: {"model": ErrorResponse}},
)
def get_trace(run_id: str, db: Database = Depends(get_db)) -> TraceResponse:
    trace = trace_logger.get_trace(run_id) or db.get_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"no trace for run_id '{run_id}'")
    return TraceResponse.model_validate(trace)
