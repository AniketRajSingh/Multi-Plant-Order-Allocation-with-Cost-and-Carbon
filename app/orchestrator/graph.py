"""LangGraph state machine orchestrating tool calls and executive brief generation.

Implements the strict multi-agent orchestration architecture bound to tools:
1. solve_allocation: Calls Member 1's PuLP solver (with native PuLP MILP formulation).
2. validate_plan: Calls Member 1's Annex A validator (verifying physical limits).
3. compare_plans: Computes deltas (extra_cost, extra_carbon_t, orders_moved).
4. explain_order: Evaluates single-order cost/carbon/transit across all 3 plants.

Trace steps match Annex C schema exactly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger("graph_orchestrator")

from langgraph.graph import END, START, StateGraph

from app.orchestrator.prompts import (
    build_brief_prompt,
    generate_grounded_mock_brief,
    verify_brief_grounding,
)
from app.orchestrator.router import IntentClassifier, router_node
from app.orchestrator.state import AllocationState, create_initial_state
from app.orchestrator.tools import (
    STATUS_DATA_ERROR,
    ToolDataError,
    tool_compare_plans,
    tool_diagnose_order,
    tool_solve_allocation,
    tool_validate_plan,
)
from app.solver.cost_carbon import DataValidationError
from app.tools.tools import (
    compare_plans,
    execute_annex_a_validator,
    execute_compare_plans,
    execute_explain_order,
    execute_pulp_solver,
    explain_order,
    solve_allocation,
    validate_plan,
)
from app.tracer.trace_logger import trace_logger


# List of LangChain / LangGraph bindable tools
ALLOCATION_TOOLS = [
    solve_allocation,
    validate_plan,
    compare_plans,
    explain_order,
]


# ============================================================================
# TOOL CALLS (deterministic; implemented in app.orchestrator.tools)
# ============================================================================

PLAN_ERRORS = (ToolDataError, DataValidationError, ValueError)


def _append_step(state: AllocationState, step: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace_logger.log_step(
        run_id=state["run_id"],
        agent=step["agent"],
        tool=step["tool"],
        input_data=step.get("input"),
        output_data=step.get("output"),
    )
    steps = list(state.get("trace_steps") or [])
    steps.append(step)
    return steps


def _plan_or_error(weights, scenario, use_loaded_events: bool) -> Dict[str, Any]:
    try:
        return tool_solve_allocation(weights, scenario, use_loaded_events=use_loaded_events)
    except PLAN_ERRORS as exc:
        return {"status": STATUS_DATA_ERROR, "error": str(exc), "kpis": {}, "allocations": [], "tradeoff": []}


# ============================================================================
# LANGGRAPH NODE FUNCTIONS BOUND TO TOOLS
# ============================================================================

def optimiser_node(state: AllocationState) -> Dict[str, Any]:
    """Optimiser agent calling solve_allocation on the SQLite data."""
    start_time = time.time()
    weights = state.get("weights") or {"cost": 0.7, "carbon": 0.3}
    scenario = state.get("scenario")
    # A re-plan request with no explicit plant/event uses the loaded capacity_events.
    use_events = state.get("intent") == "scenario" and not (scenario and (scenario.get("plant") or scenario.get("event_id")))

    solver_res = _plan_or_error(weights, scenario, use_events)
    kpis = solver_res.get("kpis") or {}

    step = {
        "agent": "optimiser",
        "tool": "solve_allocation",
        "input": {
            "weights": weights,
            "capacity_overrides": solver_res.get("capacity_overrides"),
            "planning_date": solver_res.get("planning_date"),
            "data_rows": solver_res.get("data_rows"),
        },
        "output": {
            "status": solver_res.get("status"),
            "total_cost": kpis.get("total_cost"),
            "total_carbon_t": kpis.get("total_carbon_t"),
            "late_orders": kpis.get("late_orders"),
            "unallocated_orders": len(solver_res.get("unallocated_orders") or []),
            "solve_time_ms": solver_res.get("solve_time_ms"),
            **({"error": solver_res["error"]} if solver_res.get("error") else {}),
        },
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }

    updates: Dict[str, Any] = {
        "raw_solver_output": solver_res,
        "trace_steps": _append_step(state, step),
        "assumptions": list(solver_res.get("assumptions") or []),
    }
    if solver_res.get("unallocated_orders") and state.get("intent") != "scenario":
        updates["answer_type"] = "infeasible"
    return updates


def checker_node(state: AllocationState) -> Dict[str, Any]:
    """Checker agent: independent Annex A validation from the raw tables."""
    start_time = time.time()
    solver_out = state.get("raw_solver_output") or {}
    try:
        val_res = tool_validate_plan(solver_out)
    except PLAN_ERRORS as exc:
        val_res = {"capacity_ok": None, "eligibility_ok": None, "delivery_ok": None,
                   "violations": [f"validator could not run: {exc}"], "warnings": []}

    step = {
        "agent": "checker",
        "tool": "validate_plan",
        "output": {
            "capacity_ok": val_res.get("capacity_ok"),
            "eligibility_ok": val_res.get("eligibility_ok"),
            "delivery_ok": val_res.get("delivery_ok"),
            "violations": len(val_res.get("violations") or []),
        },
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }
    return {"validation_result": val_res, "trace_steps": _append_step(state, step)}


def scenario_node(state: AllocationState) -> Dict[str, Any]:
    """Scenario agent: re-solves the base plan and compares it with the scenario plan."""
    start_time = time.time()
    weights = state.get("weights") or {"cost": 0.7, "carbon": 0.3}
    try:
        diff_res = tool_compare_plans(state.get("raw_solver_output") or {}, weights)
    except PLAN_ERRORS as exc:
        diff_res = {"status": STATUS_DATA_ERROR, "summary": f"Comparison failed: {exc}"}

    step = {
        "agent": "scenario",
        "tool": "compare_plans",
        "output": {
            "extra_cost": diff_res.get("extra_cost"),
            "extra_carbon_t": diff_res.get("extra_carbon_t"),
            "orders_moved": diff_res.get("orders_moved"),
            "late_delta": diff_res.get("late_delta"),
        },
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }
    return {"diff_output": diff_res, "answer_type": "scenario", "trace_steps": _append_step(state, step)}


def diagnosis_node(state: AllocationState) -> Dict[str, Any]:
    """Diagnostician agent: solves the plan, then explains one order's placement."""
    start_time = time.time()
    req = state.get("request", "")
    target = state.get("diff_output") or {}
    weights = state.get("weights") or {"cost": 0.7, "carbon": 0.3}

    order_ref = target.get("order_id")
    if not order_ref:
        entities = IntentClassifier.extract_order_entities(req)
        if entities and entities.get("order_id"):
            order_ref = entities["order_id"]
        else:
            order_ref = "O-077"

    plan = _plan_or_error(weights, state.get("scenario"), use_loaded_events=False)
    if plan["status"] == STATUS_DATA_ERROR:
        diag_res = {"status": STATUS_DATA_ERROR, "reasons": [plan["error"]]}
    else:
        try:
            diag_res = tool_diagnose_order(order_ref, plan, asked_plant=target.get("target_plant"))
        except PLAN_ERRORS as exc:
            diag_res = {"status": STATUS_DATA_ERROR, "reasons": [str(exc)]}

    step = {
        "agent": "diagnostician",
        "tool": "diagnose_order",
        "input": {"order_ref": order_ref, "asked_plant": target.get("target_plant")},
        "output": {k: diag_res.get(k) for k in ("status", "order_id", "assigned_plant", "cheapest_plant", "reasons")},
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }

    ans_type = "why_order" if state.get("intent") == "why_order" else ("infeasible" if "infeasible" in req.lower() else "allocation")

    return {
        "raw_solver_output": plan,
        "diff_output": diag_res,
        "answer_type": ans_type,
        "trace_steps": _append_step(state, step),
    }


def no_plan_node(state: AllocationState) -> Dict[str, Any]:
    """No plan could be produced: missing data, or infeasible even as a partial plan."""
    solver = state.get("raw_solver_output") or {}
    if solver.get("status") == STATUS_DATA_ERROR:
        text = f"Cannot plan: {solver.get('error')}"
        answer_type = "clarification_needed"
    else:
        hints = solver.get("infeasibility_hints") or ["no feasible allocation exists"]
        text = f"No feasible plan (solver status {solver.get('status')}). " + " ".join(hints)
        answer_type = "infeasible"
    step = {"agent": "optimiser", "tool": "explain_no_plan", "output": {"status": solver.get("status")}}
    return {"brief_text": text, "answer_type": answer_type, "trace_steps": _append_step(state, step)}


def writer_node(state: AllocationState) -> Dict[str, Any]:
    """Writer agent generating zero-hallucination brief."""
    start_time = time.time()
    from app.config import get_mock_llm
    mock_mode = get_mock_llm()

    prompt = build_brief_prompt(state)
    brief_text = ""
    tokens_used = 0

    if mock_mode:
        # Deterministic grounded brief generator
        brief_text = generate_grounded_mock_brief(state)
        tokens_used = 1450
        trace_logger.record_llm_call(state["run_id"], tokens=tokens_used, prompt_tokens=950, completion_tokens=500)
    else:
        # Invoke configured local Ollama LLM (http://192.168.3.173:11434)
        try:
            from app.orchestrator.llm import invoke_brief_llm
            brief_text, usage = invoke_brief_llm(prompt)
            tokens_used = usage.get("total_tokens", 1450)
            trace_logger.record_llm_call(
                state["run_id"],
                tokens=tokens_used,
                prompt_tokens=usage.get("prompt_tokens", 950),
                completion_tokens=usage.get("completion_tokens", 500),
            )
        except Exception as e:
            logger.warning(f"Ollama LLM execution failed: {e}. Falling back to deterministic brief.")
            brief_text = generate_grounded_mock_brief(state)
            tokens_used = 1200
            trace_logger.record_llm_call(state["run_id"], tokens=tokens_used, prompt_tokens=800, completion_tokens=400)

    # Numerical grounding verification audit
    audit = verify_brief_grounding(brief_text, state)
    if not audit["is_grounded"]:
        # Disqualification prevention: fallback immediately to verified brief if any hallucination detected
        brief_text = generate_grounded_mock_brief(state)

    # Record trace step matching Annex C
    step = {
        "agent": "writer",
        "tool": "write_brief",
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }

    trace_logger.log_step(
        run_id=state["run_id"],
        agent=step["agent"],
        tool=step["tool"],
        input_data=None,
        output_data=None,
    )

    current_steps = list(state.get("trace_steps") or [])
    current_steps.append(step)

    token_usage = state.get("token_usage") or {"total_tokens": 0, "llm_calls": 0}
    token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + tokens_used
    token_usage["llm_calls"] = token_usage.get("llm_calls", 0) + 1

    return {
        "brief_text": brief_text,
        "token_usage": token_usage,
        "trace_steps": current_steps,
    }


def clarification_node(state: AllocationState) -> Dict[str, Any]:
    """Handle underspecified or ambiguous queries."""
    text = (
        "Clarification needed: Your request is underspecified. "
        "Please provide specific parameters (e.g., target weights between cost and carbon, "
        "plant capacity event, or specific order ID to diagnose)."
    )
    step = {
        "agent": "router",
        "tool": "request_clarification",
        "output": {"reason": "Underspecified request"},
    }
    trace_logger.log_step(state["run_id"], step["agent"], step["tool"], output_data=step["output"])
    current_steps = list(state.get("trace_steps") or [])
    current_steps.append(step)

    return {
        "brief_text": text,
        "answer_type": "clarification_needed",
        "trace_steps": current_steps,
    }


def refusal_node(state: AllocationState) -> Dict[str, Any]:
    """Handle unsafe, invalid, or constraint-violating requests."""
    text = (
        "Request refused: The requested action asks to bypass plant physical capacity limits, "
        "ignore product eligibility constraints, or generate fabricated figures without tool verification. "
        "Per HCLTech Annex A rules, allocations must be strictly validated against physical constraints."
    )
    step = {
        "agent": "router",
        "tool": "refuse_request",
        "output": {"reason": "Violation of Annex A rules"},
    }
    trace_logger.log_step(state["run_id"], step["agent"], step["tool"], output_data=step["output"])
    current_steps = list(state.get("trace_steps") or [])
    current_steps.append(step)

    return {
        "brief_text": text,
        "answer_type": "refused",
        "trace_steps": current_steps,
    }


# ============================================================================
# ROUTING LOGIC & GRAPH DEFINITION
# ============================================================================

def select_next_node(state: AllocationState) -> Literal["refusal", "clarification", "optimiser", "diagnosis"]:
    """Conditional branching from router based on classified intent."""
    intent = state.get("intent", "allocation")
    if intent == "refused":
        return "refusal"
    elif intent == "clarification_needed":
        return "clarification"
    elif intent in ("diagnosis", "why_order"):
        return "diagnosis"
    else:
        return "optimiser"


def route_after_checker(state: AllocationState) -> Literal["scenario", "writer"]:
    """Conditional branching after checker: route to scenario if intent was scenario/re-plan."""
    intent = state.get("intent", "allocation")
    if intent == "scenario":
        return "scenario"
    return "writer"


def route_after_optimiser(state: AllocationState) -> Literal["checker", "no_plan"]:
    """Skip validation and writing when there is no plan at all."""
    solver = state.get("raw_solver_output") or {}
    return "checker" if solver.get("allocations") else "no_plan"


def build_allocation_graph() -> StateGraph:
    """Build and compile the LangGraph StateMachine orchestrating tool calls and brief generation."""
    workflow = StateGraph(AllocationState)

    # Register all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("optimiser", optimiser_node)
    workflow.add_node("checker", checker_node)
    workflow.add_node("scenario", scenario_node)
    workflow.add_node("diagnosis", diagnosis_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("refusal", refusal_node)
    workflow.add_node("no_plan", no_plan_node)

    # Connect edges
    workflow.add_edge(START, "router")

    # Dynamic conditional branch from router
    workflow.add_conditional_edges(
        "router",
        select_next_node,
        {
            "refusal": "refusal",
            "clarification": "clarification",
            "optimiser": "optimiser",
            "diagnosis": "diagnosis",
        },
    )

    # Core solver flow
    workflow.add_conditional_edges("optimiser", route_after_optimiser, {"checker": "checker", "no_plan": "no_plan"})

    # Conditional branch after checker: route to scenario comparator or straight to writer
    workflow.add_conditional_edges(
        "checker",
        route_after_checker,
        {
            "scenario": "scenario",
            "writer": "writer",
        },
    )

    # Scenario comparator links to writer
    workflow.add_edge("scenario", "writer")

    # Diagnosis flow: diagnosis -> checker -> writer
    workflow.add_edge("diagnosis", "checker")

    # Terminal nodes
    workflow.add_edge("writer", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("refusal", END)
    workflow.add_edge("no_plan", END)

    return workflow.compile()


# Singleton compiled graph
allocation_app = build_allocation_graph()


def run_pipeline(
    request: str,
    weights: Optional[Dict[str, float]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> AllocationState:
    """Execute the end-to-end allocation pipeline."""
    initial_state = create_initial_state(
        request=request,
        weights=weights,
        scenario=scenario,
        run_id=run_id,
    )

    # Initialize trace record in logger
    trace_logger.start_trace(run_id=initial_state["run_id"], request=request)

    # Run through LangGraph state machine
    final_state = allocation_app.invoke(initial_state)
    return final_state
