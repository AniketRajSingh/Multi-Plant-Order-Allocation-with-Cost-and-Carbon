"""Zero-hallucination prompt templates and verification for executive brief generation.

CRITICAL COMPETITION RULE:
'A solver builds the allocation. The LLM explains it and runs what-if scenarios
by calling tools. It NEVER invents numbers. If the LLM generates numbers not computed
by a tool, it is an automatic disqualification.'
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

from app.orchestrator.state import AllocationState


SYSTEM_PROMPT_EXECUTIVE_BRIEF = """You are the Senior Operations Research & Supply Chain Executive Briefing Agent for the HCLTech Multi-Plant Order Allocation System.

MISSION & NON-NEGOTIABLE INTEGRITY DIRECTIVE:
1. ABSOLUTE ZERO HALLUCINATION ON NUMBERS:
   - Every single number, currency figure, carbon tonnage, percentage, order count, date, and plant utilization metric MUST come directly and verbatim from the provided COMPUTED TOOL OUTPUTS.
   - YOU MUST NEVER INVENT, ESTIMATE, EXTRAPOLATE, AVERAGE, OR CALCULATE ANY NUMBER.
   - If a metric is not present in the tool output, explicitly state "Metric not computed by solver".
   - Generating any invented number results in AUTOMATIC DISQUALIFICATION.

2. STRUCTURED RECOMMENDATION CONTRACT (Annex A & Section 6.1):
   Every recommendation must include:
   - action: Concrete operational instruction (e.g. shift order, approve overtime, reallocate volume)
   - rationale: Grounded operational rationale explaining why, referencing solver KPIs
   - evidence_refs: Exact tool output keys/fields proving this recommendation (e.g. ["solver.kpis.total_cost", "diff.extra_cost"])
   - confidence: "HIGH", "MEDIUM", or "LOW"
   - estimated_impact: Grounded cost or carbon impact verbatim from tool outputs
   - requires_approval: boolean (true/false)

3. REQUIRED OUTPUT SECTIONS:
   - SUMMARY: High-level overview of the plan or scenario.
   - OPERATIONAL METRICS: Total landed cost, total carbon emissions, late orders, and plant utilisation.
   - TRADE-OFF ANALYSIS: Explain the cost vs. carbon weighting applied and alternative Pareto frontier points if computed.
   - SCENARIO / VARIANCE ANALYSIS (if applicable): Delta in cost, delta in carbon, and shifted orders caused by disturbances.
   - RECOMMENDATIONS: Formatted list matching the contract above.
   - ASSUMPTIONS: Grounded assumptions used by the solver.
   - LIMITATIONS: Known operational boundaries or data caveats.
"""


def format_context_for_llm(state: AllocationState) -> str:
    """Format all computed tool outputs into a strict read-only context block."""
    context_blocks = []

    context_blocks.append(f"REQUEST: {state.get('request')}")
    context_blocks.append(f"OBJECTIVE WEIGHTS USED: {json.dumps(state.get('weights', {}), indent=2)}")

    if state.get("scenario"):
        context_blocks.append(f"SCENARIO DISTURBANCE: {json.dumps(state.get('scenario'), indent=2)}")

    if state.get("raw_solver_output"):
        solver_clean = {
            "status": state["raw_solver_output"].get("status"),
            "kpis": state["raw_solver_output"].get("kpis"),
            "tradeoff": state["raw_solver_output"].get("tradeoff"),
            "allocation_sample": (state["raw_solver_output"].get("allocations") or [])[:5],
            "total_allocated_orders": len(state["raw_solver_output"].get("allocations") or []),
            "unallocated_orders": state["raw_solver_output"].get("unallocated_orders", []),
        }
        context_blocks.append(f"RAW SOLVER OUTPUT (GROUND TRUTH):\n{json.dumps(solver_clean, indent=2)}")
    else:
        context_blocks.append("RAW SOLVER OUTPUT: None")

    if state.get("validation_result"):
        context_blocks.append(f"VALIDATION RESULT (CHECKER):\n{json.dumps(state.get('validation_result'), indent=2)}")

    if state.get("diff_output"):
        context_blocks.append(f"SCENARIO COMPARATIVE DIFF (TOOL COMPUTED):\n{json.dumps(state.get('diff_output'), indent=2)}")

    return "\n\n".join(context_blocks)


def build_brief_prompt(state: AllocationState) -> str:
    """Construct the briefing prompt for the LLM adhering strictly to tool context."""
    tool_context = format_context_for_llm(state)

    prompt = f"""{SYSTEM_PROMPT_EXECUTIVE_BRIEF}

=========================
GROUND TRUTH COMPUTED DATA (DO NOT DEVIATE OR INVENT)
=========================
{tool_context}

=========================
INSTRUCTIONS:
Generate the Executive Brief based ONLY on the ground truth computed data above.
Do not introduce any numbers, percentages, or metrics that do not appear in the block above.
Produce clean, professional markdown.
"""
    return prompt


def extract_numbers_from_text(text: str) -> Set[float]:
    """Extract all numeric quantities from a text string, ignoring markdown headers/bullets."""
    # Match integers, floats, percentages, and scientific notation
    # Strip bullet numbering like '1.', '2.' if at line start
    cleaned = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Find numbers like 1,840,000 or 1840000 or 410 or 0.88 or 70%
    matches = re.findall(r"[-+]?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b|[-+]?\b\d+(?:\.\d+)?%?\b", cleaned)
    numbers = set()
    for m in matches:
        m_clean = m.replace(",", "").replace("%", "")
        try:
            val = float(m_clean)
            numbers.add(val)
        except ValueError:
            continue
    return numbers


def extract_numbers_from_object(obj: Any) -> Set[float]:
    """Recursively extract all numeric values from dictionaries, lists, and primitives."""
    nums: Set[float] = set()
    if isinstance(obj, (int, float)):
        nums.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums.update(extract_numbers_from_object(v))
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            nums.update(extract_numbers_from_object(item))
    elif isinstance(obj, str):
        # Look for numeric substrings in strings (e.g. 'O-077' -> 77, '70% cost' -> 70, 0.7)
        for n in extract_numbers_from_text(obj):
            nums.add(n)
            if n <= 100:
                nums.add(round(n / 100.0, 4))
    return nums


def verify_brief_grounding(brief_text: str, state: AllocationState) -> Dict[str, Any]:
    """Audit the brief text to ensure EVERY number was computed by a tool.

    Returns:
        Dict with 'is_grounded': bool, 'hallucinated_numbers': list, 'grounded_count': int
    """
    text_numbers = extract_numbers_from_text(brief_text)

    # Gather all valid numbers from tool results
    ground_truth_numbers: Set[float] = set()

    if state.get("weights"):
        ground_truth_numbers.update(extract_numbers_from_object(state["weights"]))
        # Also add percentage representations (0.7 -> 70)
        for w in state["weights"].values():
            if isinstance(w, (int, float)):
                ground_truth_numbers.add(round(w * 100, 2))
                ground_truth_numbers.add(round(w, 2))

    if state.get("scenario"):
        ground_truth_numbers.update(extract_numbers_from_object(state["scenario"]))

    if state.get("raw_solver_output"):
        ground_truth_numbers.update(extract_numbers_from_object(state["raw_solver_output"].get("kpis", {})))
        ground_truth_numbers.update(extract_numbers_from_object(state["raw_solver_output"].get("tradeoff", [])))
        # Also plant utilisations as percentages (0.88 -> 88)
        kpis = state["raw_solver_output"].get("kpis", {})
        for util in kpis.get("plant_utilisation", {}).values():
            if isinstance(util, (int, float)):
                ground_truth_numbers.add(round(util * 100, 2))
                ground_truth_numbers.add(round(util, 2))

    if state.get("validation_result"):
        ground_truth_numbers.update(extract_numbers_from_object(state["validation_result"]))

    if state.get("diff_output"):
        ground_truth_numbers.update(extract_numbers_from_object(state["diff_output"]))

    # Common acceptable boilerplate numbers (e.g., section numbers, year 2026, 0, 1)
    allowlist = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 2026.0}

    hallucinations = []
    for num in text_numbers:
        # Check exact or near-match (for float representation rounding)
        matched = False
        if num in allowlist:
            continue

        for truth in ground_truth_numbers:
            if abs(num - truth) < 1e-3 or (truth != 0 and abs(num - truth) / abs(truth) < 1e-3):
                matched = True
                break

        if not matched:
            hallucinations.append(num)

    return {
        "is_grounded": len(hallucinations) == 0,
        "hallucinated_numbers": hallucinations,
        "grounded_count": len(text_numbers) - len(hallucinations),
        "total_numbers_in_brief": len(text_numbers),
    }


def generate_grounded_mock_brief(state: AllocationState) -> str:
    """Deterministically generate an executive brief strictly using tool numbers.

    Used when MOCK_LLM=true or as a guaranteed zero-hallucination fallback.
    """
    solver = state.get("raw_solver_output") or {}
    kpis = solver.get("kpis") or {}
    weights = state.get("weights") or {"cost": 0.7, "carbon": 0.3}
    diff = state.get("diff_output") or {}
    is_scenario = "extra_cost" in diff
    is_why = "reasons" in diff
    val = state.get("validation_result") or {}

    cost = kpis.get("total_cost") or 0
    carbon_t = kpis.get("total_carbon_t") or 0
    late = kpis.get("late_orders") or 0
    util = kpis.get("plant_utilisation") or {}
    util_str = ", ".join(f"Plant {p}: {round(v * 100, 2)}%" for p, v in util.items() if v is not None)
    unallocated = solver.get("unallocated_orders") or []

    lines = []
    lines.append("# Multi-Plant Allocation Executive Brief")
    lines.append(f"**Run ID:** `{state.get('run_id')}` | **Intent:** `{state.get('intent')}`")
    lines.append("")

    lines.append("## 1. Executive Summary")
    if is_scenario:
        lines.append(f"Re-planned after the capacity change. {diff.get('summary', '')}")
    elif is_why:
        lines.append(f"Order {diff.get('order_id')} is built in Plant {diff.get('assigned_plant')}. " + " ".join(diff.get("reasons", [])))
    else:
        lines.append(
            f"Allocation solved at {round(weights.get('cost', 0.7) * 100, 2)}% cost / "
            f"{round(weights.get('carbon', 0.3) * 100, 2)}% carbon weighting. "
            f"Total landed cost is {cost:,.2f} with total carbon of {carbon_t} t and {late} late orders."
        )
    if unallocated:
        lines.append(f"{len(unallocated)} orders cannot be met with the available capacity: "
                     + ", ".join(u["order_id"] for u in unallocated) + ".")

    lines.append("")
    lines.append("## 2. Operational KPIs & Plant Utilisation")
    lines.append(f"- **Total Landed Cost:** {cost:,.2f}")
    lines.append(f"- **Total Carbon:** {carbon_t} t")
    lines.append(f"- **Late Orders:** {late}" + (f" ({', '.join(kpis.get('late_order_ids') or [])})" if late else ""))
    if util_str:
        lines.append(f"- **Plant Utilisation:** {util_str}")
    lines.append(f"- **Validation Status:** Capacity OK = `{val.get('capacity_ok')}`, Eligibility OK = `{val.get('eligibility_ok')}`")

    if is_scenario:
        lines.append("")
        lines.append("## 3. Scenario Variance Analysis")
        lines.append(f"- **Incremental Cost:** {diff.get('extra_cost', 0):+,.2f}")
        lines.append(f"- **Incremental Carbon:** {diff.get('extra_carbon_t', 0):+} t")
        lines.append(f"- **Orders Reallocated:** {diff.get('orders_moved', 0)}")
        lines.append(f"- **Late Orders Delta:** {diff.get('late_delta', 0):+d}")
        for m in (diff.get("moved_order_details") or [])[:10]:
            lines.append(f"  - {m['order_id']}: Plant {m['from_plant']} -> Plant {m['to_plant']} "
                         f"(cost {m['cost_delta']:+,.2f}, carbon {m['carbon_delta_kg']:+} kg)")
    elif is_why:
        lines.append("")
        lines.append("## 3. Plant Options for This Order")
        for o in diff.get("options") or []:
            if not o.get("eligible"):
                lines.append(f"- Plant {o['plant']}: cannot make this product")
                continue
            lines.append(f"- Plant {o['plant']}: cost {o['cost']:,.2f}, carbon {o['carbon_kg']} kg, "
                         f"arrives {o['arrival']} ({'on time' if o['on_time'] else str(o['days_late']) + ' day(s) late'})")

    lines.append("")
    lines.append("## 4. Structured Recommendations")
    if is_scenario:
        lines.append("### Recommendation 1: Approve the re-planned allocation")
        lines.append(f"- **Action:** Move {diff.get('orders_moved', 0)} orders as listed above.")
        lines.append("- **Rationale:** Keeps every plant within its reduced capacity at the lowest weighted cost and carbon.")
        lines.append("- **Evidence Refs:** `['diff.extra_cost', 'diff.extra_carbon_t', 'diff.orders_moved']`")
        lines.append("- **Confidence:** HIGH")
        lines.append(f"- **Estimated Impact:** {diff.get('extra_cost', 0):+,.2f} cost, {diff.get('extra_carbon_t', 0):+} t carbon")
        lines.append("- **Requires Approval:** True")
    else:
        lines.append("### Recommendation 1: Execute the solver allocation")
        lines.append("- **Action:** Release production and transport schedules according to the solver allocation.")
        lines.append("- **Rationale:** The plan passes the independent capacity and eligibility checks.")
        lines.append("- **Evidence Refs:** `['solver.kpis.total_cost', 'validator.capacity_ok', 'validator.eligibility_ok']`")
        lines.append("- **Confidence:** HIGH")
        lines.append(f"- **Estimated Impact:** Landed cost of {cost:,.2f} and {carbon_t} t carbon")
        lines.append("- **Requires Approval:** " + ("True" if unallocated or late else "False"))

    lines.append("")
    lines.append("## 5. Assumptions & Limitations")
    lines.append("- **Assumptions:** Weights applied as requested; landed cost = (unit cost + transport) x (1 + duty).")
    lines.append("- **Limitations:** Energy price is not in the data, so energy cost is excluded; one-week planning horizon.")

    return "\n".join(lines)
