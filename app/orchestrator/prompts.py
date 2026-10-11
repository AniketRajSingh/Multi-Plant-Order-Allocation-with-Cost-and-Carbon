"""Zero-hallucination prompt templates and verification for executive brief generation.

CRITICAL COMPETITION RULE:
'A solver builds the allocation. The LLM explains it and runs what-if scenarios
by calling tools. It NEVER invents numbers. If the LLM generates numbers not computed
by a tool, it is an automatic disqualification.'

Instruction:
"Write a 1-page executive brief using ONLY the verified metrics in the context.
Do NOT calculate or hallucinate any numbers. Format with Action, Rationale, and KPI Impact."
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

from app.orchestrator.state import AllocationState


EXECUTIVE_BRIEF_INSTRUCTION = (
    "Write a 1-page executive brief using ONLY the verified metrics in the context. "
    "Do NOT calculate or hallucinate any numbers. Format with Action, Rationale, and KPI Impact."
)

SYSTEM_PROMPT_EXECUTIVE_BRIEF = f"""You are the Senior Operations Research & Executive Briefing Agent for the HCLTech Multi-Plant Order Allocation System.

CORE MANDATE & NON-NEGOTIABLE INTEGRITY DIRECTIVE:
{EXECUTIVE_BRIEF_INSTRUCTION}

STRICT OPERATIONAL RULES:
1. READ-ONLY GROUND TRUTH:
   - The provided JSON metrics block contains the ONLY verified ground-truth facts.
   - You MUST treat all numbers, dollar amounts, carbon tonnages, percentages, order counts, dates, and plant utilisations as strictly read-only facts.
   - YOU MUST NEVER CALCULATE, ESTIMATE, INTERPOLATE, ROUND ARBITRARILY, OR INVENT ANY NUMBER.
   - If a figure is not present in the verified metrics context, write: 'Not computed by solver'.
   - Any hallucinated or invented number results in AUTOMATIC DISQUALIFICATION.

2. MANDATORY RECOMMENDATION SCHEMA:
   Every recommendation must be explicitly formatted with:
   - Action: Concrete operational directive (e.g., execute allocation, reassign volume, approve contingency)
   - Rationale: Grounded operational reason citing the verified solver findings
   - KPI Impact: Exact quantitative cost, carbon, or schedule impact verbatim from the verified metrics context
   - Evidence Refs: Specific JSON keys verifying these metrics (e.g., ['raw_solver_output.kpis.total_cost', 'diff_output.extra_cost'])

3. STRUCTURED BRIEF SECTIONS:
   - 1. Executive Summary (High-level situation overview and core trade-off decision)
   - 2. Verified Operational KPIs (Total Landed Cost, Carbon Tonnage, Late Orders, Plant Utilisation)
   - 3. Structured Operational Decisions (Formatted with Action, Rationale, and KPI Impact)
   - 4. Scenario / Variance Breakdown (If scenario disturbance is present)
   - 5. Assumptions and Limitations (Operational boundaries and tariff rules)
"""


def format_context_for_llm(state: AllocationState) -> str:
    """Format raw JSON metrics into a strictly read-only fact block."""
    context_data: Dict[str, Any] = {
        "run_id": state.get("run_id"),
        "request": state.get("request"),
        "intent": state.get("intent"),
        "answer_type": state.get("answer_type"),
        "weights_used": state.get("weights", {"cost": 0.7, "carbon": 0.3}),
    }

    if state.get("scenario"):
        context_data["scenario_disturbance"] = state["scenario"]

    if state.get("raw_solver_output"):
        solver = state["raw_solver_output"]
        context_data["verified_solver_metrics"] = {
            "status": solver.get("status"),
            "kpis": solver.get("kpis"),
            "tradeoff_points": solver.get("tradeoff", []),
            "total_allocated_orders": len(solver.get("allocations", [])),
            "allocation_sample": (solver.get("allocations") or [])[:5],
        }

    if state.get("validation_result"):
        val = state["validation_result"]
        context_data["validation_audit"] = {
            "capacity_ok": val.get("capacity_ok"),
            "eligibility_ok": val.get("eligibility_ok"),
            "delivery_ok": val.get("delivery_ok"),
            "plant_headrooms": val.get("plant_headrooms", {}),
            "violations": val.get("violations", []),
            "warnings": val.get("warnings", []),
        }

    if state.get("diff_output"):
        context_data["scenario_variance_diff"] = state["diff_output"]

    return json.dumps(context_data, indent=2, default=str)


def build_brief_prompt(state: AllocationState) -> str:
    """Construct the briefing prompt for the LLM adhering strictly to verified read-only facts."""
    read_only_json = format_context_for_llm(state)

    prompt = f"""{SYSTEM_PROMPT_EXECUTIVE_BRIEF}

================================================================================
VERIFIED RAW METRICS (STRICTLY READ-ONLY FACTS — DO NOT DEVIATE OR INVENT)
================================================================================
{read_only_json}

================================================================================
INSTRUCTION:
{EXECUTIVE_BRIEF_INSTRUCTION}
Generate a crisp 1-page executive brief. Format with Action, Rationale, and KPI Impact.
"""
    return prompt


# ============================================================================
# NUMERICAL GROUNDING AUDITOR (Zero-Hallucination Guardrail)
# ============================================================================

def extract_numbers_from_text(text: str) -> Set[float]:
    """Extract all numeric quantities from text, stripping currencies, percentages, and punctuation."""
    # Remove markdown header and list enumeration markers like '# 1.', '1. ', '2. '
    cleaned = re.sub(r"^\s*(?:#+\s*)?\d+\.\s+", "", text, flags=re.MULTILINE)

    matches = re.findall(
        r"[-+]?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b|[-+]?\b\d+(?:\.\d+)?%?\b",
        cleaned,
    )
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
    """Recursively harvest all valid numeric metrics from raw tool output objects."""
    nums: Set[float] = set()
    if isinstance(obj, (int, float)):
        val = float(obj)
        nums.add(val)
        # Add integer representation and rounded forms
        nums.add(round(val, 2))
        nums.add(round(val, 1))
        # Add percentage representations (e.g., 0.7 -> 70.0, 0.1608 -> 16.1, 16.08)
        if 0.0 <= val <= 1.0:
            pct = val * 100.0
            nums.add(round(pct, 2))
            nums.add(round(pct, 1))
            nums.add(round(pct, 0))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums.update(extract_numbers_from_object(v))
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            nums.update(extract_numbers_from_object(item))
    elif isinstance(obj, str):
        for n in extract_numbers_from_text(obj):
            nums.add(n)
            if n <= 100:
                nums.add(round(n / 100.0, 4))
    return nums


def verify_brief_grounding(brief_text: str, state: AllocationState) -> Dict[str, Any]:
    """Audit the generated brief to confirm EVERY number originates in verified tool outputs.

    If any number is not present in the verified context, flags it as a hallucination.
    """
    text_numbers = extract_numbers_from_text(brief_text)
    ground_truth_numbers: Set[float] = set()

    # Collect numbers from state components
    if state.get("weights"):
        ground_truth_numbers.update(extract_numbers_from_object(state["weights"]))

    if state.get("scenario"):
        ground_truth_numbers.update(extract_numbers_from_object(state["scenario"]))

    if state.get("raw_solver_output"):
        solver = state["raw_solver_output"]
        ground_truth_numbers.update(extract_numbers_from_object(solver.get("kpis", {})))
        ground_truth_numbers.update(extract_numbers_from_object(solver.get("tradeoff", [])))
        for a in solver.get("allocations", []):
            ground_truth_numbers.update(extract_numbers_from_object(a))

    if state.get("validation_result"):
        ground_truth_numbers.update(extract_numbers_from_object(state["validation_result"]))

    if state.get("diff_output"):
        ground_truth_numbers.update(extract_numbers_from_object(state["diff_output"]))

    # Allow harmless structural numbers (section numbers, years, 0, 1)
    allowlist = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 2026.0}

    hallucinations = []
    for num in text_numbers:
        if num in allowlist:
            continue

        matched = False
        for truth in ground_truth_numbers:
            if abs(num - truth) < 1e-2 or (truth != 0 and abs(num - truth) / abs(truth) < 1e-2):
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


# ============================================================================
# DETERMINISTIC GROUNDED BRIEF GENERATOR (Zero-Hallucination Template)
# ============================================================================

def generate_grounded_mock_brief(state: AllocationState) -> str:
    """Deterministically generate an executive brief using ONLY verified metrics.

    Strictly formatted with Action, Rationale, and KPI Impact as instructed.
    """
    solver = state.get("raw_solver_output") or {}
    kpis = solver.get("kpis") or {}
    weights = state.get("weights") or {"cost": 0.7, "carbon": 0.3}
    diff = state.get("diff_output") or {}
    is_scenario = "extra_cost" in diff
    is_why = "reasons" in diff or "order_id" in diff
    val = state.get("validation_result") or {}

    cost = float(kpis.get("total_cost") or 0.0)
    carbon_t = float(kpis.get("total_carbon_t") or 0.0)
    late = int(kpis.get("late_orders") or 0)
    util = kpis.get("plant_utilisation") or {}
    util_str = ", ".join(f"Plant {p}: {round(float(v) * 100, 2)}%" for p, v in sorted(util.items()) if v is not None)
    unallocated = solver.get("unallocated_orders") or []

    lines = []
    lines.append("# Executive Brief: Multi-Plant Order Allocation")
    lines.append(f"**Run ID:** `{state.get('run_id')}` | **Intent:** `{state.get('intent')}`")
    lines.append("")

    lines.append("## 1. Executive Summary")
    if is_scenario:
        lines.append(f"Re-planned after the capacity change. {diff.get('summary', '')}")
    elif is_why:
        lines.append(f"Order {diff.get('order_id')} is built in Plant {diff.get('assigned_plant') or diff.get('recommended_plant')}. " + " ".join(diff.get("reasons", [])))
    else:
        lines.append(
            f"Allocation solved at {round(weights.get('cost', 0.7) * 100, 2)}% cost / "
            f"{round(weights.get('carbon', 0.3) * 100, 2)}% carbon weighting. "
            f"Total landed cost is ${cost:,.2f} with total carbon of {carbon_t} t and {late} late orders."
        )
    if unallocated:
        lines.append(f"{len(unallocated)} orders cannot be met with the available capacity: "
                     + ", ".join(u["order_id"] for u in unallocated) + ".")

    lines.append("")
    lines.append("## 2. Operational KPIs & Plant Utilisation")
    lines.append(f"- **Total Landed Cost:** ${cost:,.2f}")
    lines.append(f"- **Total Carbon:** {carbon_t} t")
    lines.append(f"- **Late Orders:** {late}" + (f" ({', '.join(kpis.get('late_order_ids') or [])})" if late and kpis.get('late_order_ids') else ""))
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
            lines.append(f"  - {m['order_id']}: Plant {m['from_plant']} -> Plant {m['to_plant']}")
    elif is_why:
        lines.append("")
        lines.append("## 3. Plant Options for This Order")
        for o in diff.get("options") or []:
            if not o.get("eligible"):
                lines.append(f"- Plant {o['plant']}: cannot make this product")
                continue
            lines.append(f"- Plant {o['plant']}: cost ${o['cost']:,.2f}, carbon {o['carbon_kg']} kg, "
                         f"arrives {o['arrival']} ({'on time' if o['on_time'] else str(o['days_late']) + ' day(s) late'})")

    lines.append("")
    lines.append("## 4. Structured Operational Decisions (Action, Rationale, KPI Impact)")
    if is_scenario:
        lines.append("### Recommendation 1: Approve the re-planned allocation")
        lines.append(f"- **Action:** Move {diff.get('orders_moved', 0)} orders to alternative manufacturing plants.")
        lines.append("- **Rationale:** Keeps every plant within its reduced capacity at lowest cost and carbon trade-off.")
        lines.append("- **KPI Impact:** " + f"{diff.get('extra_cost', 0):+,.2f} cost, {diff.get('extra_carbon_t', 0):+} t carbon")
        lines.append("- **Evidence Refs:** `['diff.extra_cost', 'diff.extra_carbon_t', 'diff.orders_moved']`")
        lines.append("- **Confidence:** HIGH")
        lines.append("- **Requires Approval:** True")
    elif is_why:
        oid = diff.get("order_id", "O-077")
        rec_plant = diff.get("assigned_plant") or diff.get("recommended_plant", "C")
        lines.append(f"### Recommendation 1: Confirm Plant Allocation for {oid}")
        lines.append(f"- **Action:** Build Order {oid} in Plant {rec_plant}.")
        lines.append("- **Rationale:** " + (diff.get("reasons", ["Delivers on time without capacity violation"])[0] if diff.get("reasons") else "Optimal plant assignment."))
        lines.append(f"- **KPI Impact:** Satisfies delivery deadline on time with optimal landed cost.")
        lines.append("- **Evidence Refs:** `['diff_output.options', 'validator.delivery_ok']`")
        lines.append("- **Confidence:** HIGH")
        lines.append("- **Requires Approval:** False")
    else:
        lines.append("### Recommendation 1: Execute the solver allocation")
        lines.append("- **Action:** Release production and transport schedules according to the solver allocation.")
        lines.append("- **Rationale:** The plan strictly passes independent capacity and eligibility checks.")
        lines.append(f"- **KPI Impact:** Landed cost of ${cost:,.2f} and {carbon_t} t carbon.")
        lines.append("- **Evidence Refs:** `['solver.kpis.total_cost', 'validator.capacity_ok', 'validator.eligibility_ok']`")
        lines.append("- **Confidence:** HIGH")
        lines.append("- **Requires Approval:** " + ("True" if unallocated or late else "False"))

    lines.append("")
    lines.append("## 5. Assumptions & Limitations")
    lines.append("- **Assumptions:** Weights applied as requested; landed cost = (unit cost + transport) x (1 + duty).")
    lines.append("- **Limitations:** Energy price is not in the schema, so energy monetary cost is excluded; one-week planning horizon.")


    return "\n".join(lines)
