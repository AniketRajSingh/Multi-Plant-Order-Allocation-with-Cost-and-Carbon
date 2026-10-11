"""Intent and Answer Type Classifier for Multi-Plant Order Allocation Orchestrator.

Classifies incoming requests directly into the system's operational answer_types:
- "allocation": Standard optimization request (e.g. "Allocate this week's orders").
- "scenario": What-if request (e.g. "Plant B loses 30% capacity. Re-plan.").
- "why_order": Diagnosis query (e.g. "Why was order O-077 built in Plant C?").
- "brief": Leadership summary request (e.g. "Generate management brief for leadership").
- "infeasible" / "clarification_needed" / "refused": Edge cases.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional, Tuple

from app.orchestrator.state import AllocationState, AnswerType, IntentType


class IntentClassifier:
    """Classifies user queries into standard answer_types with entity extraction."""

    # 1. Edge Case: Refusal patterns (attempting to bypass physics, constraints, or fabricate data)
    REFUSAL_PATTERNS = [
        r"(?i)\b(?:ignore|bypass|override|disable)\b.*\b(?:capacity|constraint|eligibility|rules)\b",
        r"(?i)\b(?:force|push)\b.*\b(?:above|over|exceed)\b.*\b(?:100%|capacity|maximum)\b",
        r"(?i)\b(?:fake|invent|fudge|make up)\b.*\b(?:numbers?|costs?|carbon|results?)\b",
        r"(?i)\b(?:hack|jailbreak|disregard previous)\b",
    ]

    # 2. Edge Case: Clarification needed (too brief, ambiguous, or no actionable directive)
    CLARIFICATION_EXACT = {"help", "what", "do it", "run", "why", "plan", "?", "orders", "allocate"}

    # 3. Edge Case: Infeasible requests (orders that cannot be met, zero valid plants, impossible due dates)
    INFEASIBLE_PATTERNS = [
        r"(?i)\b(?:infeasib(?:le|ility)|impossible\s+to\s+deliver)\b",
        r"(?i)\bno\s+plant\s+(?:can|is\s+eligible\s+to)\s+build\b",
        r"(?i)\bunfulfillable\s+orders?\b",
        r"(?i)\bcannot\s+(?:meet|fulfill|deliver)\s+(?:any|all)\s+orders?\b",
    ]

    # 4. Diagnosis: why_order queries
    # e.g. "Why was order O-077 built in Plant C?", "Why is order O-077 late?", "Diagnose order O-077"
    WHY_ORDER_PATTERNS = [
        r"(?i)\bwhy\s+(?:was|is)\s+order\s+([A-Z0-9_-]+)\s+(?:built|made|produced|allocated|assigned)\b",
        r"(?i)\bwhy\s+(?:was|is)\s+order\s+([A-Z0-9_-]+)\s+(?:late|delayed|unallocated)\b",
        r"(?i)\bwhy\s+did\s+order\s+([A-Z0-9_-]+)\s+go\s+to\b",
        r"(?i)\bwhy\s+order\s+([A-Z0-9_-]+)\b",
        r"(?i)\bdiagnos(?:e|is)\s+(?:order\s+)?([A-Z0-9_-]+)?\b",
        r"(?i)\broot[\s-]cause\s+(?:for\s+order\s+)?([A-Z0-9_-]+)?\b",
        r"(?i)\bexplain\s+(?:order\s+)?([A-Z0-9_-]+)\b",
        r"(?i)\bcheck\s+order\s+([A-Z0-9_-]+)\b",
    ]

    # 5. Scenario & Re-plan: What-if requests
    # e.g. "Plant B loses 30% capacity. Re-plan.", "What if carbon tax doubles?"
    SCENARIO_PATTERNS = [
        r"(?i)\b(?:re-?plan|replan)\b",
        r"(?i)\bwhat[\s-]if\b",
        r"(?i)\bloses?\s+\d+%\s+capacity\b",
        r"(?i)\bcapacity\s+(?:loss|cut|drop|reduction|outage|event)\b",
        r"(?i)\b(?:shutdown|downtime|breakdown|offline|maintenance)\b",
        r"(?i)\bscenario\b",
        r"(?i)\bcarbon\s+tax\s+(?:increase|doubles?)\b",
        r"(?i)\bdisturbance\b",
    ]

    # 6. Management & Leadership Brief requests
    # e.g. "Leadership summary request", "Generate management brief for leadership"
    BRIEF_PATTERNS = [
        r"(?i)\b(?:management|executive|leadership|board)\s+(?:brief|summary)\b",
        r"(?i)\bleadership\s+summary\s+request\b",
        r"(?i)\bwrite\s+(?:a\s+)?(?:executive\s+)?brief\b",
        r"(?i)\bsummariz(?:e|ation)\s+for\s+(?:executives?|leadership|management|board)\b",
        r"(?i)\bgenerate\s+(?:a\s+)?brief\b",
        r"(?i)\bexecutive\s+briefing\b",
    ]

    # 7. Standard Allocation: Standard optimization request
    # e.g. "Allocate this week's orders", "Standard optimization request", "Run optimal allocation"
    ALLOCATION_PATTERNS = [
        r"(?i)\ballocat(?:e|ion)\b",
        r"(?i)\bstandard\s+optimization\s+request\b",
        r"(?i)\boptimi(?:ze|zation|se|sation)\b",
        r"(?i)\bsolve\s+(?:allocation|orders?)?\b",
        r"(?i)\bassign\s+(?:this\s+week'?s\s+)?orders\b",
        r"(?i)\boptimal\s+plan\b",
        r"(?i)\brun\s+(?:planner|allocation|optimizer)\b",
        r"(?i)\bgenerate\s+(?:allocation\s+)?plan\b",
    ]

    @classmethod
    def classify(cls, request: str, scenario_data: Optional[Dict[str, Any]] = None) -> Tuple[AnswerType, float, str]:
        """Classify request string into standard answer_types with confidence and reasoning.

        Returns:
            Tuple of (answer_type, confidence, reasoning)
        """
        text = (request or "").strip()

        # 1. Guardrail / Refusal check
        for pat in cls.REFUSAL_PATTERNS:
            if re.search(pat, text):
                return (
                    "refused",
                    0.99,
                    "Request asks to violate physical capacity, eligibility constraints, or invent data.",
                )

        # 2. Clarification needed check (too short, ambiguous, or no actionable directive)
        if len(text) < 4 or text.lower() in cls.CLARIFICATION_EXACT:
            return (
                "clarification_needed",
                0.95,
                "Request is too brief or ambiguous to determine required operational action.",
            )

        # If a scenario payload was explicitly provided in API request
        if scenario_data and (
            scenario_data.get("plant") or scenario_data.get("capacity_loss_pct") or scenario_data.get("event_id")
        ):
            return "scenario", 0.98, "Explicit scenario / capacity event disturbance payload supplied."

        # 3. why_order: Diagnosis query (e.g. "Why was order O-077 built in Plant C?")
        for pat in cls.WHY_ORDER_PATTERNS:
            if re.search(pat, text):
                return "why_order", 0.96, f"Matched why_order / diagnosis pattern: {pat}"

        # 4. Infeasible query
        for pat in cls.INFEASIBLE_PATTERNS:
            if re.search(pat, text):
                return "infeasible", 0.95, f"Matched infeasible request pattern: {pat}"

        # 5. Scenario & Re-plan: What-if request
        for pat in cls.SCENARIO_PATTERNS:
            if re.search(pat, text):
                return "scenario", 0.95, f"Matched scenario / what-if pattern: {pat}"

        # 6. Leadership summary / Executive brief
        for pat in cls.BRIEF_PATTERNS:
            if re.search(pat, text):
                return "brief", 0.93, f"Matched leadership brief pattern: {pat}"

        # 7. Standard allocation
        for pat in cls.ALLOCATION_PATTERNS:
            if re.search(pat, text):
                return "allocation", 0.92, f"Matched standard allocation pattern: {pat}"

        # Fallback keyword match
        if re.search(r"(?i)\b(?:order|plant|cost|carbon|tradeoff|weight)\b", text):
            return "allocation", 0.75, "Defaulted to allocation based on operational domain keywords."

        return (
            "clarification_needed",
            0.60,
            "Could not determine intended action. Request should specify allocation, scenario, why_order, or brief.",
        )

    @classmethod
    def extract_order_entities(cls, text: str) -> Dict[str, Optional[str]]:
        """Extract order_id (e.g. 'O-077') and plant (e.g. 'Plant C') from why_order queries."""
        # Match order identifiers like O-077, ORD-123, O12
        order_match = re.search(r"\b(O(?:RD)?-[0-9]+|[0-9]{3,}|order\s+([A-Z0-9_-]+))\b", text, re.IGNORECASE)
        order_id = None
        if order_match:
            order_id = order_match.group(1).upper()
            if order_id.lower().startswith("order "):
                order_id = order_id[6:].strip()

        # Match plant identifier like Plant C, Plant A
        plant_match = re.search(r"\bPlant\s+([A-Z0-9]+)\b", text, re.IGNORECASE)
        plant = plant_match.group(1).upper() if plant_match else None

        return {
            "order_id": order_id or "O-077",
            "plant": plant,
        }

    @classmethod
    def extract_scenario_entities(cls, text: str) -> Optional[Dict[str, Any]]:
        """Extract plant and capacity loss percentage from natural language what-if text."""
        plant_match = re.search(r"(?i)\bPlant\s+([A-Z0-9]+)\b", text)
        loss_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:capacity)?", text)

        if plant_match and loss_match:
            plant = plant_match.group(1).upper()
            pct = float(loss_match.group(1))
            return {
                "event_id": f"event-{plant.lower()}-{int(pct)}pct",
                "plant": plant,
                "capacity_loss_pct": pct,
                "description": f"Extracted from request: Plant {plant} loses {pct}% capacity",
            }
        return None

    @classmethod
    def extract_weights(cls, text: str) -> Optional[Dict[str, float]]:
        """Extract cost/carbon weights from text if specified (e.g. '70% cost, 30% carbon')."""
        cost_match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:cost)", text, re.IGNORECASE)
        carbon_match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:carbon)", text, re.IGNORECASE)

        if cost_match and carbon_match:
            c = float(cost_match.group(1))
            k = float(carbon_match.group(1))
            if c > 1.0 or k > 1.0:
                total = c + k
                return {"cost": round(c / total, 2), "carbon": round(k / total, 2)}
            return {"cost": c, "carbon": k}
        return None

    @classmethod
    def classify_request(
        cls, request: str, scenario_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method returning a full classification dictionary with extracted entities."""
        ans_type, conf, reason = cls.classify(request, scenario_data=scenario_data)

        extracted: Dict[str, Any] = {}
        if ans_type == "why_order":
            extracted.update(cls.extract_order_entities(request))
        elif ans_type == "scenario":
            scen = cls.extract_scenario_entities(request)
            if scen:
                extracted["scenario"] = scen

        w = cls.extract_weights(request)
        if w:
            extracted["weights"] = w

        return {
            "answer_type": ans_type,
            "intent": ans_type,
            "confidence": conf,
            "reasoning": reason,
            "extracted_entities": extracted,
        }


def router_node(state: AllocationState) -> Dict[str, Any]:
    """LangGraph node that classifies intent into answer_type and updates state."""
    start_time = time.time()
    req = state.get("request", "")
    existing_scenario = state.get("scenario")

    classification = IntentClassifier.classify_request(req, scenario_data=existing_scenario)
    ans_type = classification["answer_type"]
    conf = classification["confidence"]
    reason = classification["reasoning"]
    extracted = classification["extracted_entities"]

    updates: Dict[str, Any] = {
        "intent": ans_type,
        "answer_type": ans_type,
    }

    # Populate scenario if extracted
    if ans_type == "scenario" and not existing_scenario and "scenario" in extracted:
        updates["scenario"] = extracted["scenario"]

    # Populate weights if extracted
    if "weights" in extracted and not state.get("weights"):
        updates["weights"] = extracted["weights"]

    # If why_order, store target order in diff_output for downstream diagnosis
    if ans_type == "why_order":
        updates["diff_output"] = {
            "order_id": extracted.get("order_id", "O-077"),
            "target_plant": extracted.get("plant"),
        }

    # Add trace step matching Annex C format
    step = {
        "agent": "router",
        "tool": "classify_intent",
        "input": {"request": req},
        "output": {
            "answer_type": ans_type,
            "intent": ans_type,
            "confidence": conf,
            "reasoning": reason,
            "extracted": extracted,
        },
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }

    current_steps = list(state.get("trace_steps") or [])
    current_steps.append(step)
    updates["trace_steps"] = current_steps

    return updates
