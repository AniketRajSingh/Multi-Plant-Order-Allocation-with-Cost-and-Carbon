"""Intent classifier for multi-plant order allocation orchestrator.

Classifies incoming queries into:
- allocation: Normal optimal allocation request
- scenario: Capacity disturbance, what-if, or re-planning request
- diagnosis: Order feasibility, bottleneck analysis, or late-order root-cause
- brief: Executive summary, management brief, or trade-off briefing
- clarification_needed: Ambiguous or underspecified queries
- refused: Requests that violate physical constraints or prompt injections
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional, Tuple

from app.orchestrator.state import AllocationState, IntentType


class IntentClassifier:
    """Classifies user intent using deterministic heuristics with optional LLM augmentation."""

    # Refusal patterns: requests asking to violate physics, ignore constraints, or invent data
    REFUSAL_PATTERNS = [
        r"(?i)\b(?:ignore|bypass|override|disable)\b.*\b(?:capacity|constraint|eligibility|rules)\b",
        r"(?i)\b(?:force|push)\b.*\b(?:above|over|exceed)\b.*\b(?:100%|capacity|maximum)\b",
        r"(?i)\b(?:fake|invent|fudge|make up)\b.*\b(?:numbers?|costs?|carbon|results?)\b",
        r"(?i)\b(?:hack|jailbreak|disregard previous)\b",
    ]

    # Scenario & Re-plan patterns
    SCENARIO_PATTERNS = [
        r"(?i)\b(?:re-?plan|replan)\b",
        r"(?i)\bwhat[\s-]if\b",
        r"(?i)\bloses?\s+\d+%\s+capacity\b",
        r"(?i)\bcapacity\s+(?:loss|cut|drop|reduction|outage|event)\b",
        r"(?i)\b(?:shutdown|downtime|breakdown|offline|maintenance)\b",
        r"(?i)\bscenario\b",
        r"(?i)\bcarbon\s+tax\s+(?:increase|doubles?)\b",
    ]

    # Diagnosis & Root-cause patterns
    DIAGNOSIS_PATTERNS = [
        r"(?i)\bwhy\s+is\s+order\s+[\w-]+\s+(?:late|delayed|unallocated)\b",
        r"(?i)\bdiagnos(?:e|is)\b",
        r"(?i)\broot[\s-]cause\b",
        r"(?i)\bfeasib(?:le|ility)\b",
        r"(?i)\binfeasible\b",
        r"(?i)\bwhy\s+can['']?t\b",
        r"(?i)\bbottleneck\b",
        r"(?i)\beligib(?:le|ility)\b",
        r"(?i)\bcheck\s+order\b",
    ]

    # Executive Brief patterns
    BRIEF_PATTERNS = [
        r"(?i)\b(?:management|executive|leadership|board)\s+brief\b",
        r"(?i)\bwrite\s+(?:a\s+)?brief\b",
        r"(?i)\bsummariz(?:e|ation)\s+for\s+(?:executives?|leadership|management)\b",
        r"(?i)\bgenerate\s+brief\b",
        r"(?i)\bexecutive\s+summary\b",
    ]

    # Allocation & Optimization patterns
    ALLOCATION_PATTERNS = [
        r"(?i)\ballocat(?:e|ion)\b",
        r"(?i)\boptimi(?:ze|zation|se|sation)\b",
        r"(?i)\bsolve\b",
        r"(?i)\bassign\s+orders\b",
        r"(?i)\boptimal\s+plan\b",
        r"(?i)\brun\s+planner\b",
        r"(?i)\bgenerate\s+plan\b",
    ]

    @classmethod
    def classify(cls, request: str, scenario_data: Optional[Dict[str, Any]] = None) -> Tuple[IntentType, float, str]:
        """Classify request string into IntentType with confidence and reasoning.

        Returns:
            Tuple of (intent, confidence, reasoning)
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

        # 2. Clarification needed check (too short, ambiguous, or no actionable verb)
        if len(text) < 4 or text.lower() in {"help", "what", "do it", "run", "why", "plan", "?"}:
            return (
                "clarification_needed",
                0.95,
                "Request is too brief or ambiguous to determine required operational action.",
            )

        # If a scenario payload was explicitly provided in API request or text mentions re-plan
        if scenario_data and (
            scenario_data.get("plant") or scenario_data.get("capacity_loss_pct") or scenario_data.get("event_id")
        ):
            return "scenario", 0.98, "Explicit scenario / capacity event disturbance payload supplied."

        # 3. Scenario & Re-plan
        for pat in cls.SCENARIO_PATTERNS:
            if re.search(pat, text):
                return "scenario", 0.95, f"Matched scenario/re-plan pattern: {pat}"

        # 4. Diagnosis & Feasibility
        for pat in cls.DIAGNOSIS_PATTERNS:
            if re.search(pat, text):
                return "diagnosis", 0.94, f"Matched order diagnosis pattern: {pat}"

        # 5. Management Brief
        for pat in cls.BRIEF_PATTERNS:
            if re.search(pat, text):
                return "brief", 0.92, f"Matched management brief pattern: {pat}"

        # 6. Allocation
        for pat in cls.ALLOCATION_PATTERNS:
            if re.search(pat, text):
                return "allocation", 0.90, f"Matched allocation pattern: {pat}"

        # Default fallback: if orders or plants are mentioned, treat as allocation
        if re.search(r"(?i)\b(?:order|plant|cost|carbon|tradeoff|weight)\b", text):
            return "allocation", 0.75, "Defaulted to allocation based on operational domain keywords."

        return (
            "clarification_needed",
            0.60,
            "Could not determine intended action. Request should specify allocation, re-plan, diagnosis, or brief.",
        )

    @classmethod
    def extract_scenario_entities(cls, text: str) -> Optional[Dict[str, Any]]:
        """Extract plant and capacity loss percentage from natural language."""
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
            # Convert 70/30 to 0.7/0.3 if > 1
            if c > 1.0 or k > 1.0:
                total = c + k
                return {"cost": round(c / total, 2), "carbon": round(k / total, 2)}
            return {"cost": c, "carbon": k}
        return None


def router_node(state: AllocationState) -> Dict[str, Any]:
    """LangGraph node that classifies intent and updates state."""
    start_time = time.time()
    req = state.get("request", "")
    existing_scenario = state.get("scenario")

    intent, confidence, reason = IntentClassifier.classify(req, scenario_data=existing_scenario)

    updates: Dict[str, Any] = {
        "intent": intent,
    }

    # Extract entities if missing from state
    if intent == "scenario" and not existing_scenario:
        extracted = IntentClassifier.extract_scenario_entities(req)
        if extracted:
            updates["scenario"] = extracted

    # Extract weights if user specified them in natural language
    extracted_weights = IntentClassifier.extract_weights(req)
    if extracted_weights and not state.get("weights"):
        updates["weights"] = extracted_weights

    # Map intent to initial answer_type
    if intent == "refused":
        updates["answer_type"] = "refused"
    elif intent == "clarification_needed":
        updates["answer_type"] = "clarification_needed"
    elif intent == "scenario":
        updates["answer_type"] = "scenario"
    elif intent == "brief":
        updates["answer_type"] = "brief"
    else:
        updates["answer_type"] = "allocation"

    # Add trace step matching Annex C
    step = {
        "agent": "router",
        "tool": "classify_intent",
        "input": {"request": req},
        "output": {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reason,
        },
        "duration_ms": round((time.time() - start_time) * 1000, 2),
    }

    current_steps = list(state.get("trace_steps") or [])
    current_steps.append(step)
    updates["trace_steps"] = current_steps

    return updates
