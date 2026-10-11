"""Trace auditing logger matching Annex C schema for GET /trace/{run_id}.

Section 6 & Annex C mandate:
GET /trace/{run_id} returns an audit record formatted as:
{
  "run_id": "al-62c1d8",
  "request": "Plant B loses 30% capacity. Re-plan.",
  "steps": [
    {"agent": "optimiser", "tool": "solve_allocation", "input": {"weights": {"cost": 0.7, "carbon": 0.3}}, "output": {"status": "OPTIMAL"}},
    {"agent": "checker", "tool": "validate_plan", "output": {"capacity_ok": true, "eligibility_ok": true}},
    {"agent": "scenario", "tool": "compare_plans", "output": {"extra_cost": 71000, "extra_carbon_t": 12}},
    {"agent": "writer", "tool": "write_brief"}
  ],
  "llm_calls": 2,
  "tokens": 2900
}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("trace_logger")


class TraceLogger:
    """Thread-safe trace auditing manager for LangGraph pipeline runs."""

    def __init__(self, persistence_dir: Optional[str] = "data/traces"):
        self._lock = threading.Lock()
        self._traces: Dict[str, Dict[str, Any]] = {}
        self.persistence_dir = Path(persistence_dir) if persistence_dir else None
        if self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_traces()

        # Seed the 3 required sample traces if none exist
        if not self._traces:
            self.seed_default_sample_traces()

    def _load_persisted_traces(self) -> None:
        """Load any existing trace JSON files from disk."""
        if not self.persistence_dir or not self.persistence_dir.exists():
            return

        for filepath in self.persistence_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "run_id" in data:
                        self._traces[data["run_id"]] = data
            except Exception as e:
                logger.warning(f"Could not load trace from {filepath}: {e}")

    def _persist_trace_to_disk(self, run_id: str) -> None:
        """Save a trace record to JSON file."""
        if not self.persistence_dir:
            return
        trace_data = self._traces.get(run_id)
        if not trace_data:
            return

        try:
            filepath = self.persistence_dir / f"{run_id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist trace {run_id}: {e}")

    def start_trace(self, run_id: str, request: str) -> Dict[str, Any]:
        """Initialize a new trace record for a run."""
        with self._lock:
            record = {
                "run_id": run_id,
                "request": request,
                "steps": [],
                "llm_calls": 0,
                "tokens": 0,
                "_created_at": time.time(),
                "_token_breakdown": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }
            self._traces[run_id] = record
            self._persist_trace_to_disk(run_id)
            return record

    def log_step(
        self,
        run_id: str,
        agent: str,
        tool: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Append an agent tool call step matching Annex C schema.

        Schema strictly requires:
        - agent (e.g. 'optimiser', 'checker', 'scenario', 'writer')
        - tool (e.g. 'solve_allocation', 'validate_plan', 'compare_plans', 'write_brief')
        - optional input
        - optional output
        """
        with self._lock:
            if run_id not in self._traces:
                self._traces[run_id] = {
                    "run_id": run_id,
                    "request": "",
                    "steps": [],
                    "llm_calls": 0,
                    "tokens": 0,
                }

            step: Dict[str, Any] = {
                "agent": agent,
                "tool": tool,
            }
            if input_data is not None:
                step["input"] = input_data
            if output_data is not None:
                step["output"] = output_data

            self._traces[run_id]["steps"].append(step)
            self._persist_trace_to_disk(run_id)

    def record_llm_call(
        self,
        run_id: str,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Increment LLM invocation counter and token counts."""
        with self._lock:
            if run_id not in self._traces:
                return

            self._traces[run_id]["llm_calls"] = self._traces[run_id].get("llm_calls", 0) + 1
            calculated_tokens = tokens if tokens > 0 else (prompt_tokens + completion_tokens)
            self._traces[run_id]["tokens"] = self._traces[run_id].get("tokens", 0) + calculated_tokens

            if "_token_breakdown" not in self._traces[run_id]:
                self._traces[run_id]["_token_breakdown"] = {"prompt_tokens": 0, "completion_tokens": 0}
            self._traces[run_id]["_token_breakdown"]["prompt_tokens"] += prompt_tokens
            self._traces[run_id]["_token_breakdown"]["completion_tokens"] += completion_tokens

            self._persist_trace_to_disk(run_id)

    def get_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace formatted strictly according to Annex C.

        Filters out internal metadata underscores.
        """
        with self._lock:
            trace = self._traces.get(run_id)
            if not trace:
                return None

            # Return Annex C compliant structure
            return {
                "run_id": trace["run_id"],
                "request": trace.get("request", ""),
                "steps": trace.get("steps", []),
                "llm_calls": trace.get("llm_calls", 0),
                "tokens": trace.get("tokens", 0),
            }

    def list_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List summary of traces."""
        with self._lock:
            out = []
            for t in list(self._traces.values())[-limit:]:
                out.append({
                    "run_id": t["run_id"],
                    "request": t.get("request", ""),
                    "step_count": len(t.get("steps", [])),
                    "llm_calls": t.get("llm_calls", 0),
                    "tokens": t.get("tokens", 0),
                })
            return out

    def seed_default_sample_traces(self) -> None:
        """Preload the 3 mandatory sample trace records required by Section 8 and Annex C.

        Includes:
        1. al-62c1d8: Exact Annex C scenario re-plan record.
        2. al-98a101: Baseline multi-objective allocation record.
        3. al-44e2b0: Order diagnosis / infeasible constraint analysis record.
        """
        # 1. Exact Annex C Sample Record
        sample_scenario = {
            "run_id": "al-62c1d8",
            "request": "Plant B loses 30% capacity. Re-plan.",
            "steps": [
                {
                    "agent": "optimiser",
                    "tool": "solve_allocation",
                    "input": {"weights": {"cost": 0.7, "carbon": 0.3}},
                    "output": {"status": "OPTIMAL"},
                },
                {
                    "agent": "checker",
                    "tool": "validate_plan",
                    "output": {"capacity_ok": True, "eligibility_ok": True},
                },
                {
                    "agent": "scenario",
                    "tool": "compare_plans",
                    "output": {"extra_cost": 71000, "extra_carbon_t": 12},
                },
                {
                    "agent": "writer",
                    "tool": "write_brief",
                },
            ],
            "llm_calls": 2,
            "tokens": 2900,
        }

        # 2. Baseline Allocation Record
        sample_allocation = {
            "run_id": "al-98a101",
            "request": "Allocate all orders with weights 70% cost, 30% carbon.",
            "steps": [
                {
                    "agent": "router",
                    "tool": "classify_intent",
                    "input": {"request": "Allocate all orders with weights 70% cost, 30% carbon."},
                    "output": {"intent": "allocation", "confidence": 0.95},
                },
                {
                    "agent": "optimiser",
                    "tool": "solve_allocation",
                    "input": {"weights": {"cost": 0.7, "carbon": 0.3}},
                    "output": {"status": "OPTIMAL", "total_cost": 1840000, "total_carbon_t": 410, "late_orders": 2},
                },
                {
                    "agent": "checker",
                    "tool": "validate_plan",
                    "output": {"capacity_ok": True, "eligibility_ok": True, "delivery_ok": True},
                },
                {
                    "agent": "writer",
                    "tool": "write_brief",
                    "output": {"grounded": True},
                },
            ],
            "llm_calls": 1,
            "tokens": 1450,
        }

        # 3. Order Diagnosis / Infeasible Record
        sample_diagnosis = {
            "run_id": "al-44e2b0",
            "request": "Why is order O-077 arriving late if allocated to Plant A?",
            "steps": [
                {
                    "agent": "router",
                    "tool": "classify_intent",
                    "input": {"request": "Why is order O-077 arriving late if allocated to Plant A?"},
                    "output": {"intent": "diagnosis", "confidence": 0.94},
                },
                {
                    "agent": "diagnostician",
                    "tool": "diagnose_order",
                    "input": {"order_id": "O-077", "candidate_plant": "A"},
                    "output": {
                        "cheapest_plant": "A",
                        "cheapest_cost": 17400,
                        "transit_days": 7,
                        "days_late": 3,
                        "on_time_alternative": "C",
                        "alternative_cost": 18200,
                    },
                },
                {
                    "agent": "checker",
                    "tool": "validate_plan",
                    "output": {"delivery_ok": False, "violation": "Order O-077 misses due date by 3 days at Plant A"},
                },
                {
                    "agent": "writer",
                    "tool": "write_brief",
                },
            ],
            "llm_calls": 1,
            "tokens": 1280,
        }

        for sample in [sample_scenario, sample_allocation, sample_diagnosis]:
            self._traces[sample["run_id"]] = sample
            self._persist_trace_to_disk(sample["run_id"])


# Global singleton instance
trace_logger = TraceLogger()
