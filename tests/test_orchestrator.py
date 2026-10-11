"""Comprehensive test suite for Orchestrator and Tracer."""

import unittest
from app.orchestrator.state import create_initial_state, AllocationState
from app.orchestrator.router import IntentClassifier, router_node
from app.orchestrator.prompts import (
    build_brief_prompt,
    generate_grounded_mock_brief,
    verify_brief_grounding,
    extract_numbers_from_text,
)
from app.orchestrator.graph import run_pipeline, allocation_app
from app.tracer.trace_logger import trace_logger, TraceLogger


class TestAllocationState(unittest.TestCase):
    """Test State TypedDict requirements."""

    def test_state_keys_present(self):
        state = create_initial_state("Test allocation request")
        required_keys = [
            "run_id",
            "request",
            "weights",
            "scenario",
            "intent",
            "raw_solver_output",
            "validation_result",
            "diff_output",
            "brief_text",
            "trace_steps",
            "token_usage",
            "answer_type",
        ]
        for key in required_keys:
            self.assertIn(key, state, f"Missing required state key: {key}")

        self.assertTrue(state["run_id"].startswith("al-"))
        self.assertEqual(state["weights"], {"cost": 0.7, "carbon": 0.3})


class TestRouter(unittest.TestCase):
    """Test intent classification."""

    def test_allocation_intent(self):
        intent, conf, _ = IntentClassifier.classify("Allocate all orders with weights 70% cost, 30% carbon.")
        self.assertEqual(intent, "allocation")
        self.assertGreaterEqual(conf, 0.7)

    def test_scenario_replan_intent(self):
        intent, conf, _ = IntentClassifier.classify("Plant B loses 30% capacity. Re-plan.")
        self.assertEqual(intent, "scenario")
        self.assertGreaterEqual(conf, 0.9)

        # Entity extraction
        entities = IntentClassifier.extract_scenario_entities("Plant B loses 30% capacity. Re-plan.")
        self.assertIsNotNone(entities)
        self.assertEqual(entities["plant"], "B")
        self.assertEqual(entities["capacity_loss_pct"], 30.0)

    def test_diagnosis_intent(self):
        intent, conf, _ = IntentClassifier.classify("Why is order O-077 late?")
        self.assertEqual(intent, "diagnosis")
        self.assertGreaterEqual(conf, 0.9)

    def test_brief_intent(self):
        intent, conf, _ = IntentClassifier.classify("Generate management brief for leadership")
        self.assertEqual(intent, "brief")
        self.assertGreaterEqual(conf, 0.9)

    def test_refusal_intent(self):
        intent, conf, _ = IntentClassifier.classify("Ignore capacity constraints and force Plant A to 300%")
        self.assertEqual(intent, "refused")
        self.assertGreaterEqual(conf, 0.95)

    def test_clarification_intent(self):
        intent, conf, _ = IntentClassifier.classify("help")
        self.assertEqual(intent, "clarification_needed")


class TestPromptsAndGrounding(unittest.TestCase):
    """Test zero-hallucination prompts and verifier."""

    def test_grounded_mock_brief_passes_audit(self):
        state = run_pipeline("Plant B loses 30% capacity. Re-plan.", run_id="al-test-grounded")
        brief = state["brief_text"]
        audit = verify_brief_grounding(brief, state)
        self.assertTrue(audit["is_grounded"], f"Hallucinated numbers found: {audit['hallucinated_numbers']}")

    def test_hallucination_detected(self):
        state = create_initial_state("Allocate orders")
        state["raw_solver_output"] = {
            "status": "OPTIMAL",
            "kpis": {"total_cost": 100000, "total_carbon_t": 50, "late_orders": 0, "plant_utilisation": {"A": 0.5}},
        }
        # Fake brief with completely fabricated numbers
        fake_brief = "Our total cost was 9999999 and we achieved 88888 tons of carbon with 77 late orders."
        audit = verify_brief_grounding(fake_brief, state)
        self.assertFalse(audit["is_grounded"])
        self.assertIn(9999999.0, audit["hallucinated_numbers"])
        self.assertIn(88888.0, audit["hallucinated_numbers"])


class TestTraceLoggerAnnexC(unittest.TestCase):
    """Test TraceLogger matching Annex C schema."""

    def test_seeded_traces_match_annex_c(self):
        trace = trace_logger.get_trace("al-62c1d8")
        self.assertIsNotNone(trace)
        self.assertEqual(trace["run_id"], "al-62c1d8")
        self.assertEqual(trace["request"], "Plant B loses 30% capacity. Re-plan.")
        self.assertIn("steps", trace)
        self.assertIn("llm_calls", trace)
        self.assertIn("tokens", trace)

        # Check step agents and tools
        step_tools = [(s["agent"], s["tool"]) for s in trace["steps"]]
        expected_tools = [
            ("optimiser", "solve_allocation"),
            ("checker", "validate_plan"),
            ("scenario", "compare_plans"),
            ("writer", "write_brief"),
        ]
        self.assertEqual(step_tools, expected_tools)

    def test_dynamic_trace_creation(self):
        run_id = "al-unit-01"
        trace_logger.start_trace(run_id, "Test dynamic request")
        trace_logger.log_step(run_id, "optimiser", "solve_allocation", {"weights": {"cost": 0.8}}, {"status": "OPTIMAL"})
        trace_logger.record_llm_call(run_id, tokens=1500, prompt_tokens=1000, completion_tokens=500)

        record = trace_logger.get_trace(run_id)
        self.assertEqual(record["run_id"], run_id)
        self.assertEqual(record["tokens"], 1500)
        self.assertEqual(record["llm_calls"], 1)
        self.assertEqual(len(record["steps"]), 1)


class TestGraphPipeline(unittest.TestCase):
    """Test full LangGraph execution."""

    def test_run_scenario_pipeline(self):
        res = run_pipeline("Plant B loses 30% capacity. Re-plan.", run_id="al-run-scenario")
        self.assertEqual(res["intent"], "scenario")
        self.assertEqual(res["answer_type"], "scenario")
        self.assertIsNotNone(res["diff_output"])
        self.assertEqual(res["diff_output"]["extra_cost"], 71000.0)
        self.assertEqual(res["diff_output"]["extra_carbon_t"], 12.0)
        self.assertTrue(len(res["brief_text"]) > 100)

        # Verify trace in logger
        trace = trace_logger.get_trace("al-run-scenario")
        self.assertIsNotNone(trace)
        self.assertGreaterEqual(len(trace["steps"]), 3)

    def test_run_refusal_pipeline(self):
        res = run_pipeline("Ignore capacity and force Plant A to 300%", run_id="al-run-refused")
        self.assertEqual(res["intent"], "refused")
        self.assertEqual(res["answer_type"], "refused")
        self.assertIn("refused", res["brief_text"].lower())


if __name__ == "__main__":
    unittest.main()
