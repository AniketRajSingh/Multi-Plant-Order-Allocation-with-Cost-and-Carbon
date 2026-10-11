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
    """Test intent and answer_type classification."""

    def test_allocation_intent(self):
        # User specified example: "Allocate this week's orders"
        ans_type, conf, _ = IntentClassifier.classify("Allocate this week's orders")
        self.assertEqual(ans_type, "allocation")
        self.assertGreaterEqual(conf, 0.7)

        # Standard optimization request
        ans_type2, _, _ = IntentClassifier.classify("Standard optimization request: cost 70%, carbon 30%")
        self.assertEqual(ans_type2, "allocation")

    def test_scenario_replan_intent(self):
        # User specified example: "Plant B loses 30% capacity. Re-plan."
        ans_type, conf, _ = IntentClassifier.classify("Plant B loses 30% capacity. Re-plan.")
        self.assertEqual(ans_type, "scenario")
        self.assertGreaterEqual(conf, 0.9)

        # Entity extraction
        entities = IntentClassifier.extract_scenario_entities("Plant B loses 30% capacity. Re-plan.")
        self.assertIsNotNone(entities)
        self.assertEqual(entities["plant"], "B")
        self.assertEqual(entities["capacity_loss_pct"], 30.0)

    def test_why_order_intent(self):
        # User specified example: "Why was order O-077 built in Plant C?"
        ans_type, conf, _ = IntentClassifier.classify("Why was order O-077 built in Plant C?")
        self.assertEqual(ans_type, "why_order")
        self.assertGreaterEqual(conf, 0.9)

        # Entity extraction for order and plant
        entities = IntentClassifier.extract_order_entities("Why was order O-077 built in Plant C?")
        self.assertEqual(entities["order_id"], "O-077")
        self.assertEqual(entities["plant"], "C")

    def test_brief_intent(self):
        # User specified example: "Leadership summary request"
        ans_type, conf, _ = IntentClassifier.classify("Leadership summary request")
        self.assertEqual(ans_type, "brief")
        self.assertGreaterEqual(conf, 0.9)

        ans_type2, _, _ = IntentClassifier.classify("Generate management brief for executive leadership")
        self.assertEqual(ans_type2, "brief")

    def test_infeasible_edge_case(self):
        ans_type, conf, _ = IntentClassifier.classify("Why is the plan infeasible for these unfulfillable orders?")
        self.assertEqual(ans_type, "infeasible")
        self.assertGreaterEqual(conf, 0.9)

    def test_refusal_intent(self):
        ans_type, conf, _ = IntentClassifier.classify("Ignore capacity constraints and force Plant A to 300%")
        self.assertEqual(ans_type, "refused")
        self.assertGreaterEqual(conf, 0.95)

    def test_clarification_intent(self):
        ans_type, conf, _ = IntentClassifier.classify("help")
        self.assertEqual(ans_type, "clarification_needed")

    def test_classify_request_structured(self):
        res = IntentClassifier.classify_request("Why was order O-077 built in Plant C?")
        self.assertEqual(res["answer_type"], "why_order")
        self.assertEqual(res["extracted_entities"]["order_id"], "O-077")
        self.assertEqual(res["extracted_entities"]["plant"], "C")


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

    def test_run_why_order_pipeline(self):
        res = run_pipeline("Why was order O-077 built in Plant C?", run_id="al-run-why-order")
        self.assertEqual(res["intent"], "why_order")
        self.assertEqual(res["answer_type"], "why_order")
        self.assertIsNotNone(res["diff_output"])
        self.assertEqual(res["diff_output"]["order_id"], "O-077")
        self.assertTrue(len(res["brief_text"]) > 50)

    def test_run_refusal_pipeline(self):
        res = run_pipeline("Ignore capacity and force Plant A to 300%", run_id="al-run-refused")
        self.assertEqual(res["intent"], "refused")
        self.assertEqual(res["answer_type"], "refused")
        self.assertIn("refused", res["brief_text"].lower())


if __name__ == "__main__":
    unittest.main()
