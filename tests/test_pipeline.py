import unittest

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore
from app.evaluation.metrics import evaluate_proposed, summarize


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = DataStore("data")

    def test_ran_interference_report(self) -> None:
        report = OrchestratorAgent(self.store).run("INC-RAN-001").to_dict()
        self.assertEqual(report["root_cause"], "Interference on Cell-23")
        self.assertGreaterEqual(report["confidence"], 0.7)
        self.assertIn("retrieve_sop", report["metrics"]["unique_tools"])
        self.assertEqual(report["validation_result"]["status"], "validated")

    def test_proposed_evaluation_accuracy(self) -> None:
        results = evaluate_proposed(self.store)
        summary = summarize(results)[0]
        self.assertEqual(summary["incidents"], 12)
        self.assertGreaterEqual(summary["rca_accuracy"], 0.9)
        self.assertEqual(summary["hallucination_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

