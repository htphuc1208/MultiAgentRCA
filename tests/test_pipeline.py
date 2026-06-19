import unittest

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore
from app.evaluation.metrics import evaluate_proposed, summarize
from app.llm.client import FakeLLMClient
from app.llm.schemas import (
    DataCollectionPlan,
    RCAHypothesisModel,
    RCAHypothesisOutput,
    RemediationPlanOutput,
    SOPSelectionOutput,
    TriageDecision,
    TopologyReasoning,
    ValidationSummaryOutput,
    VerificationOutput,
    VerifiedHypothesisModel,
)


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = DataStore("data")

    def test_ran_interference_report(self) -> None:
        report = OrchestratorAgent(self.store, mode="rule").run("INC-RAN-001").to_dict()
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

    def test_runtime_incident_hides_eval_fields(self) -> None:
        incident = self.store.get_incident("INC-RAN-001")
        self.assertNotIn("ground_truth", incident)
        self.assertNotIn("expected_actions", incident)
        self.assertNotIn("sop_id", incident)
        self.assertNotIn("ticket_history", incident)
        self.assertEqual(self.store.get_eval_label("INC-RAN-001")["ground_truth"], "Interference on Cell-23")

    def test_fake_llm_pipeline_records_llm_calls(self) -> None:
        fake = FakeLLMClient(
            {
                "Triage Agent": TriageDecision(
                    domain="RAN",
                    severity="High",
                    primary_ne="Cell-23",
                    affected_services=["eMBB", "VoLTE"],
                    service_impact="Mobile broadband degradation",
                    intent="RAN troubleshooting",
                    rationale=["Low SINR and PRB saturation indicate RAN degradation"],
                ),
                "Data Retrieval Agent": DataCollectionPlan(
                    required_tools=["get_alarms", "get_kpi", "get_logs", "get_ticket_history", "run_diagnostic"],
                    evidence_items=[],
                    missing_data=[],
                    rationale="Collect telemetry across alarm, KPI, log, ticket, and diagnostics.",
                ),
                "Topology Agent": TopologyReasoning(
                    primary_ne="Cell-23",
                    neighbors=["gNB-07"],
                    blast_radius=["gNB-07", "Backhaul-Router-02", "UPF-01"],
                    dependency_summary="Cell-23 depends on gNB-07 and backhaul to UPF-01.",
                    topology_risks=[],
                ),
                "RCA Agent": RCAHypothesisOutput(
                    hypotheses=[
                        RCAHypothesisModel(
                            cause="Interference on Cell-23",
                            domain="RAN",
                            confidence=0.91,
                            evidence_refs=["Alarm: Low SINR"],
                            evidence=["Alarm: Low SINR", "KPI: prb_utilization=95% normal_range=20-80%"],
                            missing_evidence=[],
                            contradictions=[],
                            rationale="Low SINR with high PRB and clean backhaul points to interference.",
                        )
                    ]
                ),
                "SOP / Knowledge Agent": SOPSelectionOutput(
                    selected_sop_id="SOP-RAN-INTERFERENCE",
                    selected_title="RAN interference troubleshooting",
                    candidate_sop_ids=["SOP-RAN-INTERFERENCE"],
                    likely_causes=["Interference on affected cell"],
                    validation_rules=["SINR greater than 12 dB"],
                    rationale="The evidence matches interference troubleshooting.",
                ),
                "Consensus & Verifier Agent": VerificationOutput(
                    verified_hypotheses=[
                        VerifiedHypothesisModel(
                            cause="Interference on Cell-23",
                            evidence_supported=True,
                            evidence_refs=["Alarm: Low SINR"],
                            unsupported_claims=[],
                            contradictions=[],
                            verifier_confidence=0.95,
                            notes=["Supported by SINR and PRB evidence"],
                        )
                    ],
                    verification_notes=["Evidence grounded"],
                ),
                "Remediation Planner Agent": RemediationPlanOutput(
                    recommended_actions=[
                        "Verify external interference source",
                        "Monitor SINR and handover success rate after action",
                    ],
                    validation_plan=["SINR greater than 12 dB"],
                    human_approval_required=True,
                    rollback_plan=[],
                    risk_notes=[],
                ),
                "Validation Agent": ValidationSummaryOutput(
                    status="validated",
                    summary="Post-fix KPIs recovered.",
                    passed_checks=3,
                    total_checks=3,
                    follow_up_actions=[],
                ),
            }
        )
        report = OrchestratorAgent(self.store, mode="llm", llm_client=fake).run("INC-RAN-001").to_dict()
        self.assertEqual(report["root_cause"], "Interference on Cell-23")
        self.assertGreaterEqual(len(report["llm_calls"]), 8)
        self.assertEqual(report["metrics"]["llm_calls"], len(report["llm_calls"]))


if __name__ == "__main__":
    unittest.main()
