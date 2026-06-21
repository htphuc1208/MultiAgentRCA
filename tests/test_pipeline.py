import json
import os
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore
from app.evaluation.metrics import evaluate_proposed, summarize
from app.llm.client import DeepSeekLLMClient, FakeLLMClient, MissingAPIKeyError
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
from app.tools.registry import telecom_tool_definitions


class FakeDeepSeekChatClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake DeepSeek response queued")
        return self.responses.pop(0)


def deepseek_response(content: str | None = None, tool_calls: list[object] | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 7})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def deepseek_tool_call(name: str, arguments: dict) -> object:
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
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

    def test_deepseek_tool_schema_conversion(self) -> None:
        converted = DeepSeekLLMClient.to_deepseek_tools(telecom_tool_definitions(["get_alarms"]))
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "get_alarms")
        self.assertIn("parameters", converted[0]["function"])
        self.assertNotIn("strict", converted[0]["function"])

    def test_deepseek_structured_json_without_tools(self) -> None:
        client = FakeDeepSeekChatClient(
            [
                deepseek_response(
                    content=json.dumps(
                        {
                            "domain": "RAN",
                            "severity": "High",
                            "primary_ne": "Cell-23",
                            "affected_services": ["eMBB"],
                            "service_impact": "Mobile broadband degradation",
                            "intent": "RAN troubleshooting",
                            "rationale": ["Low SINR"],
                        }
                    )
                )
            ]
        )
        parsed, call = DeepSeekLLMClient(api_key="test", client=client, thinking="disabled").structured(
            agent="Triage Agent",
            system_prompt="Classify and return json.",
            user_payload={"incident": {"incident_id": "INC-RAN-001"}},
            response_model=TriageDecision,
        )

        self.assertEqual(parsed.domain, "RAN")
        self.assertEqual(call.token_usage["total_tokens"], 7)
        self.assertEqual(client.requests[-1]["response_format"], {"type": "json_object"})
        self.assertEqual(client.requests[-1]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_deepseek_structured_tool_loop_records_tool_calls(self) -> None:
        client = FakeDeepSeekChatClient(
            [
                deepseek_response(
                    tool_calls=[
                        deepseek_tool_call(
                            "get_alarms",
                            {"ne_id": "Cell-23", "time_window": None, "incident_id": "INC-RAN-001"},
                        )
                    ]
                ),
                deepseek_response(
                    content=json.dumps(
                        {
                            "required_tools": ["get_alarms"],
                            "evidence_items": ["Alarm: Low SINR"],
                            "missing_data": [],
                            "rationale": "Collected active alarms.",
                        }
                    )
                ),
            ]
        )

        def execute_tool(name: str, arguments: dict):
            self.assertEqual(name, "get_alarms")
            self.assertEqual(arguments["incident_id"], "INC-RAN-001")
            return ["Low SINR"]

        parsed, call = DeepSeekLLMClient(api_key="test", client=client).structured(
            agent="Data Retrieval Agent",
            system_prompt="Collect telemetry and return json.",
            user_payload={"incident": {"incident_id": "INC-RAN-001"}},
            response_model=DataCollectionPlan,
            tools=telecom_tool_definitions(["get_alarms"]),
            tool_executor=execute_tool,
            max_tool_calls=1,
        )

        self.assertEqual(parsed.evidence_items, ["Alarm: Low SINR"])
        self.assertEqual(call.tool_calls[0]["name"], "get_alarms")
        self.assertIn("tools", client.requests[0])
        self.assertNotIn("tools", client.requests[-1])

    def test_deepseek_missing_api_key_error(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaisesRegex(MissingAPIKeyError, "DEEPSEEK_API_KEY"):
                DeepSeekLLMClient()


if __name__ == "__main__":
    unittest.main()
