from __future__ import annotations

from typing import Any

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore
from app.llm.client import BaseLLMClient, OpenAILLMClient
from app.llm.prompts import STRICT_EVIDENCE_POLICY
from app.llm.schemas import RCAHypothesisOutput
from app.memory.blackboard import Blackboard
from app.models import Hypothesis, RCAReport
from app.tools.registry import TelecomToolExecutor, telecom_tool_definitions
from app.tools.telecom_tools import TelecomToolbox


SINGLE_AGENT_PROMPT = f"""
You are a single ReAct-style telecom RCA agent. Use the available tools to gather enough data,
then produce up to three root-cause hypotheses.
{STRICT_EVIDENCE_POLICY}
"""


class SingleAgentRunner:
    def __init__(
        self,
        store: DataStore,
        *,
        mode: str = "rule",
        llm_client: BaseLLMClient | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tool_calls: int = 8,
    ) -> None:
        self.store = store
        self.mode = mode
        self.llm_client = llm_client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls

    def run(self, incident_id: str) -> RCAReport:
        if self.mode != "llm":
            return OrchestratorAgent(self.store, mode="rule", use_consensus=False).run(incident_id)
        return self._run_llm(incident_id)

    def _run_llm(self, incident_id: str) -> RCAReport:
        blackboard = Blackboard()
        tools = TelecomToolbox(self.store, blackboard)
        incident = self.store.get_incident(incident_id)
        blackboard.set("incident", incident)
        llm_client = self.llm_client or OpenAILLMClient(model=self.model, reasoning_effort=self.reasoning_effort)
        parsed, llm_call = llm_client.structured(
            agent="Single ReAct-style Agent",
            system_prompt=SINGLE_AGENT_PROMPT,
            user_payload={
                "incident": self._incident_stub(incident),
                "available_kpis": list(incident.get("kpis", {})),
                "available_diagnostics": list(incident.get("diagnostics", {})),
            },
            response_model=RCAHypothesisOutput,
            tools=telecom_tool_definitions(
                [
                    "get_alarms",
                    "get_kpi",
                    "get_logs",
                    "get_topology",
                    "get_ticket_history",
                    "retrieve_sop",
                    "run_diagnostic",
                ]
            ),
            tool_executor=TelecomToolExecutor(tools),
            max_tool_calls=self.max_tool_calls,
        )
        blackboard.record_llm(llm_call)
        hypotheses = [
            Hypothesis(
                cause=item.cause,
                domain=item.domain,
                confidence=item.confidence,
                evidence=item.evidence,
                evidence_refs=item.evidence_refs,
                missing_evidence=item.missing_evidence,
                contradictions=item.contradictions,
                source_agents=["Single ReAct-style Agent"],
                scores={
                    "evidence_match": min(1.0, len(item.evidence) / 4),
                    "topology_consistency": 0.0,
                    "sop_alignment": 0.0,
                    "historical_similarity": 0.0,
                    "agent_vote_confidence": item.confidence,
                    "consensus_score": item.confidence,
                },
            )
            for item in parsed.hypotheses
        ]
        selected = sorted(hypotheses, key=lambda hypothesis: hypothesis.confidence, reverse=True)[0]
        selected_sop = self._selected_sop(llm_call.tool_calls)
        recommended_actions = ["Request human approval before network-impacting action"]
        recommended_actions.extend(selected_sop.get("steps", []))
        validation = tools.validate_after_fix(incident["primary_ne"], incident_id)
        checks = validation.get("checks", [])
        passed = sum(1 for check in checks if check.get("passed"))
        token_usage = llm_call.token_usage or {"total_tokens": 0}
        report = RCAReport(
            incident_id=incident_id,
            domain=incident["domain"],
            severity=incident["severity"],
            root_cause=selected.cause,
            confidence=selected.confidence,
            evidence=selected.evidence,
            hypotheses=hypotheses,
            recommended_actions=recommended_actions,
            validation_plan=selected_sop.get("validation_rules", []),
            validation_result={
                "validation_result": validation,
                "passed_checks": passed,
                "total_checks": len(checks),
                "status": "validated" if checks and passed == len(checks) else "needs_review",
            },
            trace=blackboard.trace,
            metrics={
                "tool_calls": len(blackboard.tool_calls),
                "unique_tools": sorted({call.name for call in blackboard.tool_calls}),
                "agent_steps": len(blackboard.trace),
                "evidence_items": len(selected.evidence),
                "llm_calls": len(blackboard.llm_calls),
                "matches_ground_truth": selected.cause.lower()
                == self.store.get_eval_label(incident_id)["ground_truth"].lower(),
            },
            llm_calls=blackboard.llm_calls,
            token_usage=token_usage,
            latency_ms=llm_call.latency_ms,
            evidence_refs=selected.evidence_refs,
            selected_sop=selected_sop,
            verification_notes=[],
        )
        return report

    def _selected_sop(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        for call in tool_calls:
            if call["name"] == "retrieve_sop":
                return call.get("output") or {}
        return {}

    def _incident_stub(self, incident: dict[str, Any]) -> dict[str, Any]:
        return {
            "incident_id": incident["incident_id"],
            "domain": incident.get("domain"),
            "severity": incident.get("severity"),
            "symptom": incident.get("symptom"),
            "description": incident.get("description"),
            "time_window": incident.get("time_window"),
            "primary_ne": incident.get("primary_ne"),
            "service_impact": incident.get("service_impact"),
            "affected_services": incident.get("affected_services", []),
        }

