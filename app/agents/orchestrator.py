from __future__ import annotations

from typing import Any

from app.agents.data_agent import DataRetrievalAgent
from app.agents.planner_agent import RemediationPlannerAgent
from app.agents.rca_agent import RCAAgent
from app.agents.sop_agent import SOPAgent
from app.agents.topology_agent import TopologyAgent
from app.agents.triage_agent import TriageAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.verifier_agent import ConsensusVerifierAgent
from app.data_store import DataStore
from app.llm.client import BaseLLMClient, OpenAILLMClient
from app.memory.blackboard import Blackboard
from app.models import Hypothesis
from app.models import RCAReport
from app.tools.telecom_tools import TelecomToolbox


class OrchestratorAgent:
    """Supervisor that executes the end-to-end RCA workflow."""

    name = "Orchestrator Agent"
    
    # DataStore: Interface to access incident data, topology, historical incidents, etc.
    def __init__(
        self,
        store: DataStore,
        *,
        mode: str = "rule",
        llm_client: BaseLLMClient | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tool_calls: int = 6,
        use_consensus: bool = True,
    ) -> None:
        self.store = store
        self.mode = mode
        self.llm_client = llm_client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls
        self.use_consensus = use_consensus

    def run(self, incident_id: str) -> RCAReport:
        blackboard = Blackboard()
        tools = TelecomToolbox(self.store, blackboard)
        llm_client = self._llm_client()
        incident = self.store.get_incident(incident_id)
        blackboard.set("incident", incident)
        blackboard.record_agent(
            agent=self.name,
            action="accept incident",
            summary=f"Accepted incident {incident_id} and initialized shared blackboard.",
            inputs={"incident_id": incident_id},
            outputs={"symptom": incident["symptom"], "primary_ne": incident["primary_ne"]},
        )

        pre_selection_agents = [
            TriageAgent,
            DataRetrievalAgent,
            TopologyAgent,
            RCAAgent,
            SOPAgent,
        ]

        for agent_cls in pre_selection_agents:
            agent_cls(
                blackboard,
                tools,
                mode=self.mode,
                llm_client=llm_client,
                max_tool_calls=self.max_tool_calls,
            ).run()

        if self.use_consensus:
            ConsensusVerifierAgent(
                blackboard,
                tools,
                mode=self.mode,
                llm_client=llm_client,
                max_tool_calls=self.max_tool_calls,
            ).run()
        else:
            self._select_without_consensus(blackboard)

        for agent_cls in [RemediationPlannerAgent, ValidationAgent]:
            agent_cls(
                blackboard,
                tools,
                mode=self.mode,
                llm_client=llm_client,
                max_tool_calls=self.max_tool_calls,
            ).run()

        plan = blackboard.get("remediation_plan")
        selected = blackboard.get("selected_hypothesis")
        validation = blackboard.get("validation_result")
        triage = blackboard.get("triage")
        token_usage = self._token_usage(blackboard)
        report = RCAReport(
            incident_id=incident_id,
            domain=triage["domain"],
            severity=triage["severity"],
            root_cause=selected.cause,
            confidence=selected.scores["consensus_score"],
            evidence=selected.evidence,
            hypotheses=blackboard.get("verified_hypotheses"),
            recommended_actions=plan["recommended_actions"],
            validation_plan=plan["validation_plan"],
            validation_result=validation,
            trace=blackboard.trace,
            metrics=self._report_metrics(blackboard, incident),
            llm_calls=blackboard.llm_calls,
            token_usage=token_usage,
            latency_ms=sum(call.get("latency_ms", 0) for call in blackboard.llm_calls),
            evidence_refs=selected.evidence_refs,
            selected_sop=blackboard.get("sop_context", {}).get("sop", {}),
            verification_notes=selected.verification_notes,
        )
        return report

    def _report_metrics(self, blackboard: Blackboard, incident: dict[str, Any]) -> dict[str, Any]:
        selected = blackboard.get("selected_hypothesis")
        tool_names = [call.name for call in blackboard.tool_calls]
        evidence = blackboard.get("data_evidence", {}).get("evidence_items", [])
        try:
            label = self.store.get_eval_label(incident["incident_id"])
            matches_ground_truth = selected.cause.lower() == label["ground_truth"].lower()
        except KeyError:
            matches_ground_truth = False
        return {
            "tool_calls": len(tool_names),
            "unique_tools": sorted(set(tool_names)),
            "agent_steps": len(blackboard.trace),
            "evidence_items": len(evidence),
            "llm_calls": len(blackboard.llm_calls),
            "matches_ground_truth": matches_ground_truth,
        }

    def _llm_client(self) -> BaseLLMClient | None:
        if self.mode != "llm":
            return None
        if self.llm_client is not None:
            return self.llm_client
        return OpenAILLMClient(model=self.model, reasoning_effort=self.reasoning_effort)

    def _select_without_consensus(self, blackboard: Blackboard) -> None:
        hypotheses: list[Hypothesis] = blackboard.get("hypotheses", [])
        if not hypotheses:
            raise RuntimeError("No hypotheses available for no-consensus selection")
        selected = sorted(hypotheses, key=lambda hypothesis: hypothesis.confidence, reverse=True)[0]
        selected.scores = {
            "evidence_match": min(1.0, len(selected.evidence) / 4),
            "topology_consistency": 0.0,
            "sop_alignment": 0.0,
            "historical_similarity": 0.0,
            "agent_vote_confidence": selected.confidence,
            "consensus_score": selected.confidence,
        }
        blackboard.set("verified_hypotheses", hypotheses)
        blackboard.set("selected_hypothesis", selected)
        blackboard.record_agent(
            agent=self.name,
            action="select without consensus",
            summary=f"Selected top RCA hypothesis without verifier consensus: {selected.cause}.",
            inputs={"hypothesis_count": len(hypotheses)},
            outputs={"selected_hypothesis": selected.to_dict()},
        )

    def _token_usage(self, blackboard: Blackboard) -> dict[str, Any]:
        totals: dict[str, Any] = {"total_tokens": 0}
        for call in blackboard.llm_calls:
            usage = call.get("token_usage", {})
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
        return totals
