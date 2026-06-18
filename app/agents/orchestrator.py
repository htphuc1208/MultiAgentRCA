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
from app.memory.blackboard import Blackboard
from app.models import RCAReport
from app.tools.telecom_tools import TelecomToolbox


class OrchestratorAgent:
    """Supervisor that executes the end-to-end RCA workflow."""

    name = "Orchestrator Agent"
    
    # DataStore: Interface to access incident data, topology, historical incidents, etc.
    def __init__(self, store: DataStore) -> None:
        self.store = store

    def run(self, incident_id: str) -> RCAReport:
        blackboard = Blackboard()
        tools = TelecomToolbox(self.store, blackboard)
        incident = self.store.get_incident(incident_id)
        blackboard.set("incident", incident)
        blackboard.record_agent(
            agent=self.name,
            action="accept incident",
            summary=f"Accepted incident {incident_id} and initialized shared blackboard.",
            inputs={"incident_id": incident_id},
            outputs={"symptom": incident["symptom"], "primary_ne": incident["primary_ne"]},
        )

        for agent_cls in [
            TriageAgent,
            DataRetrievalAgent,
            TopologyAgent,
            RCAAgent,
            SOPAgent,
            ConsensusVerifierAgent,
            RemediationPlannerAgent,
            ValidationAgent,
        ]:
            agent_cls(blackboard, tools).run()

        plan = blackboard.get("remediation_plan")
        selected = blackboard.get("selected_hypothesis")
        validation = blackboard.get("validation_result")
        triage = blackboard.get("triage")
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
        )
        return report

    def _report_metrics(self, blackboard: Blackboard, incident: dict[str, Any]) -> dict[str, Any]:
        selected = blackboard.get("selected_hypothesis")
        tool_names = [call.name for call in blackboard.tool_calls]
        evidence = blackboard.get("data_evidence", {}).get("evidence_items", [])
        return {
            "tool_calls": len(tool_names),
            "unique_tools": sorted(set(tool_names)),
            "agent_steps": len(blackboard.trace),
            "evidence_items": len(evidence),
            "matches_ground_truth": selected.cause.lower() == incident["ground_truth"].lower(),
        }

