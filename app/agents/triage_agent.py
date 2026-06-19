from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.prompts import TRIAGE_PROMPT
from app.llm.schemas import TriageDecision


class TriageAgent(BaseAgent):
    name = "Triage Agent"

    def run(self) -> dict[str, Any]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=TRIAGE_PROMPT,
            user_payload={"incident": incident},
            response_model=TriageDecision,
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        output = parsed.model_dump()
        self.blackboard.set("triage", output)
        return self._record(
            "llm classify incident",
            f"LLM classified incident as {output['domain']} with {output['severity']} severity.",
            {"incident_id": incident["incident_id"], "symptom": incident["symptom"]},
            output,
            start,
        )

    def _run_rule(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        domain = incident.get("domain") or self._classify_domain(incident.get("symptom", ""))
        severity = incident.get("severity", "Medium")
        affected_services = incident.get("affected_services", [])
        output = {
            "domain": domain,
            "severity": severity,
            "primary_ne": incident["primary_ne"],
            "affected_services": affected_services,
            "intent": f"{domain} troubleshooting",
            "service_impact": incident.get("service_impact", "Unknown"),
        }
        self.blackboard.set("triage", output)
        return self._record(
            "classify incident",
            f"Classified incident as {domain} with {severity} severity.",
            {"incident_id": incident["incident_id"], "symptom": incident["symptom"]},
            output,
            start,
        )

    def _classify_domain(self, symptom: str) -> str:
        text = symptom.lower()
        if any(term in text for term in ["cell", "gnb", "handover", "sinr", "rsrp"]):
            return "RAN"
        if any(term in text for term in ["amf", "smf", "upf", "pdu", "registration"]):
            return "Core"
        return "Transport"
