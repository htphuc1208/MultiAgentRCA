from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class TriageAgent(BaseAgent):
    name = "Triage Agent"

    def run(self) -> dict[str, Any]:
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

