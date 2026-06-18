from __future__ import annotations

from app.agents.base import BaseAgent


class SOPAgent(BaseAgent):
    name = "SOP / Knowledge Agent"

    def run(self) -> dict[str, object]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        sop = self.tools.retrieve_sop(
            symptom=incident["symptom"],
            domain=triage["domain"],
            incident_id=incident["incident_id"],
        )
        output = {
            "sop": sop,
            "sop_id": sop.get("sop_id"),
            "likely_causes": sop.get("likely_causes", []),
            "validation_rules": sop.get("validation_rules", []),
        }
        self.blackboard.set("sop_context", output)
        return self._record(
            "retrieve operating procedure",
            f"Retrieved SOP {sop.get('sop_id')} for {triage['domain']} incident.",
            {"incident_id": incident["incident_id"], "symptom": incident["symptom"]},
            output,
            start,
        )

