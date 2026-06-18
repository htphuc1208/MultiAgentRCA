from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class RemediationPlannerAgent(BaseAgent):
    name = "Remediation Planner Agent"

    def run(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        selected = self.blackboard.get("selected_hypothesis")
        sop = self.blackboard.get("sop_context")["sop"]
        actions = self._plan_actions(selected.cause, sop)
        validation_plan = sop.get("validation_rules", [])
        output = {
            "root_cause": selected.cause,
            "confidence": selected.scores["consensus_score"],
            "recommended_actions": actions,
            "validation_plan": validation_plan,
            "human_approval_required": True,
        }
        self.blackboard.set("remediation_plan", output)
        return self._record(
            "plan remediation",
            f"Prepared {len(actions)} ordered remediation steps with human approval gate.",
            {"incident_id": incident["incident_id"], "root_cause": selected.cause},
            output,
            start,
        )

    def _plan_actions(self, root_cause: str, sop: dict[str, Any]) -> list[str]:
        steps = list(sop.get("steps", []))
        if not steps:
            steps = [f"Investigate root cause: {root_cause}"]
        return ["Request human approval before network-impacting action", *steps]

