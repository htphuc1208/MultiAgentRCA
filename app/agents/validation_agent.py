from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class ValidationAgent(BaseAgent):
    name = "Validation Agent"

    def run(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        result = self.tools.validate_after_fix(triage["primary_ne"], incident["incident_id"])
        checks = result.get("checks", [])
        passed = sum(1 for check in checks if check.get("passed"))
        output = {
            "validation_result": result,
            "passed_checks": passed,
            "total_checks": len(checks),
            "status": "validated" if checks and passed == len(checks) else "needs_review",
        }
        self.blackboard.set("validation_result", output)
        return self._record(
            "validate outcome",
            f"Validation status: {output['status']} ({passed}/{len(checks)} checks passed).",
            {"incident_id": incident["incident_id"], "primary_ne": triage["primary_ne"]},
            output,
            start,
        )

