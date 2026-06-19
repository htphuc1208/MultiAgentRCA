from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.prompts import VALIDATION_PROMPT
from app.llm.schemas import ValidationSummaryOutput


class ValidationAgent(BaseAgent):
    name = "Validation Agent"

    def run(self) -> dict[str, Any]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        output = self._validate()
        incident = self.blackboard.get("incident")
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=VALIDATION_PROMPT,
            user_payload={
                "incident_id": incident["incident_id"],
                "validation_result": output,
                "remediation_plan": self.blackboard.get("remediation_plan"),
            },
            response_model=ValidationSummaryOutput,
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        summary = parsed.model_dump()
        summary["status"] = output["status"]
        summary["passed_checks"] = output["passed_checks"]
        summary["total_checks"] = output["total_checks"]
        output["llm_summary"] = summary
        self.blackboard.set("validation_result", output)
        return self._record(
            "llm summarize validation",
            f"Validation status: {output['status']} ({output['passed_checks']}/{output['total_checks']} checks passed).",
            {"incident_id": incident["incident_id"], "primary_ne": self.blackboard.get("triage")["primary_ne"]},
            output,
            start,
        )

    def _run_rule(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        output = self._validate()
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        return self._record(
            "validate outcome",
            f"Validation status: {output['status']} ({output['passed_checks']}/{output['total_checks']} checks passed).",
            {"incident_id": incident["incident_id"], "primary_ne": triage["primary_ne"]},
            output,
            start,
        )

    def _validate(self) -> dict[str, Any]:
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
        return output
