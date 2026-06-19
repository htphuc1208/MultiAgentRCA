from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.prompts import PLANNER_PROMPT
from app.llm.schemas import RemediationPlanOutput


class RemediationPlannerAgent(BaseAgent):
    name = "Remediation Planner Agent"

    def run(self) -> dict[str, Any]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        selected = self.blackboard.get("selected_hypothesis")
        sop_context = self.blackboard.get("sop_context")
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=PLANNER_PROMPT,
            user_payload={
                "incident": self._incident_stub(incident),
                "selected_hypothesis": selected.to_dict(),
                "sop_context": sop_context,
                "data_evidence": self.blackboard.get("data_evidence"),
            },
            response_model=RemediationPlanOutput,
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        output = {
            "root_cause": selected.cause,
            "confidence": selected.scores["consensus_score"],
            "recommended_actions": self._ensure_human_gate(parsed.recommended_actions),
            "validation_plan": parsed.validation_plan or sop_context.get("validation_rules", []),
            "human_approval_required": True,
            "rollback_plan": parsed.rollback_plan,
            "risk_notes": parsed.risk_notes,
        }
        self.blackboard.set("remediation_plan", output)
        return self._record(
            "llm plan remediation",
            f"LLM prepared {len(output['recommended_actions'])} ordered remediation steps.",
            {"incident_id": incident["incident_id"], "root_cause": selected.cause},
            output,
            start,
        )

    def _run_rule(self) -> dict[str, Any]:
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

    def _incident_stub(self, incident: dict[str, Any]) -> dict[str, Any]:
        return {
            "incident_id": incident["incident_id"],
            "domain": incident.get("domain"),
            "symptom": incident.get("symptom"),
            "primary_ne": incident.get("primary_ne"),
            "service_impact": incident.get("service_impact"),
        }

    def _ensure_human_gate(self, actions: list[str]) -> list[str]:
        gate = "Request human approval before network-impacting action"
        if not actions:
            return [gate]
        if gate.lower() in actions[0].lower():
            return actions
        return [gate, *actions]

    def _plan_actions(self, root_cause: str, sop: dict[str, Any]) -> list[str]:
        steps = list(sop.get("steps", []))
        if not steps:
            steps = [f"Investigate root cause: {root_cause}"]
        return ["Request human approval before network-impacting action", *steps]
