from __future__ import annotations

from app.agents.base import BaseAgent
from app.llm.prompts import SOP_PROMPT
from app.llm.schemas import SOPSelectionOutput
from app.tools.registry import TelecomToolExecutor, telecom_tool_definitions


class SOPAgent(BaseAgent):
    name = "SOP / Knowledge Agent"

    def run(self) -> dict[str, object]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, object]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        data = self.blackboard.get("data_evidence")
        hypotheses = self.blackboard.get("hypotheses", [])
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=SOP_PROMPT,
            user_payload={
                "incident": self._incident_stub(incident),
                "triage": triage,
                "alarms": data.get("alarms", []),
                "evidence_items": data.get("evidence_items", []),
                "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
            },
            response_model=SOPSelectionOutput,
            tools=telecom_tool_definitions(["retrieve_sop"]),
            tool_executor=TelecomToolExecutor(self.tools),
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        sop = self._select_sop_from_tool_output(parsed.selected_sop_id, llm_call.tool_calls)
        output = {
            "sop": sop,
            "sop_id": sop.get("sop_id") or parsed.selected_sop_id,
            "selected_sop": parsed.model_dump(),
            "likely_causes": sop.get("likely_causes", parsed.likely_causes),
            "validation_rules": sop.get("validation_rules", parsed.validation_rules),
        }
        self.blackboard.set("sop_context", output)
        return self._record(
            "llm retrieve operating procedure",
            f"LLM selected SOP {output['sop_id']} for {triage['domain']} incident.",
            {"incident_id": incident["incident_id"], "symptom": incident["symptom"]},
            output,
            start,
        )

    def _run_rule(self) -> dict[str, object]:
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

    def _incident_stub(self, incident: dict[str, object]) -> dict[str, object]:
        return {
            "incident_id": incident["incident_id"],
            "domain": incident.get("domain"),
            "symptom": incident.get("symptom"),
            "description": incident.get("description"),
            "time_window": incident.get("time_window"),
            "primary_ne": incident.get("primary_ne"),
        }

    def _select_sop_from_tool_output(self, selected_sop_id: str, tool_calls: list[dict]) -> dict:
        fallback = {}
        for call in tool_calls:
            if call["name"] != "retrieve_sop":
                continue
            output = call.get("output") or {}
            if not fallback:
                fallback = output
            if output.get("sop_id") == selected_sop_id:
                return output
        if fallback:
            return fallback
        return self.tools.store.get_sop(selected_sop_id)
