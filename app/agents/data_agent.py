from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.prompts import DATA_AGENT_PROMPT
from app.llm.schemas import DataCollectionPlan
from app.tools.registry import TelecomToolExecutor, telecom_tool_definitions


class DataRetrievalAgent(BaseAgent):
    name = "Data Retrieval Agent"

    def run(self) -> dict[str, Any]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=DATA_AGENT_PROMPT,
            user_payload={
                "incident": self._incident_stub(incident),
                "triage": triage,
                "available_kpis": list(incident.get("kpis", {})),
                "available_diagnostics": list(incident.get("diagnostics", {})),
            },
            response_model=DataCollectionPlan,
            tools=telecom_tool_definitions(
                ["get_alarms", "get_kpi", "get_logs", "get_ticket_history", "run_diagnostic"]
            ),
            tool_executor=TelecomToolExecutor(self.tools),
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        output = self._aggregate_llm_tool_outputs(parsed.model_dump(), llm_call.tool_calls)
        self.blackboard.set("data_evidence", output)
        return self._record(
            "llm collect telemetry",
            f"LLM collected telemetry with {len(llm_call.tool_calls)} tool calls.",
            {"incident_id": incident["incident_id"], "primary_ne": triage["primary_ne"]},
            output,
            start,
        )

    def _run_rule(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        incident_id = incident["incident_id"]
        primary_ne = triage["primary_ne"]
        time_window = incident.get("time_window")

        alarms = self.tools.get_alarms(primary_ne, time_window, incident_id)
        kpis = {
            metric: self.tools.get_kpi(metric, primary_ne, time_window, incident_id)
            for metric in incident.get("kpis", {})
        }
        logs = self.tools.get_logs(primary_ne, None, time_window, incident_id)
        ticket_history = self.tools.get_ticket_history(primary_ne, incident_id)
        diagnostics = {
            command: self.tools.run_diagnostic(command, incident_id)
            for command in incident.get("diagnostics", {})
        }

        evidence_items = []
        evidence_items.extend(f"Alarm: {alarm}" for alarm in alarms)
        evidence_items.extend(self._summarize_kpis(kpis))
        evidence_items.extend(f"Log: {line}" for line in logs)
        evidence_items.extend(
            f"Diagnostic: {command} -> {output}" for command, output in diagnostics.items()
        )
        evidence_items.extend(
            f"Ticket: {ticket['summary']}" for ticket in ticket_history if "summary" in ticket
        )

        output = {
            "alarms": alarms,
            "kpis": kpis,
            "logs": logs,
            "ticket_history": ticket_history,
            "diagnostics": diagnostics,
            "evidence_items": evidence_items,
        }
        self.blackboard.set("data_evidence", output)
        return self._record(
            "collect telemetry",
            f"Collected {len(alarms)} alarms, {len(kpis)} KPIs, {len(logs)} logs, and diagnostics.",
            {"incident_id": incident_id, "primary_ne": primary_ne, "time_window": time_window},
            output,
            start,
        )

    def _incident_stub(self, incident: dict[str, Any]) -> dict[str, Any]:
        return {
            "incident_id": incident["incident_id"],
            "domain": incident.get("domain"),
            "severity": incident.get("severity"),
            "symptom": incident.get("symptom"),
            "description": incident.get("description"),
            "time_window": incident.get("time_window"),
            "primary_ne": incident.get("primary_ne"),
            "service_impact": incident.get("service_impact"),
            "affected_services": incident.get("affected_services", []),
        }

    def _aggregate_llm_tool_outputs(
        self,
        parsed: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        output = {
            "alarms": [],
            "kpis": {},
            "logs": [],
            "ticket_history": [],
            "diagnostics": {},
            "evidence_items": list(parsed.get("evidence_items", [])),
            "missing_data": parsed.get("missing_data", []),
            "collection_rationale": parsed.get("rationale", ""),
        }
        for call in tool_calls:
            name = call["name"]
            args = call.get("arguments", {})
            result = call.get("output")
            if name == "get_alarms":
                output["alarms"] = result or []
                output["evidence_items"].extend(f"Alarm: {alarm}" for alarm in result or [])
            elif name == "get_kpi":
                metric = args.get("metric", "unknown")
                output["kpis"][metric] = result or {}
                output["evidence_items"].extend(self._summarize_kpis({metric: result or {}}))
            elif name == "get_logs":
                output["logs"].extend(result or [])
                output["evidence_items"].extend(f"Log: {line}" for line in result or [])
            elif name == "get_ticket_history":
                output["ticket_history"].extend(result or [])
                output["evidence_items"].extend(
                    f"Ticket: {ticket['summary']}" for ticket in result or [] if "summary" in ticket
                )
            elif name == "run_diagnostic":
                command = args.get("command", "unknown")
                output["diagnostics"][command] = result
                output["evidence_items"].append(f"Diagnostic: {command} -> {result}")
        output["evidence_items"] = list(dict.fromkeys(output["evidence_items"]))
        return output

    def _summarize_kpis(self, kpis: dict[str, dict[str, Any]]) -> list[str]:
        summaries = []
        for metric, payload in kpis.items():
            latest = payload.get("latest")
            unit = payload.get("unit", "")
            normal = payload.get("normal_range")
            if normal:
                summaries.append(
                    f"KPI: {metric}={latest}{unit} normal_range={normal[0]}-{normal[1]}{unit}"
                )
            else:
                summaries.append(f"KPI: {metric}={latest}{unit}")
        return summaries
