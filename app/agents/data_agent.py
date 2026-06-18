from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class DataRetrievalAgent(BaseAgent):
    name = "Data Retrieval Agent"

    def run(self) -> dict[str, Any]:
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

