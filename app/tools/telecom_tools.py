from __future__ import annotations

from typing import Any

from app.data_store import DataStore
from app.memory.blackboard import Blackboard


class TelecomToolbox:
    """Mock OSS/NMS tools backed by synthetic JSON data."""

    def __init__(self, store: DataStore, blackboard: Blackboard | None = None) -> None:
        self.store = store
        self.blackboard = blackboard

    def _record(self, name: str, inputs: dict[str, Any], output: Any) -> Any:
        if self.blackboard is not None:
            self.blackboard.record_tool(name, inputs, output)
        return output

    def _incident(self, incident_id: str) -> dict[str, Any]:
        return self.store.get_incident(incident_id)

    def get_alarms(
        self,
        ne_id: str,
        time_window: str | None = None,
        incident_id: str | None = None,
    ) -> list[str]:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("alarms", [])
        return self._record(
            "get_alarms",
            {"ne_id": ne_id, "time_window": time_window, "incident_id": incident_id},
            output,
        )

    def get_kpi(
        self,
        metric: str,
        ne_id: str,
        time_window: str | None = None,
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("kpis", {}).get(metric, {})
        return self._record(
            "get_kpi",
            {
                "metric": metric,
                "ne_id": ne_id,
                "time_window": time_window,
                "incident_id": incident_id,
            },
            output,
        )

    def get_logs(
        self,
        ne_id: str,
        keyword: str | None = None,
        time_window: str | None = None,
        incident_id: str | None = None,
    ) -> list[str]:
        incident = self._incident(incident_id) if incident_id else {}
        logs = incident.get("logs", [])
        if keyword:
            needle = keyword.lower()
            logs = [line for line in logs if needle in line.lower()]
        return self._record(
            "get_logs",
            {
                "ne_id": ne_id,
                "keyword": keyword,
                "time_window": time_window,
                "incident_id": incident_id,
            },
            logs,
        )

    def get_topology(self, ne_id: str, incident_id: str | None = None) -> dict[str, Any]:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("topology", {})
        return self._record("get_topology", {"ne_id": ne_id, "incident_id": incident_id}, output)

    def retrieve_sop(
        self,
        symptom: str,
        domain: str | None = None,
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        if incident_id:
            incident = self._incident(incident_id)
            output = self.store.get_sop(incident["sop_id"])
            return self._record(
                "retrieve_sop",
                {"symptom": symptom, "domain": domain, "incident_id": incident_id},
                output,
            )

        symptom_text = symptom.lower()
        candidates = []
        for sop in self.store.list_sops():
            if domain and sop["domain"] != domain:
                continue
            overlap = sum(1 for term in sop.get("symptom_keywords", []) if term in symptom_text)
            candidates.append((overlap, sop))
        candidates.sort(key=lambda item: item[0], reverse=True)
        output = candidates[0][1] if candidates else {}
        return self._record(
            "retrieve_sop",
            {"symptom": symptom, "domain": domain, "incident_id": incident_id},
            output,
        )

    def get_ticket_history(self, ne_id: str, incident_id: str | None = None) -> list[dict[str, str]]:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("ticket_history", [])
        return self._record("get_ticket_history", {"ne_id": ne_id, "incident_id": incident_id}, output)

    def run_diagnostic(self, command: str, incident_id: str | None = None) -> str:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("diagnostics", {}).get(command, "No simulated output for command")
        return self._record("run_diagnostic", {"command": command, "incident_id": incident_id}, output)

    def validate_after_fix(self, ne_id: str, incident_id: str | None = None) -> dict[str, Any]:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("validation", {})
        return self._record("validate_after_fix", {"ne_id": ne_id, "incident_id": incident_id}, output)

