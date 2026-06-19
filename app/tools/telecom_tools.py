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
        alarms: list[str] | None = None,
        hypothesis: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        incident = self._incident(incident_id) if incident_id else {}
        query_parts = [
            symptom,
            hypothesis or "",
            " ".join(alarms or incident.get("alarms", [])),
            " ".join(keywords or []),
        ]
        symptom_text = " ".join(query_parts).lower()
        candidates: list[tuple[float, dict[str, Any]]] = []
        for sop in self.store.list_sops():
            if domain and sop["domain"] != domain:
                continue
            overlap = sum(2 for term in sop.get("symptom_keywords", []) if term in symptom_text)
            cause_overlap = sum(
                1 for cause in sop.get("likely_causes", []) if self._token_overlap(cause.lower(), symptom_text) >= 0.3
            )
            title_overlap = self._token_overlap(sop.get("title", "").lower(), symptom_text)
            candidates.append((overlap + cause_overlap + title_overlap, sop))
        candidates.sort(key=lambda item: item[0], reverse=True)
        output = dict(candidates[0][1]) if candidates else {}
        output["candidates"] = [
            {
                "sop_id": sop["sop_id"],
                "title": sop["title"],
                "score": round(score, 3),
                "likely_causes": sop.get("likely_causes", []),
            }
            for score, sop in candidates[:3]
        ]
        return self._record(
            "retrieve_sop",
            {
                "symptom": symptom,
                "domain": domain,
                "incident_id": incident_id,
                "alarms": alarms,
                "hypothesis": hypothesis,
                "keywords": keywords,
            },
            output,
        )

    def get_ticket_history(self, ne_id: str, incident_id: str | None = None) -> list[dict[str, str]]:
        output = self.store.get_tickets(ne_id=ne_id, incident_id=incident_id)
        return self._record("get_ticket_history", {"ne_id": ne_id, "incident_id": incident_id}, output)

    def run_diagnostic(self, command: str, incident_id: str | None = None) -> str:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("diagnostics", {}).get(command, "No simulated output for command")
        return self._record("run_diagnostic", {"command": command, "incident_id": incident_id}, output)

    def validate_after_fix(self, ne_id: str, incident_id: str | None = None) -> dict[str, Any]:
        incident = self._incident(incident_id) if incident_id else {}
        output = incident.get("validation", {})
        return self._record("validate_after_fix", {"ne_id": ne_id, "incident_id": incident_id}, output)

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = {token for token in left.replace("/", " ").replace("-", " ").split() if len(token) > 2}
        right_tokens = {token for token in right.replace("/", " ").replace("-", " ").split() if len(token) > 2}
        if not left_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens)
