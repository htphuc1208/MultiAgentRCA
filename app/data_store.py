from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataStore:
    """Loads synthetic runtime data, hidden eval labels, tickets, and SOPs."""

    HIDDEN_INCIDENT_FIELDS = {"ground_truth", "expected_actions", "sop_id", "ticket_history"}

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self._incidents = self._load_json("incidents.json")
        self._sops = self._load_json("sops.json")
        self._eval_labels = self._load_json("eval_labels.json")
        self._tickets = self._load_json("tickets.json")

    def _load_json(self, filename: str) -> Any:
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_incidents(self) -> list[dict[str, Any]]:
        return [self._sanitize_incident(incident) for incident in self._incidents]

    def list_raw_incidents(self) -> list[dict[str, Any]]:
        return list(self._incidents)

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        for incident in self._incidents:
            if incident["incident_id"] == incident_id:
                return self._sanitize_incident(incident)
        raise KeyError(f"Unknown incident_id: {incident_id}")

    def get_raw_incident(self, incident_id: str) -> dict[str, Any]:
        for incident in self._incidents:
            if incident["incident_id"] == incident_id:
                return dict(incident)
        raise KeyError(f"Unknown incident_id: {incident_id}")

    def get_eval_label(self, incident_id: str) -> dict[str, Any]:
        for label in self._eval_labels:
            if label["incident_id"] == incident_id:
                return dict(label)
        raise KeyError(f"Unknown eval label for incident_id: {incident_id}")

    def list_eval_labels(self) -> list[dict[str, Any]]:
        return list(self._eval_labels)

    def list_tickets(self) -> list[dict[str, Any]]:
        return list(self._tickets)

    def get_tickets(self, ne_id: str | None = None, incident_id: str | None = None) -> list[dict[str, Any]]:
        tickets = []
        for ticket in self._tickets:
            if ne_id and ticket.get("ne_id") != ne_id:
                continue
            if incident_id and ticket.get("related_incident_id") != incident_id:
                continue
            tickets.append(dict(ticket))
        return tickets

    def list_sops(self) -> list[dict[str, Any]]:
        return list(self._sops)

    def get_sop(self, sop_id: str) -> dict[str, Any]:
        for sop in self._sops:
            if sop["sop_id"] == sop_id:
                return dict(sop)
        raise KeyError(f"Unknown sop_id: {sop_id}")

    def _sanitize_incident(self, incident: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in dict(incident).items() if key not in self.HIDDEN_INCIDENT_FIELDS}
