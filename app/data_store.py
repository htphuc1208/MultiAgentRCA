from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataStore:
    """Loads synthetic telecom incidents and SOPs from JSON files."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self._incidents = self._load_json("incidents.json")
        self._sops = self._load_json("sops.json")

    def _load_json(self, filename: str) -> Any:
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_incidents(self) -> list[dict[str, Any]]:
        return list(self._incidents)

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        for incident in self._incidents:
            if incident["incident_id"] == incident_id:
                return dict(incident)
        raise KeyError(f"Unknown incident_id: {incident_id}")

    def list_sops(self) -> list[dict[str, Any]]:
        return list(self._sops)

    def get_sop(self, sop_id: str) -> dict[str, Any]:
        for sop in self._sops:
            if sop["sop_id"] == sop_id:
                return dict(sop)
        raise KeyError(f"Unknown sop_id: {sop_id}")

