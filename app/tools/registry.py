from __future__ import annotations

from typing import Any

from app.tools.telecom_tools import TelecomToolbox


def telecom_tool_definitions(names: list[str] | None = None) -> list[dict[str, Any]]:
    definitions = {
        "get_alarms": {
            "type": "function",
            "name": "get_alarms",
            "description": "Fetch active alarms for a network element in an incident time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ne_id": {"type": "string"},
                    "time_window": {"type": ["string", "null"]},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["ne_id", "time_window", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "get_kpi": {
            "type": "function",
            "name": "get_kpi",
            "description": "Fetch one KPI time series for a network element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "ne_id": {"type": "string"},
                    "time_window": {"type": ["string", "null"]},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["metric", "ne_id", "time_window", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "get_logs": {
            "type": "function",
            "name": "get_logs",
            "description": "Fetch logs for a network element, optionally filtered by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ne_id": {"type": "string"},
                    "keyword": {"type": ["string", "null"]},
                    "time_window": {"type": ["string", "null"]},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["ne_id", "keyword", "time_window", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "get_topology": {
            "type": "function",
            "name": "get_topology",
            "description": "Fetch topology graph around a network element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ne_id": {"type": "string"},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["ne_id", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "retrieve_sop": {
            "type": "function",
            "name": "retrieve_sop",
            "description": "Search SOP/runbook knowledge base using symptom, domain, alarms, and hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "string"},
                    "domain": {"type": ["string", "null"], "enum": ["RAN", "Core", "Transport", None]},
                    "incident_id": {"type": ["string", "null"]},
                    "alarms": {"type": "array", "items": {"type": "string"}},
                    "hypothesis": {"type": ["string", "null"]},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symptom", "domain", "incident_id", "alarms", "hypothesis", "keywords"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "get_ticket_history": {
            "type": "function",
            "name": "get_ticket_history",
            "description": "Fetch similar historical tickets for a network element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ne_id": {"type": "string"},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["ne_id", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "run_diagnostic": {
            "type": "function",
            "name": "run_diagnostic",
            "description": "Run a simulated diagnostic command and return command output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["command", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "validate_after_fix": {
            "type": "function",
            "name": "validate_after_fix",
            "description": "Fetch simulated post-fix validation checks for a network element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ne_id": {"type": "string"},
                    "incident_id": {"type": ["string", "null"]},
                },
                "required": ["ne_id", "incident_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    selected = names or list(definitions)
    return [definitions[name] for name in selected]


class TelecomToolExecutor:
    def __init__(self, toolbox: TelecomToolbox) -> None:
        self.toolbox = toolbox

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if not hasattr(self.toolbox, name):
            raise KeyError(f"Unknown telecom tool: {name}")
        tool = getattr(self.toolbox, name)
        return tool(**arguments)

