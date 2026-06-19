from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ToolCall:
    name: str
    inputs: dict[str, Any]
    output: Any
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStep:
    agent: str
    action: str
    summary: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return data


@dataclass
class Hypothesis:
    cause: str
    domain: str
    confidence: float
    evidence: list[str]
    evidence_refs: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    source_agents: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RCAReport:
    incident_id: str
    domain: str
    severity: str
    root_cause: str
    confidence: float
    evidence: list[str]
    hypotheses: list[Hypothesis]
    recommended_actions: list[str]
    validation_plan: list[str]
    validation_result: dict[str, Any]
    trace: list[AgentStep]
    metrics: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    evidence_refs: list[str] = field(default_factory=list)
    selected_sop: dict[str, Any] = field(default_factory=dict)
    verification_notes: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hypotheses"] = [hypothesis.to_dict() for hypothesis in self.hypotheses]
        data["trace"] = [step.to_dict() for step in self.trace]
        return data
