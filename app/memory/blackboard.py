from __future__ import annotations

from typing import Any

from app.models import AgentStep, ToolCall


class Blackboard:
    """Shared state and audit trace used by all agents."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.trace: list[AgentStep] = []
        self.tool_calls: list[ToolCall] = []

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def append(self, key: str, value: Any) -> None:
        self.data.setdefault(key, []).append(value)

    def record_tool(self, name: str, inputs: dict[str, Any], output: Any) -> ToolCall:
        call = ToolCall(name=name, inputs=inputs, output=output)
        self.tool_calls.append(call)
        return call

    def record_agent(
        self,
        agent: str,
        action: str,
        summary: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        tool_calls: list[ToolCall] | None = None,
    ) -> AgentStep:
        step = AgentStep(
            agent=agent,
            action=action,
            summary=summary,
            inputs=inputs,
            outputs=outputs,
            tool_calls=tool_calls or [],
        )
        self.trace.append(step)
        return step

    def snapshot(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "trace": [step.to_dict() for step in self.trace],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }

