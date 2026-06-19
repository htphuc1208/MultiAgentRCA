from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.memory.blackboard import Blackboard
from app.models import ToolCall
from app.tools.telecom_tools import TelecomToolbox
from app.llm.client import BaseLLMClient


class BaseAgent(ABC):
    name = "Base Agent"

    def __init__(
        self,
        blackboard: Blackboard,
        tools: TelecomToolbox,
        *,
        mode: str = "rule",
        llm_client: BaseLLMClient | None = None,
        max_tool_calls: int = 6,
    ) -> None:
        self.blackboard = blackboard
        self.tools = tools
        self.mode = mode
        self.llm_client = llm_client
        self.max_tool_calls = max_tool_calls

    @abstractmethod
    def run(self) -> dict[str, Any]:
        raise NotImplementedError

    def _new_tool_calls(self, start_index: int) -> list[ToolCall]:
        return self.blackboard.tool_calls[start_index:]

    def _record(
        self,
        action: str,
        summary: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        start_tool_index: int,
    ) -> dict[str, Any]:
        self.blackboard.record_agent(
            agent=self.name,
            action=action,
            summary=summary,
            inputs=inputs,
            outputs=outputs,
            tool_calls=self._new_tool_calls(start_tool_index),
        )
        return outputs

    @property
    def use_llm(self) -> bool:
        return self.mode == "llm" and self.llm_client is not None

    def _record_llm(self, call: object) -> None:
        self.blackboard.record_llm(call)
