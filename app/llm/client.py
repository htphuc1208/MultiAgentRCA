from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass
class LLMCall:
    agent: str
    model: str
    prompt_version: str
    input_summary: str
    parsed_output: dict[str, Any]
    token_usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseLLMClient:
    def structured(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[T],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_calls: int = 6,
    ) -> tuple[T, LLMCall]:
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.reasoning_effort = reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT", "medium")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is required for --mode llm. Set it in the environment or run with --mode rule."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: python3 -m pip install -r requirements.txt") from exc
        self.client = OpenAI(api_key=self.api_key)

    def structured(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[T],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_calls: int = 6,
    ) -> tuple[T, LLMCall]:
        started = time.perf_counter()
        tool_records: list[dict[str, Any]] = []
        input_items: list[Any] = [
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)}
        ]

        if tools and tool_executor:
            input_items = self._run_tool_loop(
                input_items=input_items,
                instructions=system_prompt,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_calls=max_tool_calls,
                records=tool_records,
            )

        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=input_items,
            text_format=response_model,
            reasoning={"effort": self.reasoning_effort},
        )
        parsed = response.output_parsed
        usage = self._usage(response)
        call = LLMCall(
            agent=agent,
            model=self.model,
            prompt_version="v1",
            input_summary=self._input_summary(user_payload),
            parsed_output=parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict(),
            token_usage=usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tool_calls=tool_records,
        )
        return parsed, call

    def _run_tool_loop(
        self,
        *,
        input_items: list[Any],
        instructions: str,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
        max_tool_calls: int,
        records: list[dict[str, Any]],
    ) -> list[Any]:
        calls_seen = 0
        for _ in range(max_tool_calls):
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_items,
                tools=tools,
                reasoning={"effort": self.reasoning_effort},
            )
            output = list(getattr(response, "output", []) or [])
            input_items += output
            function_calls = [item for item in output if getattr(item, "type", None) == "function_call"]
            if not function_calls:
                break
            for item in function_calls:
                args = json.loads(item.arguments or "{}")
                result = tool_executor(item.name, args)
                records.append({"name": item.name, "arguments": args, "output": result})
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
                calls_seen += 1
                if calls_seen >= max_tool_calls:
                    return input_items
        return input_items

    def _usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return dict(usage)

    def _input_summary(self, payload: dict[str, Any]) -> str:
        incident_id = payload.get("incident", {}).get("incident_id") or payload.get("incident_id")
        keys = ",".join(sorted(payload.keys()))
        return f"incident={incident_id or 'n/a'} keys={keys}"


class FakeLLMClient(BaseLLMClient):
    """Deterministic LLM double for unit tests and offline smoke checks."""

    def __init__(self, responses: dict[str, BaseModel]) -> None:
        self.responses = responses
        self.model = "fake-llm"
        self.calls: list[LLMCall] = []

    def structured(
        self,
        *,
        agent: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[T],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_calls: int = 6,
    ) -> tuple[T, LLMCall]:
        if agent not in self.responses:
            raise KeyError(f"No fake LLM response configured for {agent}")
        parsed = self.responses[agent]
        if not isinstance(parsed, response_model):
            parsed = response_model.model_validate(parsed.model_dump())
        tool_records = []
        if tools and tool_executor:
            for tool in tools[:max_tool_calls]:
                name = tool["name"]
                args = self._default_args(name, user_payload)
                output = tool_executor(name, args)
                tool_records.append({"name": name, "arguments": args, "output": output})
        call = LLMCall(
            agent=agent,
            model=self.model,
            prompt_version="fake-v1",
            input_summary=f"fake keys={','.join(sorted(user_payload.keys()))}",
            parsed_output=parsed.model_dump(),
            token_usage={"total_tokens": 0},
            latency_ms=0,
            tool_calls=tool_records,
        )
        self.calls.append(call)
        return parsed, call

    def _default_args(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        incident = payload.get("incident", {})
        triage = payload.get("triage", {})
        ne_id = triage.get("primary_ne") or incident.get("primary_ne", "")
        incident_id = incident.get("incident_id", "")
        if name == "get_kpi":
            metric = next(iter(incident.get("kpis", {"unknown": {}})))
            return {"metric": metric, "ne_id": ne_id, "incident_id": incident_id}
        if name == "get_logs":
            return {"ne_id": ne_id, "keyword": None, "incident_id": incident_id}
        if name == "retrieve_sop":
            return {
                "symptom": incident.get("symptom", ""),
                "domain": triage.get("domain") or incident.get("domain"),
                "incident_id": incident_id,
                "alarms": incident.get("alarms", []),
                "hypothesis": "",
                "keywords": [],
            }
        if name == "run_diagnostic":
            command = next(iter(incident.get("diagnostics", {"unknown": ""})))
            return {"command": command, "incident_id": incident_id}
        return {"ne_id": ne_id, "incident_id": incident_id}

