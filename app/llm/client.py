from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


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


class DeepSeekLLMClient(BaseLLMClient):
    """OpenAI-compatible DeepSeek chat-completions client with JSON-mode parsing."""

    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking: str | None = None,
        client: Any | None = None,
        max_json_retries: int = 2,
        max_tokens: int | None = None,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.reasoning_effort = reasoning_effort or os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.thinking = self._normalize_thinking(thinking or os.getenv("DEEPSEEK_THINKING", "disabled"))
        self.max_json_retries = max_json_retries
        self.max_tokens = max_tokens or int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))

        if client is not None:
            self.client = client
            return
        if not self.api_key:
            raise MissingAPIKeyError(
                "DEEPSEEK_API_KEY is required for --mode llm with --provider deepseek. "
                "Set it in the environment or run with --mode rule."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: python3 -m pip install -r requirements.txt") from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
        usage_totals: dict[str, Any] = {}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ]

        if tools and tool_executor:
            messages = self._run_tool_loop(
                messages=messages,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_calls=max_tool_calls,
                records=tool_records,
                usage_totals=usage_totals,
            )

        parsed, response = self._final_structured_response(
            messages=messages,
            tool_records=tool_records,
            system_prompt=system_prompt,
            response_model=response_model,
        )
        self._merge_usage(usage_totals, self._usage(response))
        call = LLMCall(
            agent=agent,
            model=self.model,
            prompt_version="deepseek-chat-v1",
            input_summary=self._input_summary(user_payload),
            parsed_output=parsed.model_dump(),
            token_usage=usage_totals,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tool_calls=tool_records,
        )
        return parsed, call

    @staticmethod
    def to_deepseek_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            if "function" in tool:
                function = dict(tool["function"])
            else:
                function = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            function.pop("strict", None)
            converted.append({"type": "function", "function": function})
        return converted

    def _run_tool_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
        max_tool_calls: int,
        records: list[dict[str, Any]],
        usage_totals: dict[str, Any],
    ) -> list[dict[str, Any]]:
        calls_seen = 0
        chat_tools = self.to_deepseek_tools(tools)
        for _ in range(max_tool_calls):
            response = self._chat_create(messages=messages, tools=chat_tools)
            self._merge_usage(usage_totals, self._usage(response))
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                content = getattr(message, "content", None)
                if content:
                    messages.append({"role": "assistant", "content": content})
                break

            remaining = max_tool_calls - calls_seen
            selected_calls = tool_calls[:remaining]
            messages.append(self._assistant_tool_message(message, selected_calls))
            for tool_call in selected_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = tool_executor(name, args)
                records.append({"name": name, "arguments": args, "output": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
                calls_seen += 1
                if calls_seen >= max_tool_calls:
                    return messages
        return messages

    def _final_structured_response(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_records: list[dict[str, Any]],
        system_prompt: str,
        response_model: type[T],
    ) -> tuple[T, Any]:
        last_error = ""
        for _ in range(self.max_json_retries + 1):
            final_messages = [
                {"role": "system", "content": self._structured_prompt(system_prompt, response_model, last_error)},
                *self._clean_final_messages(messages, tool_records),
            ]
            response = self._chat_create(
                messages=final_messages,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                last_error = "The model returned empty content."
                continue
            try:
                parsed_json = self._load_json_content(content)
                return response_model.model_validate(parsed_json), response
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
        raise ValueError(f"DeepSeek structured output failed validation: {last_error}") from None

    def _clean_final_messages(
        self,
        messages: list[dict[str, Any]],
        tool_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        final_messages = []
        for message in messages[1:]:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                final_messages.append({"role": role, "content": content})
            elif role == "assistant" and content and not message.get("tool_calls"):
                final_messages.append({"role": role, "content": content})
        if tool_records:
            final_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool execution results JSON. Use these outputs as runtime evidence:\n"
                        f"{json.dumps(tool_records, ensure_ascii=False, default=str)}"
                    ),
                }
            )
        return final_messages

    def _chat_create(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.thinking:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
            if self.thinking == "enabled" and self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        return self.client.chat.completions.create(**kwargs)

    def _structured_prompt(
        self,
        system_prompt: str,
        response_model: type[BaseModel],
        last_error: str = "",
    ) -> str:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        correction = ""
        if last_error:
            correction = f"\nPrevious JSON output failed validation. Fix this error: {last_error[:1200]}\n"
        return f"""{system_prompt}

Return only valid JSON. Do not include markdown, code fences, explanations, or extra text.
The JSON must validate against this schema:
{schema}
{correction}"""

    def _assistant_tool_message(self, message: Any, tool_calls: list[Any]) -> dict[str, Any]:
        output: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": getattr(tool_call, "type", "function"),
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments or "{}",
                    },
                }
                for tool_call in tool_calls
            ],
        }
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            output["reasoning_content"] = reasoning_content
        return output

    def _load_json_content(self, content: str) -> Any:
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return json.loads(content)

    def _usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return dict(usage)

    def _merge_usage(self, totals: dict[str, Any], usage: dict[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value

    def _normalize_thinking(self, thinking: str) -> str:
        value = thinking.strip().lower()
        if value in {"1", "true", "on", "yes", "enabled"}:
            return "enabled"
        if value in {"0", "false", "off", "no", "none", "disabled"}:
            return "disabled"
        return value

    def _input_summary(self, payload: dict[str, Any]) -> str:
        incident_id = payload.get("incident", {}).get("incident_id") or payload.get("incident_id")
        keys = ",".join(sorted(payload.keys()))
        return f"incident={incident_id or 'n/a'} keys={keys}"


def create_llm_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> BaseLLMClient:
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER", "deepseek")).lower()
    if selected == "deepseek":
        return DeepSeekLLMClient(model=model, reasoning_effort=reasoning_effort)
    if selected == "openai":
        return OpenAILLMClient(model=model, reasoning_effort=reasoning_effort)
    raise ValueError(f"Unknown LLM provider: {selected}. Expected 'deepseek' or 'openai'.")


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
