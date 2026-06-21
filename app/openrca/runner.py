from __future__ import annotations

from typing import Any

from app.llm.client import BaseLLMClient, create_llm_client
from app.openrca.dataset import OpenRCADataset
from app.openrca.formatter import format_prediction
from app.openrca.schemas import OpenRCAInvestigationOutput, OpenRCAPredictionOutput, OpenRCATaskParse
from app.openrca.tools import OpenRCATelemetryTools, OpenRCAToolExecutor, openrca_tool_definitions


TASK_PARSE_PROMPT = """
You are parsing an OpenRCA Telecom benchmark task.
Extract the requested time range, number of failures if stated, and which root-cause fields the task asks for.
All datetimes are UTC+8 and must use '%Y-%m-%d %H:%M:%S'.
Return only the requested schema.
"""


INVESTIGATION_PROMPT = """
You are an OpenRCA Telecom telemetry investigation agent.
Use the available tools to inspect metric anomalies first, then traces, and logs only if useful.
Do not use scoring_points, record.csv, or any ground-truth labels.
Root cause components and reasons must come from the candidate catalog.
Return a bounded evidence summary suitable for final RCA.
"""


PREDICTION_PROMPT = """
You are producing the final OpenRCA Telecom benchmark answer.
Use only the task instruction, parsed task fields, candidate catalog, tool evidence, and investigation summary.
Be decisive. Do not answer unknown/null/not found.
Only include fields requested by the task. Components and reasons must exactly match the candidate catalog strings.
Datetime values must use '%Y-%m-%d %H:%M:%S' in UTC+8.
Return only the requested schema.
"""


class OpenRCARunner:
    def __init__(
        self,
        dataset: OpenRCADataset,
        *,
        llm_client: BaseLLMClient | None = None,
        provider: str = "deepseek",
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tool_calls: int = 10,
    ) -> None:
        self.dataset = dataset
        self.llm_client = llm_client
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls

    def run_row(self, row_id: int) -> dict[str, Any]:
        task = self.dataset.get_runtime_task(row_id)
        client = self._client()
        telemetry_tools = OpenRCATelemetryTools(self.dataset)
        catalog = telemetry_tools.get_candidate_catalog()

        task_parse, parse_call = client.structured(
            agent="OpenRCA Task Parser",
            system_prompt=TASK_PARSE_PROMPT,
            user_payload={
                "task": task,
                "available_telemetry_dates": self.dataset.available_dates(),
            },
            response_model=OpenRCATaskParse,
        )

        investigation, investigation_call = client.structured(
            agent="OpenRCA Telemetry Investigator",
            system_prompt=INVESTIGATION_PROMPT,
            user_payload={
                "task": task,
                "parsed_task": task_parse.model_dump(),
                "candidate_catalog": catalog,
            },
            response_model=OpenRCAInvestigationOutput,
            tools=openrca_tool_definitions(
                [
                    "get_candidate_catalog",
                    "summarize_metric_anomalies",
                    "summarize_trace_latency",
                    "search_logs",
                ]
            ),
            tool_executor=OpenRCAToolExecutor(telemetry_tools),
            max_tool_calls=self.max_tool_calls,
        )

        prediction, prediction_call = client.structured(
            agent="OpenRCA Prediction Formatter",
            system_prompt=PREDICTION_PROMPT,
            user_payload={
                "task": task,
                "parsed_task": task_parse.model_dump(),
                "candidate_catalog": catalog,
                "investigation": investigation.model_dump(),
            },
            response_model=OpenRCAPredictionOutput,
        )
        prediction_text = format_prediction(prediction)
        llm_calls = [parse_call.to_dict(), investigation_call.to_dict(), prediction_call.to_dict()]
        return {
            "row_id": row_id,
            "task_index": task["task_index"],
            "instruction": task["instruction"],
            "task_parse": task_parse.model_dump(),
            "investigation": investigation.model_dump(),
            "prediction": prediction_text,
            "prediction_structured": prediction.model_dump(),
            "llm_calls": llm_calls,
            "token_usage": self._token_usage(llm_calls),
            "latency_ms": sum(call.get("latency_ms", 0) for call in llm_calls),
        }

    def _client(self) -> BaseLLMClient:
        if self.llm_client is not None:
            return self.llm_client
        return create_llm_client(
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )

    def _token_usage(self, llm_calls: list[dict[str, Any]]) -> dict[str, Any]:
        totals: dict[str, Any] = {"total_tokens": 0}
        for call in llm_calls:
            for key, value in call.get("token_usage", {}).items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
        return totals
