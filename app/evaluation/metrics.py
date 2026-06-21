from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.orchestrator import OrchestratorAgent
from app.baselines.single_agent import SingleAgentRunner
from app.data_store import DataStore
from app.llm.client import BaseLLMClient


@dataclass
class EvaluationResult:
    configuration: str
    incident_id: str
    predicted_root_cause: str
    ground_truth: str
    rca_accuracy: float
    top3_accuracy: float
    remediation_correctness: float
    tool_use_validity: float
    evidence_coverage: float
    hallucination_rate: float
    stability: float
    latency_ms: int
    token_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "incident_id": self.incident_id,
            "predicted_root_cause": self.predicted_root_cause,
            "ground_truth": self.ground_truth,
            "rca_accuracy": self.rca_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "remediation_correctness": self.remediation_correctness,
            "tool_use_validity": self.tool_use_validity,
            "evidence_coverage": self.evidence_coverage,
            "hallucination_rate": self.hallucination_rate,
            "stability": self.stability,
            "latency_ms": self.latency_ms,
            "token_total": self.token_total,
        }


def evaluate_proposed(
    store: DataStore,
    *,
    mode: str = "rule",
    llm_client: BaseLLMClient | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    reasoning_effort: str | None = None,
    repeats: int = 1,
) -> list[EvaluationResult]:
    return _evaluate_configuration(
        store,
        configuration="Proposed Multi-Agent + SOP + Consensus",
        mode=mode,
        use_consensus=True,
        llm_client=llm_client,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        repeats=repeats,
    )


def evaluate_baselines(
    store: DataStore,
    *,
    mode: str = "rule",
    llm_client: BaseLLMClient | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    reasoning_effort: str | None = None,
    repeats: int = 1,
) -> list[EvaluationResult]:
    results = []
    results.extend(
        _evaluate_configuration(
            store,
            configuration="Baseline 1 Rule/SOP lookup only",
            mode="rule",
            use_consensus=True,
            repeats=1,
        )
    )
    results.extend(
        _evaluate_single_agent(
            store,
            configuration="Baseline 2 Single ReAct-style agent",
            mode=mode,
            llm_client=llm_client,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            repeats=repeats,
        )
    )
    results.extend(
        _evaluate_configuration(
            store,
            configuration="Baseline 3 Multi-Agent without consensus",
            mode=mode,
            use_consensus=False,
            llm_client=llm_client,
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            repeats=repeats,
        )
    )
    return results


def _evaluate_single_agent(
    store: DataStore,
    *,
    configuration: str,
    mode: str,
    llm_client: BaseLLMClient | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    reasoning_effort: str | None = None,
    repeats: int = 1,
) -> list[EvaluationResult]:
    results = []
    for incident in store.list_incidents():
        run_payloads = []
        for _ in range(repeats):
            report = SingleAgentRunner(
                store,
                mode=mode,
                llm_client=llm_client,
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
            ).run(incident["incident_id"])
            run_payloads.append(report.to_dict())
        results.append(_result_from_payload(configuration, incident["incident_id"], store, run_payloads))
    return results


def summarize(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.configuration, []).append(result)
    rows = []
    for configuration, items in grouped.items():
        rows.append(
            {
                "configuration": configuration,
                "incidents": len(items),
                "rca_accuracy": _avg(item.rca_accuracy for item in items),
                "top3_accuracy": _avg(item.top3_accuracy for item in items),
                "remediation_correctness": _avg(item.remediation_correctness for item in items),
                "tool_use_validity": _avg(item.tool_use_validity for item in items),
                "evidence_coverage": _avg(item.evidence_coverage for item in items),
                "hallucination_rate": _avg(item.hallucination_rate for item in items),
                "stability": _avg(item.stability for item in items),
                "latency_ms": round(_avg(item.latency_ms for item in items)),
                "token_total": round(_avg(item.token_total for item in items)),
            }
        )
    return rows


def _evaluate_configuration(
    store: DataStore,
    *,
    configuration: str,
    mode: str,
    use_consensus: bool,
    llm_client: BaseLLMClient | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    reasoning_effort: str | None = None,
    repeats: int = 1,
) -> list[EvaluationResult]:
    results = []
    for incident in store.list_incidents():
        run_payloads = []
        for _ in range(repeats):
            report = OrchestratorAgent(
                store,
                mode=mode,
                llm_client=llm_client,
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                use_consensus=use_consensus,
            ).run(incident["incident_id"])
            run_payloads.append(report.to_dict())
        results.append(_result_from_payload(configuration, incident["incident_id"], store, run_payloads))
    return results


def _result_from_payload(
    configuration: str,
    incident_id: str,
    store: DataStore,
    payloads: list[dict[str, Any]],
) -> EvaluationResult:
    payload = payloads[0]
    label = store.get_eval_label(incident_id)
    top3 = [hypothesis["cause"].lower() for hypothesis in payload["hypotheses"][:3]]
    predicted = payload["root_cause"]
    predictions = [item["root_cause"] for item in payloads]
    return EvaluationResult(
        configuration=configuration,
        incident_id=incident_id,
        predicted_root_cause=predicted,
        ground_truth=label["ground_truth"],
        rca_accuracy=float(predicted.lower() == label["ground_truth"].lower()),
        top3_accuracy=float(label["ground_truth"].lower() in top3),
        remediation_correctness=_remediation_correctness(payload["recommended_actions"], label["expected_actions"]),
        tool_use_validity=_tool_use_validity(payload["metrics"]["unique_tools"]),
        evidence_coverage=_evidence_coverage(payload),
        hallucination_rate=_hallucination_rate(payload),
        stability=_stability(predictions),
        latency_ms=payload.get("latency_ms", 0),
        token_total=int(payload.get("token_usage", {}).get("total_tokens", 0) or 0),
    )


def _tool_use_validity(tool_names: list[str]) -> float:
    required = {
        "get_alarms",
        "get_kpi",
        "get_logs",
        "get_topology",
        "retrieve_sop",
        "validate_after_fix",
    }
    return len(required & set(tool_names)) / len(required)


def _evidence_coverage(payload: dict[str, Any]) -> float:
    has_evidence = bool(payload["evidence"])
    has_tools = payload["metrics"]["tool_calls"] >= 6
    has_trace = payload["metrics"]["agent_steps"] >= 8
    has_refs = bool(payload.get("evidence_refs")) or payload["metrics"]["tool_calls"] > 0
    return (float(has_evidence) + float(has_tools) + float(has_trace) + float(has_refs)) / 4


def _hallucination_rate(payload: dict[str, Any]) -> float:
    evidence_count = len(payload["evidence"])
    if evidence_count == 0:
        return 1.0
    supported_prefixes = ("Alarm:", "KPI:", "Log:", "Diagnostic:", "Ticket:")
    unsupported = [item for item in payload["evidence"] if not item.startswith(supported_prefixes)]
    return len(unsupported) / evidence_count


def _remediation_correctness(recommended: list[str], expected: list[str]) -> float:
    if not expected:
        return 0.8 if recommended else 0.0
    recommended_text = " ".join(recommended).lower()
    hits = sum(1 for action in expected if _token_overlap(action.lower(), recommended_text) >= 0.35)
    return hits / len(expected)


def _stability(predictions: list[str]) -> float:
    if not predictions:
        return 0.0
    majority = max(set(predictions), key=predictions.count)
    return predictions.count(majority) / len(predictions)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.replace("/", " ").split() if len(token) > 2}
    right_tokens = {token for token in right.replace("/", " ").split() if len(token) > 2}
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _avg(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 0.0
