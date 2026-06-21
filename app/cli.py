from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore
from app.llm.client import MissingAPIKeyError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent telecom RCA prototype.")
    parser.add_argument("--incident-id", default="INC-RAN-001", help="Incident id to analyze.")
    parser.add_argument("--data-dir", default="data", help="Directory containing incidents.json and sops.json.")
    parser.add_argument("--output", help="Optional path to write the full JSON report.")
    parser.add_argument("--list", action="store_true", help="List available incidents and exit.")
    parser.add_argument("--mode", choices=["llm", "rule"], default="llm", help="Reasoning mode.")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "openai"],
        default=os.getenv("LLM_PROVIDER", "deepseek"),
        help="LLM provider for --mode llm.",
    )
    parser.add_argument("--model", help="Provider model override for --mode llm.")
    parser.add_argument("--reasoning-effort", default=None, help="Provider reasoning effort override.")
    parser.add_argument("--max-tool-calls", type=int, default=8, help="Maximum tool calls per LLM-backed agent.")
    args = parser.parse_args()

    store = DataStore(args.data_dir)
    if args.list:
        for incident in store.list_incidents():
            print(f"{incident['incident_id']} [{incident['domain']}] {incident['symptom']}")
        return

    try:
        report = OrchestratorAgent(
            store,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tool_calls=args.max_tool_calls,
        ).run(args.incident_id)
    except MissingAPIKeyError as exc:
        raise SystemExit(str(exc)) from exc
    payload = report.to_dict()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(_compact_report(payload), indent=2))


def _compact_report(payload: dict) -> dict:
    return {
        "incident_id": payload["incident_id"],
        "domain": payload["domain"],
        "severity": payload["severity"],
        "root_cause": payload["root_cause"],
        "confidence": payload["confidence"],
        "evidence": payload["evidence"],
        "recommended_actions": payload["recommended_actions"],
        "validation_status": payload["validation_result"]["status"],
        "llm_calls": len(payload.get("llm_calls", [])),
        "token_usage": payload.get("token_usage", {}),
        "metrics": payload["metrics"],
    }


if __name__ == "__main__":
    main()
