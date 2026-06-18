from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent telecom RCA prototype.")
    parser.add_argument("--incident-id", default="INC-RAN-001", help="Incident id to analyze.")
    parser.add_argument("--data-dir", default="data", help="Directory containing incidents.json and sops.json.")
    parser.add_argument("--output", help="Optional path to write the full JSON report.")
    parser.add_argument("--list", action="store_true", help="List available incidents and exit.")
    args = parser.parse_args()

    store = DataStore(args.data_dir)
    if args.list:
        for incident in store.list_incidents():
            print(f"{incident['incident_id']} [{incident['domain']}] {incident['symptom']}")
        return

    report = OrchestratorAgent(store).run(args.incident_id)
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
        "metrics": payload["metrics"],
    }


if __name__ == "__main__":
    main()

