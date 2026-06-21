from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from app.data_store import DataStore
from app.evaluation.metrics import evaluate_baselines, evaluate_proposed, summarize
from app.llm.client import MissingAPIKeyError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RCA experiment configurations.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--mode", choices=["llm", "rule"], default="rule")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "openai"],
        default=os.getenv("LLM_PROVIDER", "deepseek"),
    )
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    store = DataStore(args.data_dir)
    repeats = args.repeats
    if args.mode == "llm" and repeats == 1:
        repeats = 3
    try:
        results = [
            *evaluate_baselines(
                store,
                mode=args.mode,
                provider=args.provider,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                repeats=repeats,
            ),
            *evaluate_proposed(
                store,
                mode=args.mode,
                provider=args.provider,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                repeats=repeats,
            ),
        ]
    except MissingAPIKeyError as exc:
        raise SystemExit(str(exc)) from exc
    summary = summarize(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "experiment_results.csv", [result.to_dict() for result in results])
    _write_csv(output_dir / "experiment_summary.csv", summary)

    for row in summary:
        print(
            f"{row['configuration']}: RCA={row['rca_accuracy']:.3f}, "
            f"Top3={row['top3_accuracy']:.3f}, ToolUse={row['tool_use_validity']:.3f}"
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
