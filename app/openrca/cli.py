from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from app.llm.client import MissingAPIKeyError
from app.openrca.dataset import OpenRCADataset, OpenRCADatasetError
from app.openrca.evaluator import build_eval_result, summarize_results
from app.openrca.runner import OpenRCARunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenRCA Telecom benchmark integration.")
    parser.add_argument("--dataset", default="Telecom", help="OpenRCA dataset name under OPENRCA_DATA_DIR.")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("OPENRCA_DATA_DIR"),
        help="Directory containing Telecom/query.csv and Telecom/telemetry. Defaults to OPENRCA_DATA_DIR.",
    )
    parser.add_argument("--row-id", type=int, help="Run one row from query.csv.")
    parser.add_argument("--start-row", type=int, default=0, help="First row for batch mode.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum rows for batch mode.")
    parser.add_argument("--output-dir", default="reports/openrca", help="Directory for prediction/evaluation CSVs.")
    parser.add_argument("--mode", choices=["llm"], default="llm", help="OpenRCA v1 supports LLM + tools mode.")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "openai"],
        default=os.getenv("LLM_PROVIDER", "deepseek"),
        help="LLM provider.",
    )
    parser.add_argument("--model", help="Provider model override.")
    parser.add_argument("--reasoning-effort", help="Provider reasoning effort override.")
    parser.add_argument("--max-tool-calls", type=int, default=10)
    args = parser.parse_args()

    try:
        dataset = OpenRCADataset(args.data_dir, dataset=args.dataset)
        runner = OpenRCARunner(
            dataset,
            provider=args.provider,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tool_calls=args.max_tool_calls,
        )
        results = [runner.run_row(row_id) for row_id in _row_ids(args, len(dataset.query_df))]
    except (OpenRCADatasetError, MissingAPIKeyError) as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.dataset.lower().replace("/", "_")
    prediction_path = output_dir / f"{prefix}_predictions.csv"
    evaluation_path = output_dir / f"{prefix}_evaluation.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"

    prediction_rows = [_prediction_row(result) for result in results]
    eval_results = [
        build_eval_result(
            row_id=result["row_id"],
            task_index=result["task_index"],
            instruction=result["instruction"],
            prediction=result["prediction"],
            scoring_points=dataset.get_scoring_points(result["row_id"]),
        )
        for result in results
    ]
    summary_rows = summarize_results(eval_results)

    _write_csv(prediction_path, prediction_rows)
    _write_csv(evaluation_path, [result.to_dict() for result in eval_results])
    _write_csv(summary_path, summary_rows)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "rows": len(results),
                "predictions": str(prediction_path),
                "evaluation": str(evaluation_path),
                "summary": str(summary_path),
                "scores": summary_rows[-1] if summary_rows else {},
            },
            indent=2,
        )
    )


def _row_ids(args: argparse.Namespace, query_len: int) -> list[int]:
    if args.row_id is not None:
        return [args.row_id]
    start = max(0, args.start_row)
    stop = min(query_len, start + max(1, args.limit))
    return list(range(start, stop))


def _prediction_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": result["row_id"],
        "task_index": result["task_index"],
        "instruction": result["instruction"],
        "prediction": result["prediction"],
        "latency_ms": result.get("latency_ms", 0),
        "token_total": result.get("token_usage", {}).get("total_tokens", 0),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
