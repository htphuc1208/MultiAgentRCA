from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from app.openrca.dataset import OpenRCADataset


TELECOM_REASONS = [
    "CPU fault",
    "network delay",
    "network loss",
    "db connection limit",
    "db close",
]

TELECOM_COMPONENTS = [
    *[f"os_{index:03d}" for index in range(1, 23)],
    *[f"docker_{index:03d}" for index in range(1, 9)],
    *[f"db_{index:03d}" for index in range(1, 14)],
]

UTC_PLUS_8 = ZoneInfo("Asia/Shanghai")


class OpenRCATelemetryTools:
    """Bounded tools for reading OpenRCA Telecom telemetry."""

    def __init__(self, dataset: OpenRCADataset, *, chunksize: int = 100_000) -> None:
        self.dataset = dataset
        self.chunksize = chunksize

    def get_candidate_catalog(self) -> dict[str, Any]:
        return {
            "components": TELECOM_COMPONENTS,
            "reasons": TELECOM_REASONS,
            "telemetry_dates": self.dataset.available_dates(),
            "timezone": "UTC+8 / Asia/Shanghai",
        }

    def summarize_metric_anomalies(
        self,
        start_datetime: str | None,
        end_datetime: str | None,
        metric_file: str | None = None,
        components: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        window = self._window(start_datetime, end_datetime)
        if "error" in window:
            return {"error": window["error"], "anomalies": []}

        files = self._telemetry_files("metric", window["start_ms"], window["end_ms"], metric_file)
        if not files:
            return {
                "time_range": window["label"],
                "files_scanned": 0,
                "anomalies": [],
                "telemetry_gaps": ["No metric CSV files matched the requested range."],
            }

        component_filter = set(components or [])
        thresholds = self._metric_thresholds(files, component_filter)
        anomalies: list[dict[str, Any]] = []
        for path in files:
            for chunk in self._read_csv(path):
                metrics = self._normalize_metric_chunk(chunk)
                if metrics.empty:
                    continue
                if component_filter:
                    metrics = metrics[metrics["component"].isin(component_filter)]
                metrics = metrics[
                    (metrics["timestamp_ms"] >= window["start_ms"])
                    & (metrics["timestamp_ms"] <= window["end_ms"])
                ]
                if metrics.empty:
                    continue
                for row in metrics.itertuples(index=False):
                    key = (row.component, row.metric)
                    threshold = thresholds.get(key)
                    if not threshold:
                        continue
                    direction = ""
                    boundary = 0.0
                    severity = 0.0
                    if row.value > threshold["p95"]:
                        direction = "high"
                        boundary = threshold["p95"]
                        severity = self._relative_deviation(row.value, boundary)
                    elif row.value < threshold["p05"]:
                        direction = "low"
                        boundary = threshold["p05"]
                        severity = self._relative_deviation(boundary, row.value)
                    if direction:
                        anomalies.append(
                            {
                                "file": path.name,
                                "timestamp": self._format_ms(row.timestamp_ms),
                                "component": row.component,
                                "metric": row.metric,
                                "value": round(float(row.value), 6),
                                "direction": direction,
                                "threshold": round(float(boundary), 6),
                                "severity": round(float(severity), 6),
                            }
                        )
                        if len(anomalies) > limit * 20:
                            anomalies = sorted(anomalies, key=lambda item: item["severity"], reverse=True)[: limit * 10]

        anomalies = sorted(anomalies, key=lambda item: item["severity"], reverse=True)[:limit]
        return {
            "time_range": window["label"],
            "files_scanned": len(files),
            "anomalies": anomalies,
            "telemetry_gaps": [] if anomalies else ["No metric anomalies found in the requested window."],
        }

    def summarize_trace_latency(
        self,
        start_datetime: str | None,
        end_datetime: str | None,
        limit: int = 20,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        window = self._window(start_datetime, end_datetime)
        if "error" in window:
            return {"error": window["error"], "top_slow_spans": [], "component_latency": []}

        files = self._telemetry_files("trace", window["start_ms"], window["end_ms"], "trace_span.csv")
        slow_spans: list[dict[str, Any]] = []
        stats: dict[str, dict[str, float]] = {}
        for path in files:
            for chunk in self._read_csv(path):
                spans = self._normalize_trace_chunk(chunk)
                if spans.empty:
                    continue
                spans = spans[
                    (spans["timestamp_ms"] >= window["start_ms"])
                    & (spans["timestamp_ms"] <= window["end_ms"])
                ]
                if spans.empty:
                    continue
                for row in spans.itertuples(index=False):
                    component = row.component or "unknown"
                    item = stats.setdefault(component, {"count": 0, "total_ms": 0.0, "failures": 0, "max_ms": 0.0})
                    item["count"] += 1
                    item["total_ms"] += float(row.elapsed_ms)
                    item["max_ms"] = max(item["max_ms"], float(row.elapsed_ms))
                    if not row.success:
                        item["failures"] += 1
                    slow_spans.append(
                        {
                            "file": path.name,
                            "timestamp": self._format_ms(row.timestamp_ms),
                            "trace_id": row.trace_id,
                            "component": component,
                            "service": row.service,
                            "call_type": row.call_type,
                            "elapsed_ms": round(float(row.elapsed_ms), 6),
                            "success": bool(row.success),
                        }
                    )
                    if len(slow_spans) > limit * 20:
                        slow_spans = sorted(slow_spans, key=lambda item: item["elapsed_ms"], reverse=True)[: limit * 10]

        component_latency = [
            {
                "component": component,
                "count": int(item["count"]),
                "avg_elapsed_ms": round(item["total_ms"] / item["count"], 6) if item["count"] else 0.0,
                "max_elapsed_ms": round(item["max_ms"], 6),
                "failures": int(item["failures"]),
            }
            for component, item in stats.items()
        ]
        component_latency.sort(key=lambda item: (item["failures"], item["avg_elapsed_ms"]), reverse=True)
        slow_spans = sorted(slow_spans, key=lambda item: item["elapsed_ms"], reverse=True)[:limit]
        return {
            "time_range": window["label"],
            "files_scanned": len(files),
            "top_slow_spans": slow_spans,
            "component_latency": component_latency[:limit],
            "telemetry_gaps": [] if files else ["No trace_span.csv files matched the requested range."],
        }

    def search_logs(
        self,
        start_datetime: str | None,
        end_datetime: str | None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        limit = self._limit(limit)
        window = self._window(start_datetime, end_datetime)
        if "error" in window:
            return {"error": window["error"], "matches": []}

        date_dirs = self._date_dirs_for_window(window["start_ms"], window["end_ms"])
        matches: list[dict[str, str]] = []
        needle = (keyword or "").lower().strip()
        files_scanned = 0
        for date_dir in date_dirs:
            log_dir = date_dir / "log"
            if not log_dir.exists():
                continue
            for path in sorted(file for file in log_dir.rglob("*") if file.is_file()):
                files_scanned += 1
                if path.suffix.lower() == ".csv":
                    matches.extend(self._search_csv_log(path, needle, limit - len(matches)))
                else:
                    matches.extend(self._search_text_log(path, needle, limit - len(matches)))
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break
        return {
            "time_range": window["label"],
            "files_scanned": files_scanned,
            "keyword": keyword,
            "matches": matches[:limit],
            "telemetry_gaps": [] if files_scanned else ["No log directory found for the requested telemetry dates."],
        }

    def _metric_thresholds(
        self,
        files: list[Path],
        component_filter: set[str],
        sample_cap_per_series: int = 5000,
    ) -> dict[tuple[str, str], dict[str, float]]:
        samples: dict[tuple[str, str], list[float]] = {}
        for path in files:
            for chunk in self._read_csv(path):
                metrics = self._normalize_metric_chunk(chunk)
                if metrics.empty:
                    continue
                if component_filter:
                    metrics = metrics[metrics["component"].isin(component_filter)]
                for key, series in metrics.groupby(["component", "metric"])["value"]:
                    bucket = samples.setdefault(key, [])
                    room = sample_cap_per_series - len(bucket)
                    if room <= 0:
                        continue
                    bucket.extend(float(value) for value in series.head(room))
        thresholds = {}
        for key, values in samples.items():
            if not values:
                continue
            value_series = pd.Series(values)
            thresholds[key] = {
                "p95": float(value_series.quantile(0.95)),
                "p05": float(value_series.quantile(0.05)),
            }
        return thresholds

    def _normalize_metric_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        columns = set(chunk.columns)
        if {"cmdb_id", "name", "timestamp", "value"}.issubset(columns):
            output = pd.DataFrame(
                {
                    "component": chunk["cmdb_id"].astype(str),
                    "metric": chunk["name"].astype(str),
                    "timestamp_ms": pd.to_numeric(chunk["timestamp"], errors="coerce"),
                    "value": pd.to_numeric(chunk["value"], errors="coerce"),
                }
            )
            return output.dropna(subset=["timestamp_ms", "value"])

        if {"serviceName", "startTime"}.issubset(columns):
            metric_columns = [
                column
                for column in ["avg_time", "num", "succee_num", "succee_rate"]
                if column in chunk.columns
            ]
            frames = []
            for metric in metric_columns:
                frames.append(
                    pd.DataFrame(
                        {
                            "component": chunk["serviceName"].astype(str),
                            "metric": metric,
                            "timestamp_ms": pd.to_numeric(chunk["startTime"], errors="coerce"),
                            "value": pd.to_numeric(chunk[metric], errors="coerce"),
                        }
                    )
                )
            if not frames:
                return pd.DataFrame(columns=["component", "metric", "timestamp_ms", "value"])
            return pd.concat(frames, ignore_index=True).dropna(subset=["timestamp_ms", "value"])

        return pd.DataFrame(columns=["component", "metric", "timestamp_ms", "value"])

    def _normalize_trace_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        if "startTime" not in chunk.columns or "elapsedTime" not in chunk.columns:
            return pd.DataFrame(
                columns=["timestamp_ms", "elapsed_ms", "success", "trace_id", "component", "service", "call_type"]
            )
        success = self._column(chunk, "success", True)
        output = pd.DataFrame(
            {
                "timestamp_ms": pd.to_numeric(chunk["startTime"], errors="coerce"),
                "elapsed_ms": pd.to_numeric(chunk["elapsedTime"], errors="coerce"),
                "success": success.astype(str).str.lower().isin(["true", "1", "yes"]),
                "trace_id": self._column(chunk, "traceId", "").astype(str),
                "component": self._column(chunk, "cmdb_id", "").astype(str),
                "service": self._column(chunk, "serviceName", "").astype(str),
                "call_type": self._column(chunk, "callType", "").astype(str),
            }
        )
        return output.dropna(subset=["timestamp_ms", "elapsed_ms"])

    def _telemetry_files(
        self,
        kind: str,
        start_ms: int,
        end_ms: int,
        filename: str | None = None,
    ) -> list[Path]:
        files = []
        for date_dir in self._date_dirs_for_window(start_ms, end_ms):
            base = date_dir / kind
            if not base.exists():
                continue
            if filename:
                candidate = base / filename
                if candidate.exists():
                    files.append(candidate)
            else:
                files.extend(sorted(base.glob("*.csv")))
        return files

    def _date_dirs_for_window(self, start_ms: int, end_ms: int) -> list[Path]:
        start = datetime.fromtimestamp(start_ms / 1000, UTC_PLUS_8).date()
        end = datetime.fromtimestamp(end_ms / 1000, UTC_PLUS_8).date()
        names = set()
        current = start
        while current <= end:
            names.add(current.strftime("%Y_%m_%d"))
            current += timedelta(days=1)
        return [path for path in self.dataset.telemetry_date_dirs() if path.name in names]

    def _read_csv(self, path: Path) -> Iterator[pd.DataFrame]:
        try:
            reader = pd.read_csv(path, chunksize=self.chunksize)
        except pd.errors.EmptyDataError:
            return
        for chunk in reader:
            yield chunk

    def _column(self, chunk: pd.DataFrame, column: str, default: Any) -> pd.Series:
        if column in chunk.columns:
            return chunk[column]
        return pd.Series([default] * len(chunk), index=chunk.index)

    def _search_csv_log(self, path: Path, needle: str, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        matches = []
        for chunk in self._read_csv(path):
            for index, row in chunk.astype(str).iterrows():
                text = " ".join(row.to_list())
                if needle and needle not in text.lower():
                    continue
                matches.append({"file": path.name, "row": str(index), "text": text[:500]})
                if len(matches) >= limit:
                    return matches
        return matches

    def _search_text_log(self, path: Path, needle: str, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        matches = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    text = line.strip()
                    if needle and needle not in text.lower():
                        continue
                    matches.append({"file": path.name, "line": str(line_number), "text": text[:500]})
                    if len(matches) >= limit:
                        return matches
        except OSError as exc:
            return [{"file": path.name, "error": str(exc)}]
        return matches

    def _window(self, start_datetime: str | None, end_datetime: str | None) -> dict[str, Any]:
        if not start_datetime or not end_datetime:
            return {"error": "Both start_datetime and end_datetime are required in '%Y-%m-%d %H:%M:%S' UTC+8 format."}
        try:
            start = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC_PLUS_8)
            end = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC_PLUS_8)
        except ValueError:
            return {"error": "Invalid datetime format. Expected '%Y-%m-%d %H:%M:%S' in UTC+8."}
        if end < start:
            return {"error": "end_datetime must be greater than or equal to start_datetime."}
        return {
            "start_ms": int(start.timestamp() * 1000),
            "end_ms": int(end.timestamp() * 1000),
            "label": f"{start_datetime} to {end_datetime} UTC+8",
        }

    def _format_ms(self, timestamp_ms: int | float) -> str:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000, UTC_PLUS_8).strftime("%Y-%m-%d %H:%M:%S")

    def _limit(self, limit: int) -> int:
        try:
            return min(50, max(1, int(limit)))
        except (TypeError, ValueError):
            return 20

    def _relative_deviation(self, left: float, right: float) -> float:
        denominator = max(abs(float(right)), 1e-9)
        return abs(float(left) - float(right)) / denominator


def openrca_tool_definitions(names: list[str] | None = None) -> list[dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {
        "get_candidate_catalog": {
            "type": "function",
            "name": "get_candidate_catalog",
            "description": "Return allowed OpenRCA Telecom root cause components and reasons.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        },
        "summarize_metric_anomalies": {
            "type": "function",
            "name": "summarize_metric_anomalies",
            "description": "Summarize bounded KPI anomalies in OpenRCA metric CSV files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_datetime": {"type": ["string", "null"]},
                    "end_datetime": {"type": ["string", "null"]},
                    "metric_file": {"type": ["string", "null"]},
                    "components": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                "required": ["start_datetime", "end_datetime", "metric_file", "components", "limit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "summarize_trace_latency": {
            "type": "function",
            "name": "summarize_trace_latency",
            "description": "Summarize slow and failed trace spans in OpenRCA trace_span.csv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_datetime": {"type": ["string", "null"]},
                    "end_datetime": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                },
                "required": ["start_datetime", "end_datetime", "limit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "search_logs": {
            "type": "function",
            "name": "search_logs",
            "description": "Search OpenRCA log files for a keyword with bounded output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_datetime": {"type": ["string", "null"]},
                    "end_datetime": {"type": ["string", "null"]},
                    "keyword": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                },
                "required": ["start_datetime", "end_datetime", "keyword", "limit"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    selected = names or list(definitions)
    return [definitions[name] for name in selected]


class OpenRCAToolExecutor:
    def __init__(self, tools: OpenRCATelemetryTools) -> None:
        self.tools = tools

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        tool: Callable[..., Any] | None = getattr(self.tools, name, None)
        if tool is None:
            raise KeyError(f"Unknown OpenRCA tool: {name}")
        return tool(**arguments)
