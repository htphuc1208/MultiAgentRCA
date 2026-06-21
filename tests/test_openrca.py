import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.openrca.dataset import OpenRCADataset
from app.openrca.evaluator import evaluate_prediction
from app.openrca.formatter import format_prediction
from app.openrca.schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput
from app.openrca.tools import OpenRCATelemetryTools


class OpenRCATest(unittest.TestCase):
    def test_loader_runtime_task_does_not_leak_scoring_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_dataset(Path(tmp))
            dataset = OpenRCADataset(root)

            task = dataset.get_runtime_task(0)

            self.assertEqual(task["row_id"], 0)
            self.assertEqual(task["task_index"], "task_7")
            self.assertIn("instruction", task)
            self.assertNotIn("scoring_points", task)
            self.assertNotIn("record", task)
            self.assertIn("docker_001", dataset.get_scoring_points(0))

    def test_formatter_and_evaluator_accept_openrca_prediction_shape(self) -> None:
        prediction = format_prediction(
            OpenRCAPredictionOutput(
                root_causes=[
                    OpenRCAPredictionItem(
                        root_cause_occurrence_datetime="2020-04-11 00:15:30",
                        root_cause_component="docker_001",
                        root_cause_reason="CPU fault",
                    )
                ]
            )
        )
        scoring_points = "\n".join(
            [
                "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2020-04-11 00:15:00",
                "The only predicted root cause component is docker_001",
                "The only predicted root cause reason is CPU fault",
            ]
        )

        passed, failed, score = evaluate_prediction(prediction, scoring_points)

        self.assertEqual(score, 1.0)
        self.assertEqual(failed, [])
        self.assertIn("docker_001", passed)

    def test_evaluator_handles_multi_root_cause_permutation(self) -> None:
        prediction = format_prediction(
            OpenRCAPredictionOutput(
                root_causes=[
                    OpenRCAPredictionItem(root_cause_component="docker_001", root_cause_reason="network loss"),
                    OpenRCAPredictionItem(root_cause_component="os_001", root_cause_reason="CPU fault"),
                ]
            )
        )
        scoring_points = "\n".join(
            [
                "The 1-th predicted root cause component is os_001",
                "The 1-th predicted root cause reason is CPU fault",
                "The 2-th predicted root cause component is docker_001",
                "The 2-th predicted root cause reason is network loss",
            ]
        )

        _, failed, score = evaluate_prediction(prediction, scoring_points)

        self.assertEqual(score, 1.0)
        self.assertEqual(failed, [])

    def test_metric_tool_returns_bounded_anomaly_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_dataset(Path(tmp))
            metric_path = root / "Telecom" / "telemetry" / "2020_04_11" / "metric" / "metric_container.csv"
            metric_path.write_text(
                "\n".join(
                    [
                        "itemid,name,bomc_id,timestamp,value,cmdb_id",
                        f"1,cpu_usage,ZJ,{self._ms('2020-04-11 00:00:00')},1.0,docker_001",
                        f"2,cpu_usage,ZJ,{self._ms('2020-04-11 00:05:00')},1.1,docker_001",
                        f"3,cpu_usage,ZJ,{self._ms('2020-04-11 00:10:00')},1.2,docker_001",
                        f"4,cpu_usage,ZJ,{self._ms('2020-04-11 00:15:00')},50.0,docker_001",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = OpenRCADataset(root)
            tools = OpenRCATelemetryTools(dataset, chunksize=2)

            summary = tools.summarize_metric_anomalies(
                "2020-04-11 00:15:00",
                "2020-04-11 00:16:00",
                metric_file="metric_container.csv",
                components=["docker_001"],
                limit=3,
            )

            self.assertEqual(summary["files_scanned"], 1)
            self.assertLessEqual(len(summary["anomalies"]), 3)
            self.assertEqual(summary["anomalies"][0]["component"], "docker_001")
            self.assertEqual(summary["anomalies"][0]["metric"], "cpu_usage")
            self.assertEqual(summary["anomalies"][0]["direction"], "high")

    def _fixture_dataset(self, root: Path) -> Path:
        telecom = root / "Telecom"
        metric_dir = telecom / "telemetry" / "2020_04_11" / "metric"
        trace_dir = telecom / "telemetry" / "2020_04_11" / "trace"
        metric_dir.mkdir(parents=True)
        trace_dir.mkdir(parents=True)
        (trace_dir / "trace_span.csv").write_text(
            "callType,startTime,elapsedTime,success,traceId,id,pid,cmdb_id,dsName,serviceName\n",
            encoding="utf-8",
        )
        (metric_dir / "metric_container.csv").write_text(
            "itemid,name,bomc_id,timestamp,value,cmdb_id\n",
            encoding="utf-8",
        )
        (telecom / "query.csv").write_text(
            "\n".join(
                [
                    "task_index,instruction,scoring_points",
                    '"task_7","Find the root cause between 2020-04-11 00:00:00 and 2020-04-11 00:30:00","The only predicted root cause component is docker_001"',
                ]
            ),
            encoding="utf-8",
        )
        return root

    def _ms(self, text: str) -> int:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return int(dt.timestamp() * 1000)


if __name__ == "__main__":
    unittest.main()
