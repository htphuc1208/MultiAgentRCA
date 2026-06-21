from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


class OpenRCADatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRCARuntimeTask:
    row_id: int
    task_index: str
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "task_index": self.task_index,
            "instruction": self.instruction,
        }


class OpenRCADataset:
    """Loads OpenRCA query metadata and validates telemetry paths.

    Runtime methods intentionally exclude scoring_points and record.csv labels.
    Evaluation methods expose scoring_points only after prediction generation.
    """

    REQUIRED_QUERY_COLUMNS = {"task_index", "instruction", "scoring_points"}

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        dataset: str = "Telecom",
    ) -> None:
        root = root_dir or os.getenv("OPENRCA_DATA_DIR")
        if not root:
            raise OpenRCADatasetError(
                "OPENRCA_DATA_DIR is not set. Download OpenRCA data and set it to the directory "
                "containing Telecom/query.csv."
            )
        self.root_dir = Path(root).expanduser().resolve()
        self.dataset = dataset
        self.dataset_dir = self.root_dir / dataset
        self.query_path = self.dataset_dir / "query.csv"
        self.telemetry_dir = self.dataset_dir / "telemetry"
        self._query_df: pd.DataFrame | None = None
        self._validate()

    def _validate(self) -> None:
        if not self.dataset_dir.exists():
            raise OpenRCADatasetError(f"OpenRCA dataset directory does not exist: {self.dataset_dir}")
        if not self.query_path.exists():
            raise OpenRCADatasetError(f"Missing OpenRCA query file: {self.query_path}")
        if not self.telemetry_dir.exists():
            raise OpenRCADatasetError(f"Missing OpenRCA telemetry directory: {self.telemetry_dir}")
        date_dirs = self.telemetry_date_dirs()
        if not date_dirs:
            raise OpenRCADatasetError(f"No telemetry date directories found under: {self.telemetry_dir}")
        query_df = self.query_df
        missing = self.REQUIRED_QUERY_COLUMNS - set(query_df.columns)
        if missing:
            raise OpenRCADatasetError(
                f"{self.query_path} is missing required columns: {', '.join(sorted(missing))}"
            )

    @property
    def query_df(self) -> pd.DataFrame:
        if self._query_df is None:
            self._query_df = pd.read_csv(self.query_path)
        return self._query_df

    def telemetry_date_dirs(self) -> list[Path]:
        if not self.telemetry_dir.exists():
            return []
        return sorted(path for path in self.telemetry_dir.iterdir() if path.is_dir())

    def get_runtime_task(self, row_id: int) -> dict[str, Any]:
        row = self._row(row_id)
        return OpenRCARuntimeTask(
            row_id=row_id,
            task_index=str(row["task_index"]),
            instruction=str(row["instruction"]),
        ).to_dict()

    def get_scoring_points(self, row_id: int) -> str:
        return str(self._row(row_id)["scoring_points"])

    def iter_runtime_tasks(self, *, start_row: int = 0, limit: int | None = None) -> Iterator[dict[str, Any]]:
        stop = len(self.query_df) if limit is None else min(len(self.query_df), start_row + limit)
        for row_id in range(start_row, stop):
            yield self.get_runtime_task(row_id)

    def available_dates(self, limit: int = 20) -> list[str]:
        return [path.name for path in self.telemetry_date_dirs()[:limit]]

    def _row(self, row_id: int) -> pd.Series:
        if row_id < 0 or row_id >= len(self.query_df):
            raise OpenRCADatasetError(f"row_id {row_id} is outside query.csv range 0..{len(self.query_df) - 1}")
        return self.query_df.iloc[row_id]
