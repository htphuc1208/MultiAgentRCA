from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


OpenRCAField = Literal[
    "root cause occurrence datetime",
    "root cause component",
    "root cause reason",
]


class OpenRCATaskParse(BaseModel):
    """Structured interpretation of an OpenRCA natural-language task."""

    time_range_start: str | None = Field(
        default=None,
        description="Start datetime in '%Y-%m-%d %H:%M:%S' format using UTC+8, if present.",
    )
    time_range_end: str | None = Field(
        default=None,
        description="End datetime in '%Y-%m-%d %H:%M:%S' format using UTC+8, if present.",
    )
    number_of_failures: int | None = Field(default=None, ge=1)
    requested_fields: list[OpenRCAField] = Field(default_factory=list)
    rationale: str = ""


class OpenRCAInvestigationOutput(BaseModel):
    """Bounded telemetry investigation summary produced after tool use."""

    candidate_components: list[str] = Field(default_factory=list)
    candidate_reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    telemetry_gaps: list[str] = Field(default_factory=list)
    reasoning: str = ""


class OpenRCAPredictionItem(BaseModel):
    root_cause_occurrence_datetime: str | None = Field(
        default=None,
        description="Format '%Y-%m-%d %H:%M:%S' in UTC+8 when requested.",
    )
    root_cause_component: str | None = None
    root_cause_reason: str | None = None


class OpenRCAPredictionOutput(BaseModel):
    root_causes: list[OpenRCAPredictionItem] = Field(min_length=1)
    rationale: str = ""
