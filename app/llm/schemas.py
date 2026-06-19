from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TriageDecision(BaseModel):
    domain: Literal["RAN", "Core", "Transport"]
    severity: Literal["Low", "Medium", "High", "Critical"]
    primary_ne: str
    affected_services: list[str] = Field(default_factory=list)
    service_impact: str
    intent: str
    rationale: list[str] = Field(default_factory=list)


class DataCollectionPlan(BaseModel):
    required_tools: list[str] = Field(default_factory=list)
    evidence_items: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    rationale: str


class TopologyReasoning(BaseModel):
    primary_ne: str
    neighbors: list[str] = Field(default_factory=list)
    blast_radius: list[str] = Field(default_factory=list)
    dependency_summary: str
    topology_risks: list[str] = Field(default_factory=list)


class RCAHypothesisModel(BaseModel):
    cause: str
    domain: Literal["RAN", "Core", "Transport"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    rationale: str


class RCAHypothesisOutput(BaseModel):
    hypotheses: list[RCAHypothesisModel] = Field(min_length=1, max_length=3)

    @field_validator("hypotheses")
    @classmethod
    def top_three_required(cls, value: list[RCAHypothesisModel]) -> list[RCAHypothesisModel]:
        if len(value) > 3:
            raise ValueError("RCA output must contain at most 3 hypotheses")
        return value


class SOPSelectionOutput(BaseModel):
    selected_sop_id: str
    selected_title: str
    candidate_sop_ids: list[str] = Field(default_factory=list)
    likely_causes: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    rationale: str


class VerifiedHypothesisModel(BaseModel):
    cause: str
    evidence_supported: bool
    evidence_refs: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    verifier_confidence: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class VerificationOutput(BaseModel):
    verified_hypotheses: list[VerifiedHypothesisModel] = Field(min_length=1)
    verification_notes: list[str] = Field(default_factory=list)


class RemediationPlanOutput(BaseModel):
    recommended_actions: list[str] = Field(min_length=1)
    validation_plan: list[str] = Field(default_factory=list)
    human_approval_required: bool
    rollback_plan: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)


class ValidationSummaryOutput(BaseModel):
    status: Literal["validated", "needs_review"]
    summary: str
    passed_checks: int
    total_checks: int
    follow_up_actions: list[str] = Field(default_factory=list)

