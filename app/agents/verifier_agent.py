from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.prompts import VERIFIER_PROMPT
from app.llm.schemas import VerificationOutput
from app.models import Hypothesis


class ConsensusVerifierAgent(BaseAgent):
    name = "Consensus & Verifier Agent"

    def run(self) -> dict[str, Any]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        hypotheses: list[Hypothesis] = self.blackboard.get("hypotheses", [])
        sop_context = self.blackboard.get("sop_context")
        topology_context = self.blackboard.get("topology_context")
        data_evidence = self.blackboard.get("data_evidence")
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=VERIFIER_PROMPT,
            user_payload={
                "incident": self._incident_stub(incident),
                "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
                "sop_context": sop_context,
                "topology_context": topology_context,
                "data_evidence": data_evidence,
            },
            response_model=VerificationOutput,
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        self._merge_llm_verification(hypotheses, parsed)
        return self._run_rule(start_tool_index=start, action_prefix="llm ")

    def _run_rule(self, start_tool_index: int | None = None, action_prefix: str = "") -> dict[str, Any]:
        start = len(self.blackboard.tool_calls) if start_tool_index is None else start_tool_index
        incident = self.blackboard.get("incident")
        hypotheses: list[Hypothesis] = self.blackboard.get("hypotheses", [])
        sop_context = self.blackboard.get("sop_context")
        topology_context = self.blackboard.get("topology_context")
        data_evidence = self.blackboard.get("data_evidence")

        scored = [
            self._score_hypothesis(hypothesis, incident, sop_context, topology_context, data_evidence)
            for hypothesis in hypotheses
        ]
        scored.sort(key=lambda item: item.scores["consensus_score"], reverse=True)
        selected = scored[0]
        output = {
            "selected_hypothesis": selected,
            "ranked_hypotheses": scored,
            "verification_notes": self._verification_notes(selected),
        }
        self.blackboard.set("verified_hypotheses", scored)
        self.blackboard.set("selected_hypothesis", selected)
        return self._record(
            f"{action_prefix}score and verify hypotheses".strip(),
            f"Selected '{selected.cause}' with consensus score {selected.scores['consensus_score']:.2f}.",
            {"incident_id": incident["incident_id"], "hypothesis_count": len(hypotheses)},
            {
                "selected_hypothesis": selected.to_dict(),
                "ranked_hypotheses": [hypothesis.to_dict() for hypothesis in scored],
                "verification_notes": output["verification_notes"],
            },
            start,
        )

    def _incident_stub(self, incident: dict[str, Any]) -> dict[str, Any]:
        return {
            "incident_id": incident["incident_id"],
            "domain": incident.get("domain"),
            "symptom": incident.get("symptom"),
            "primary_ne": incident.get("primary_ne"),
            "service_impact": incident.get("service_impact"),
        }

    def _merge_llm_verification(self, hypotheses: list[Hypothesis], parsed: VerificationOutput) -> None:
        verified = {item.cause.lower(): item for item in parsed.verified_hypotheses}
        for hypothesis in hypotheses:
            item = verified.get(hypothesis.cause.lower())
            if not item:
                continue
            if item.evidence_refs:
                hypothesis.evidence_refs = item.evidence_refs
            hypothesis.contradictions = list(dict.fromkeys([*hypothesis.contradictions, *item.contradictions]))
            hypothesis.verification_notes = item.notes or parsed.verification_notes
            if item.unsupported_claims:
                hypothesis.missing_evidence = list(
                    dict.fromkeys([*hypothesis.missing_evidence, *item.unsupported_claims])
                )
            hypothesis.confidence = round((hypothesis.confidence + item.verifier_confidence) / 2, 3)

    def _score_hypothesis(
        self,
        hypothesis: Hypothesis,
        incident: dict[str, Any],
        sop_context: dict[str, Any],
        topology_context: dict[str, Any],
        data_evidence: dict[str, Any],
    ) -> Hypothesis:
        evidence_match = min(1.0, len(hypothesis.evidence) / 4)
        topology_consistency = self._topology_consistency(hypothesis, incident, topology_context)
        sop_alignment = self._sop_alignment(hypothesis, sop_context)
        historical_similarity = self._historical_similarity(hypothesis, data_evidence)
        agent_vote_confidence = hypothesis.confidence
        contradictions = self._find_contradictions(hypothesis, data_evidence)

        consensus = (
            0.35 * evidence_match
            + 0.25 * topology_consistency
            + 0.20 * sop_alignment
            + 0.10 * historical_similarity
            + 0.10 * agent_vote_confidence
        )
        if contradictions:
            consensus = max(0.0, consensus - 0.2)
        hypothesis.contradictions = list(dict.fromkeys([*hypothesis.contradictions, *contradictions]))
        hypothesis.scores = {
            "evidence_match": round(evidence_match, 3),
            "topology_consistency": round(topology_consistency, 3),
            "sop_alignment": round(sop_alignment, 3),
            "historical_similarity": round(historical_similarity, 3),
            "agent_vote_confidence": round(agent_vote_confidence, 3),
            "consensus_score": round(consensus, 3),
        }
        if self.name not in hypothesis.source_agents:
            hypothesis.source_agents.append(self.name)
        return hypothesis

    def _topology_consistency(
        self,
        hypothesis: Hypothesis,
        incident: dict[str, Any],
        topology_context: dict[str, Any],
    ) -> float:
        cause = hypothesis.cause.lower()
        primary_ne = incident["primary_ne"].lower()
        nodes = " ".join(node["id"].lower() for node in topology_context.get("topology", {}).get("nodes", []))
        if primary_ne in cause or primary_ne in nodes and any(term in cause for term in ["link", "fiber", "cell", "upf", "amf", "smf", "dns"]):
            return 1.0
        if any(term in cause for term in ["overload", "database", "resolver", "routing", "configuration"]):
            return 0.75
        return 0.45

    def _sop_alignment(self, hypothesis: Hypothesis, sop_context: dict[str, Any]) -> float:
        cause_text = hypothesis.cause.lower()
        likely_causes = [cause.lower() for cause in sop_context.get("likely_causes", [])]
        if any(self._token_overlap(cause_text, likely) >= 0.5 for likely in likely_causes):
            return 1.0
        sop_text = " ".join(sop_context.get("sop", {}).get("steps", [])).lower()
        if any(token in sop_text for token in cause_text.split()):
            return 0.65
        return 0.35

    def _historical_similarity(self, hypothesis: Hypothesis, data_evidence: dict[str, Any]) -> float:
        history = " ".join(ticket.get("summary", "") for ticket in data_evidence.get("ticket_history", [])).lower()
        if not history:
            return 0.4
        return 0.9 if self._token_overlap(hypothesis.cause.lower(), history) >= 0.3 else 0.55

    def _find_contradictions(self, hypothesis: Hypothesis, data_evidence: dict[str, Any]) -> list[str]:
        diagnostics = " ".join(data_evidence.get("diagnostics", {}).values()).lower()
        cause = hypothesis.cause.lower()
        contradictions = []
        if "backhaul" in cause and "0% packet loss" in diagnostics and "link up" in diagnostics:
            contradictions.append("Backhaul hypothesis conflicts with clean packet-loss diagnostic.")
        if "interference" in cause and "sinr normal" in diagnostics:
            contradictions.append("Interference hypothesis conflicts with normal SINR diagnostic.")
        return contradictions

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = {token for token in left.replace("/", " ").split() if len(token) > 2}
        right_tokens = {token for token in right.replace("/", " ").split() if len(token) > 2}
        if not left_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens)

    def _verification_notes(self, hypothesis: Hypothesis) -> list[str]:
        notes = [
            f"Evidence score: {hypothesis.scores['evidence_match']:.2f}",
            f"Topology score: {hypothesis.scores['topology_consistency']:.2f}",
            f"SOP alignment: {hypothesis.scores['sop_alignment']:.2f}",
        ]
        if hypothesis.contradictions:
            notes.extend(hypothesis.contradictions)
        return notes
