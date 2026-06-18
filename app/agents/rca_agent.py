from __future__ import annotations

import re
from typing import Any

from app.agents.base import BaseAgent
from app.models import Hypothesis


class RCAAgent(BaseAgent):
    name = "RCA Agent"

    def run(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        data = self.blackboard.get("data_evidence")
        topology = self.blackboard.get("topology_context")
        corpus = self._corpus(data, topology, incident)
        hypotheses = self._generate_hypotheses(triage["domain"], triage["primary_ne"], corpus)
        output = {"hypotheses": hypotheses}
        self.blackboard.set("hypotheses", hypotheses)
        return self._record(
            "generate root-cause hypotheses",
            f"Generated {len(hypotheses)} candidate root causes.",
            {"domain": triage["domain"], "primary_ne": triage["primary_ne"]},
            {"hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses]},
            start,
        )

    def _corpus(
        self,
        data: dict[str, Any],
        topology: dict[str, Any],
        incident: dict[str, Any],
    ) -> dict[str, Any]:
        evidence = data.get("evidence_items", [])
        text = " ".join(evidence + [incident.get("symptom", ""), topology.get("dependency_summary", "")])
        return {"text": text.lower(), "evidence_items": evidence, "incident": incident}

    def _generate_hypotheses(
        self,
        domain: str,
        primary_ne: str,
        corpus: dict[str, Any],
    ) -> list[Hypothesis]:
        if domain == "RAN":
            candidates = self._ran_candidates(primary_ne, corpus)
        elif domain == "Core":
            candidates = self._core_candidates(corpus)
        else:
            candidates = self._transport_candidates(primary_ne, corpus)

        candidates.extend(self._fallback_candidates(domain, primary_ne))
        unique: dict[str, Hypothesis] = {}
        for candidate in candidates:
            unique.setdefault(candidate.cause, candidate)
        return sorted(unique.values(), key=lambda item: item.confidence, reverse=True)[:3]

    def _ran_candidates(self, primary_ne: str, corpus: dict[str, Any]) -> list[Hypothesis]:
        text = corpus["text"]
        candidates = []
        if ("low sinr" in text or "sinr_avg" in text) and ("high prb" in text or "prb_utilization" in text):
            candidates.append(
                self._hypothesis(
                    f"Interference on {primary_ne}",
                    "RAN",
                    ["sinr", "prb", "interference", "neighbor cells normal", "0% packet loss"],
                    corpus,
                )
            )
        if "backhaul" in text and any(term in text for term in ["down", "loss", "unreachable", "link down"]):
            candidates.append(
                self._hypothesis(
                    f"Backhaul link failure on {primary_ne}",
                    "RAN",
                    ["backhaul", "link down", "packet loss", "unreachable", "n2"],
                    corpus,
                )
            )
        if "pci" in text or "handover" in text and "collision" in text:
            cells = re.findall(r"cell-\d+", text, flags=re.IGNORECASE)
            if len(cells) >= 2:
                cause = f"PCI collision between {cells[0].title()} and {cells[1].title()}"
            else:
                cause = "PCI collision between neighboring cells"
            candidates.append(self._hypothesis(cause, "RAN", ["pci", "collision", "handover"], corpus))
        if "rru" in text or "radio unit" in text or "power module" in text:
            candidates.append(
                self._hypothesis(
                    "Remote radio unit power failure",
                    "RAN",
                    ["rru", "power", "cell unavailable", "vswr"],
                    corpus,
                )
            )
        return candidates

    def _core_candidates(self, corpus: dict[str, Any]) -> list[Hypothesis]:
        text = corpus["text"]
        candidates = []
        if "smf" in text and ("timeout" in text or "database" in text or "policy" in text):
            candidates.append(
                self._hypothesis(
                    "SMF policy database timeout",
                    "Core",
                    ["smf", "policy", "database", "timeout", "pdu session"],
                    corpus,
                )
            )
        if "amf" in text and ("overload" in text or "registration" in text or "signaling" in text):
            candidates.append(
                self._hypothesis(
                    "AMF overload after signaling spike",
                    "Core",
                    ["amf", "overload", "registration", "signaling", "cpu"],
                    corpus,
                )
            )
        if "upf" in text and ("cpu" in text or "saturation" in text or "throughput" in text):
            candidates.append(
                self._hypothesis("UPF CPU saturation", "Core", ["upf", "cpu", "saturation", "throughput"], corpus)
            )
        if "dns" in text or "nrf" in text and "resolver" in text:
            candidates.append(
                self._hypothesis(
                    "Core DNS resolver outage",
                    "Core",
                    ["dns", "resolver", "nrf", "service discovery", "servfail"],
                    corpus,
                )
            )
        return candidates

    def _transport_candidates(self, primary_ne: str, corpus: dict[str, Any]) -> list[Hypothesis]:
        text = corpus["text"]
        candidates = []
        if "optical" in text or "attenuation" in text or "rx power" in text:
            candidates.append(
                self._hypothesis(
                    f"Fiber attenuation on {primary_ne}",
                    "Transport",
                    ["optical", "attenuation", "rx power", "crc", "loss"],
                    corpus,
                )
            )
        if "admin down" in text or "change window" in text:
            candidates.append(
                self._hypothesis(
                    "Router interface admin down after change",
                    "Transport",
                    ["admin down", "change", "interface down", "commit"],
                    corpus,
                )
            )
        if "bgp" in text and ("mtu" in text or "fragmentation" in text or "flap" in text):
            candidates.append(
                self._hypothesis(
                    "BGP neighbor instability due to MTU mismatch",
                    "Transport",
                    ["bgp", "mtu", "flap", "fragmentation"],
                    corpus,
                )
            )
        if "congestion" in text or "utilization" in text and "queue" in text:
            candidates.append(
                self._hypothesis(
                    f"Transport link congestion on {primary_ne}",
                    "Transport",
                    ["congestion", "utilization", "queue", "drops", "latency"],
                    corpus,
                )
            )
        return candidates

    def _fallback_candidates(self, domain: str, primary_ne: str) -> list[Hypothesis]:
        defaults = {
            "RAN": [
                f"Configuration drift on {primary_ne}",
                f"Backhaul degradation affecting {primary_ne}",
                f"Radio interference near {primary_ne}",
            ],
            "Core": [
                "Control-plane overload",
                "Database dependency timeout",
                "Service discovery failure",
            ],
            "Transport": [
                f"Physical link degradation on {primary_ne}",
                "Routing instability",
                "Interface configuration error",
            ],
        }
        return [
            Hypothesis(
                cause=cause,
                domain=domain,
                confidence=0.25,
                evidence=[],
                missing_evidence=["No direct supporting telemetry in current window"],
                source_agents=[self.name],
            )
            for cause in defaults[domain]
        ]

    def _hypothesis(
        self,
        cause: str,
        domain: str,
        terms: list[str],
        corpus: dict[str, Any],
    ) -> Hypothesis:
        evidence = self._matching_evidence(terms, corpus["evidence_items"])
        matched_terms = {term for term in terms if term in corpus["text"]}
        confidence = min(0.95, 0.42 + 0.1 * len(evidence) + 0.04 * len(matched_terms))
        missing = [term for term in terms if term not in corpus["text"]]
        return Hypothesis(
            cause=cause,
            domain=domain,
            confidence=round(confidence, 2),
            evidence=evidence[:6],
            missing_evidence=missing[:4],
            source_agents=[self.name],
        )

    def _matching_evidence(self, terms: list[str], evidence_items: list[str]) -> list[str]:
        matches = []
        for item in evidence_items:
            text = item.lower()
            if any(term in text for term in terms):
                matches.append(item)
        return matches

