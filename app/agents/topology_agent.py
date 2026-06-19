from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from app.agents.base import BaseAgent
from app.llm.prompts import TOPOLOGY_PROMPT
from app.llm.schemas import TopologyReasoning
from app.tools.registry import TelecomToolExecutor, telecom_tool_definitions


class TopologyAgent(BaseAgent):
    name = "Topology Agent"

    def run(self) -> dict[str, Any]:
        if self.use_llm:
            return self._run_llm()
        return self._run_rule()

    def _run_llm(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        triage = self.blackboard.get("triage")
        parsed, llm_call = self.llm_client.structured(
            agent=self.name,
            system_prompt=TOPOLOGY_PROMPT,
            user_payload={"incident": self._incident_stub(incident), "triage": triage},
            response_model=TopologyReasoning,
            tools=telecom_tool_definitions(["get_topology"]),
            tool_executor=TelecomToolExecutor(self.tools),
            max_tool_calls=self.max_tool_calls,
        )
        self._record_llm(llm_call)
        topology = {}
        for call in llm_call.tool_calls:
            if call["name"] == "get_topology":
                topology = call.get("output") or {}
                break
        if not topology:
            topology = self.tools.get_topology(triage["primary_ne"], incident["incident_id"])
        output = parsed.model_dump()
        output["topology"] = topology
        self.blackboard.set("topology_context", output)
        return self._record(
            "llm map dependencies",
            f"LLM mapped topology around {triage['primary_ne']}.",
            {"incident_id": incident["incident_id"], "primary_ne": triage["primary_ne"]},
            output,
            start,
        )

    def _run_rule(self) -> dict[str, Any]:
        start = len(self.blackboard.tool_calls)
        incident = self.blackboard.get("incident")
        primary_ne = self.blackboard.get("triage")["primary_ne"]
        topology = self.tools.get_topology(primary_ne, incident["incident_id"])
        neighbors = self._neighbors(topology, primary_ne)
        blast_radius = self._reachable(topology, primary_ne, limit=5)
        output = {
            "topology": topology,
            "primary_ne": primary_ne,
            "neighbors": neighbors,
            "blast_radius": blast_radius,
            "dependency_summary": self._dependency_summary(primary_ne, neighbors, blast_radius),
        }
        self.blackboard.set("topology_context", output)
        return self._record(
            "map dependencies",
            f"Mapped {len(neighbors)} direct neighbors around {primary_ne}.",
            {"incident_id": incident["incident_id"], "primary_ne": primary_ne},
            output,
            start,
        )

    def _incident_stub(self, incident: dict[str, Any]) -> dict[str, Any]:
        return {
            "incident_id": incident["incident_id"],
            "symptom": incident.get("symptom"),
            "time_window": incident.get("time_window"),
            "primary_ne": incident.get("primary_ne"),
        }

    def _neighbors(self, topology: dict[str, Any], node: str) -> list[str]:
        neighbors: set[str] = set()
        for left, right in topology.get("edges", []):
            if left == node:
                neighbors.add(right)
            if right == node:
                neighbors.add(left)
        return sorted(neighbors)

    def _reachable(self, topology: dict[str, Any], start: str, limit: int) -> list[str]:
        graph: dict[str, list[str]] = defaultdict(list)
        for left, right in topology.get("edges", []):
            graph[left].append(right)
            graph[right].append(left)
        seen = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        out = []
        while queue:
            node, depth = queue.popleft()
            if depth >= limit:
                continue
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    out.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return out

    def _dependency_summary(self, primary_ne: str, neighbors: list[str], blast_radius: list[str]) -> str:
        if not neighbors:
            return f"{primary_ne} has no modeled direct dependencies."
        return f"{primary_ne} is adjacent to {', '.join(neighbors)}; reachable impact includes {', '.join(blast_radius[:5])}."
