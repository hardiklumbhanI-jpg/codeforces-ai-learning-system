"""Topic taxonomy and prerequisite graph helpers for Model 2."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .config import KNOWLEDGE_GRAPH_PATH


def load_knowledge_graph(path: Path | str = KNOWLEDGE_GRAPH_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        graph = json.load(file)
    validate_knowledge_graph(graph)
    return graph


def validate_knowledge_graph(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes", [])
    node_ids = [node.get("id") for node in nodes]
    if len(node_ids) != len(set(node_ids)) or any(not item for item in node_ids):
        raise ValueError("Knowledge graph node ids must be non-empty and unique.")
    node_set = set(node_ids)
    for edge in graph.get("edges", []):
        if edge.get("from") not in node_set or edge.get("to") not in node_set:
            raise ValueError(f"Knowledge graph edge references an unknown node: {edge}")


def node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in graph["nodes"]}


def tag_topic_ids(tag: str, graph: dict[str, Any]) -> list[str]:
    normalized = str(tag).strip().lower()
    matches = []
    for node in graph["nodes"]:
        aliases = {str(item).lower() for item in node.get("aliases", [])}
        aliases.add(node["name"].lower())
        aliases.add(node["id"].replace("_", " ").lower())
        if normalized in aliases:
            matches.append(node["id"])
    return matches


def prerequisites(topic_id: str, graph: dict[str, Any], transitive: bool = True) -> list[str]:
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        if edge.get("relation") == "prerequisite":
            incoming[edge["to"]].append(edge["from"])
    if not transitive:
        return sorted(incoming.get(topic_id, []))
    found: set[str] = set()
    queue: deque[str] = deque(incoming.get(topic_id, []))
    while queue:
        node = queue.popleft()
        if node in found:
            continue
        found.add(node)
        queue.extend(incoming.get(node, []))
    return sorted(found)


def parent_topics(topic_id: str, graph: dict[str, Any]) -> list[str]:
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        if edge.get("relation") == "is_a":
            parents[edge["from"]].append(edge["to"])
    return parents.get(topic_id, [])


def expand_topics(topic_ids: list[str], graph: dict[str, Any]) -> list[str]:
    """Add taxonomy parents while preserving deterministic order."""
    result: list[str] = []
    seen: set[str] = set()
    queue: deque[str] = deque(topic_ids)
    while queue:
        topic = queue.popleft()
        if topic in seen:
            continue
        seen.add(topic)
        result.append(topic)
        queue.extend(parent_topics(topic, graph))
    return result
