"""Independent required-topic representation for Model 2.

Scores are relevance/importance estimates, not percentages of knowledge a
person must possess.  Official tags provide strong evidence; transparent
statement-keyword matches and prerequisite links provide weaker evidence.
Model 1 topic predictions are deliberately never imported here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from .knowledge_graph import load_knowledge_graph, node_map, prerequisites, tag_topic_ids


def _keywords(node: dict[str, Any]) -> set[str]:
    values = set(node.get("keywords", [])) | set(node.get("aliases", []))
    values.add(node["name"])
    return {str(value).strip().lower() for value in values if str(value).strip()}


def required_topic_scores(
    statement: str,
    official_tags: Iterable[str],
    graph: dict[str, Any] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    graph = graph or load_knowledge_graph()
    nodes = node_map(graph)
    text = re.sub(r"\s+", " ", str(statement).lower())
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)

    for tag in official_tags:
        matches = tag_topic_ids(tag, graph)
        for topic_id in matches:
            scores[topic_id] = max(scores[topic_id], 0.90)
            evidence[topic_id].append(f"official tag: {tag}")

    for topic_id, node in nodes.items():
        matches = [keyword for keyword in _keywords(node) if len(keyword) >= 3 and keyword in text]
        if matches:
            keyword_score = min(0.78, 0.42 + 0.08 * len(matches))
            if keyword_score > scores[topic_id]:
                scores[topic_id] = keyword_score
            evidence[topic_id].append("statement terms: " + ", ".join(sorted(matches)[:4]))

    # A prerequisite matters, but less than a directly tagged/mentioned skill.
    direct = list(scores.items())
    for topic_id, relevance in direct:
        for prerequisite in prerequisites(topic_id, graph, transitive=False):
            propagated = relevance * 0.58
            if propagated > scores[prerequisite]:
                scores[prerequisite] = propagated
                evidence[prerequisite].append(f"prerequisite of {nodes[topic_id]['name']}")

    if not scores:
        scores["implementation"] = 0.45
        evidence["implementation"].append("default when metadata is sparse")

    results = [
        {
            "topic_id": topic_id,
            "topic": nodes[topic_id]["name"],
            "relevance": round(min(1.0, relevance), 4),
            "meaning": "estimated relevance/importance for solving this problem",
            "evidence": evidence[topic_id],
        }
        for topic_id, relevance in scores.items()
        if topic_id in nodes and relevance >= 0.25
    ]
    results.sort(key=lambda item: (-item["relevance"], item["topic"]))
    return results[:limit]
