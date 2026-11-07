import math
import re


TOPIC_KNOWLEDGE = {
    "binary search": {
        "topic": "Binary Search",
        "concepts": [
            "Sorted search space",
            "Monotonic condition",
            "Lower bound and upper bound",
        ],
        "techniques": [
            "Binary search on answer",
            "Two-pointer boundary checks",
        ],
        "must_know": ["Arrays", "Sorting", "Loops"],
        "should_know": ["Monotonic functions", "Time complexity"],
        "advanced": ["Binary search on real values"],
    },
    "brute force": {
        "topic": "Brute Force",
        "concepts": ["Complete search", "State enumeration"],
        "techniques": ["Nested loops", "Backtracking"],
        "must_know": ["Loops", "Conditionals", "Arrays"],
        "should_know": ["Complexity estimation"],
        "advanced": ["Pruning search space"],
    },
    "constructive algorithms": {
        "topic": "Constructive Algorithms",
        "concepts": ["Pattern observation", "Invariant building"],
        "techniques": ["Greedy construction", "Case analysis"],
        "must_know": ["Implementation", "Math basics"],
        "should_know": ["Invariants", "Proof by construction"],
        "advanced": ["Adversarial examples"],
    },
    "data structures": {
        "topic": "Data Structures",
        "concepts": ["Storing state", "Efficient update and query"],
        "techniques": ["Maps", "Sets", "Fenwick tree", "Segment tree"],
        "must_know": ["Arrays", "Sorting", "Hash maps"],
        "should_know": ["Stacks", "Queues", "Priority queues"],
        "advanced": ["Lazy segment tree", "Sparse table"],
    },
    "dfs and similar": {
        "topic": "Graph Traversal",
        "concepts": ["Visited state", "Connected components"],
        "techniques": ["DFS", "BFS"],
        "must_know": ["Graph representation", "Recursion", "Queues"],
        "should_know": ["Connected components", "Tree traversal"],
        "advanced": ["Low-link values", "Topological order"],
    },
    "divide and conquer": {
        "topic": "Divide and Conquer",
        "concepts": ["Split problem", "Merge answers"],
        "techniques": ["Recursive decomposition", "Merge sort idea"],
        "must_know": ["Recursion", "Arrays"],
        "should_know": ["Recurrence complexity"],
        "advanced": ["CDQ divide and conquer"],
    },
    "dp": {
        "topic": "Dynamic Programming",
        "concepts": ["State definition", "Transition", "Base case"],
        "techniques": ["Memoization", "Tabulation", "State compression"],
        "must_know": ["Arrays", "Recursion", "Complexity estimation"],
        "should_know": ["State transition", "Overlapping subproblems"],
        "advanced": ["Bitmask DP", "Tree DP", "DP on graphs"],
    },
    "dsu": {
        "topic": "Disjoint Set Union",
        "concepts": ["Components", "Union and find operations"],
        "techniques": ["Path compression", "Union by size"],
        "must_know": ["Arrays", "Trees"],
        "should_know": ["Graph components"],
        "advanced": ["Rollback DSU"],
    },
    "flows": {
        "topic": "Network Flow",
        "concepts": ["Capacity", "Residual graph", "Augmenting path"],
        "techniques": ["Max flow", "Min cut", "Matching"],
        "must_know": ["Graphs", "BFS", "DFS"],
        "should_know": ["Shortest paths", "Greedy"],
        "advanced": ["Dinic algorithm", "Min-cost max-flow"],
    },
    "games": {
        "topic": "Game Theory",
        "concepts": ["Winning state", "Losing state"],
        "techniques": ["Nim idea", "Minimax reasoning"],
        "must_know": ["Logic", "Recursion"],
        "should_know": ["State transition"],
        "advanced": ["Sprague-Grundy theorem"],
    },
    "geometry": {
        "topic": "Geometry",
        "concepts": ["Points", "Lines", "Area", "Distance"],
        "techniques": ["Cross product", "Orientation test"],
        "must_know": ["Coordinate geometry", "Math basics"],
        "should_know": ["Vectors", "Precision handling"],
        "advanced": ["Convex hull", "Rotating calipers"],
    },
    "graphs": {
        "topic": "Graphs",
        "concepts": ["Nodes and edges", "Traversal", "Paths"],
        "techniques": ["BFS", "DFS", "Shortest path"],
        "must_know": ["Graph representation", "Queues", "Recursion"],
        "should_know": ["Connected components", "Topological sort"],
        "advanced": ["Dijkstra", "MST", "SCC"],
    },
    "greedy": {
        "topic": "Greedy Algorithms",
        "concepts": ["Local choice", "Exchange argument"],
        "techniques": ["Sorting-based greedy", "Priority queue greedy"],
        "must_know": ["Sorting", "Arrays"],
        "should_know": ["Proof of correctness", "Counterexamples"],
        "advanced": ["Greedy with data structures"],
    },
    "hashing": {
        "topic": "Hashing",
        "concepts": ["Hash table", "Collision idea"],
        "techniques": ["String hashing", "Frequency maps"],
        "must_know": ["Strings", "Maps"],
        "should_know": ["Modular arithmetic"],
        "advanced": ["Rolling hash"],
    },
    "implementation": {
        "topic": "Implementation",
        "concepts": ["Careful simulation", "Edge cases"],
        "techniques": ["Parsing", "State simulation"],
        "must_know": ["Loops", "Conditionals", "Arrays"],
        "should_know": ["Debugging", "Boundary cases"],
        "advanced": ["Complex simulations"],
    },
    "math": {
        "topic": "Mathematics",
        "concepts": ["Formulas", "Parity", "Counting"],
        "techniques": ["Algebra", "Modulo arithmetic"],
        "must_know": ["Basic arithmetic", "Divisibility"],
        "should_know": ["Combinatorics", "GCD"],
        "advanced": ["Number theory", "Probability"],
    },
    "number theory": {
        "topic": "Number Theory",
        "concepts": ["Divisibility", "Prime numbers", "GCD"],
        "techniques": ["Sieve", "Modular inverse", "Factorization"],
        "must_know": ["Math basics", "Modulo"],
        "should_know": ["GCD", "Prime factorization"],
        "advanced": ["Euler phi", "Chinese remainder theorem"],
    },
    "sortings": {
        "topic": "Sorting",
        "concepts": ["Ordering", "Comparator logic"],
        "techniques": ["Sort then scan", "Custom sorting"],
        "must_know": ["Arrays", "Comparison"],
        "should_know": ["Two pointers", "Greedy choice"],
        "advanced": ["Coordinate compression"],
    },
    "strings": {
        "topic": "Strings",
        "concepts": ["Characters", "Substrings", "Prefixes"],
        "techniques": ["Prefix function", "Z-function", "Hashing"],
        "must_know": ["Arrays", "Loops"],
        "should_know": ["String traversal"],
        "advanced": ["KMP", "Trie", "Suffix array"],
    },
    "trees": {
        "topic": "Trees",
        "concepts": ["Rooted tree", "Parent-child relation", "Subtree"],
        "techniques": ["DFS on tree", "Tree DP", "Binary lifting"],
        "must_know": ["Graphs", "DFS", "Recursion"],
        "should_know": ["Depth", "Subtree traversal"],
        "advanced": ["LCA", "Heavy-light decomposition"],
    },
    "two pointers": {
        "topic": "Two Pointers",
        "concepts": ["Left and right pointers", "Window movement"],
        "techniques": ["Sliding window", "Meet in the middle scan"],
        "must_know": ["Arrays", "Sorting"],
        "should_know": ["Monotonic movement"],
        "advanced": ["Variable-size window"],
    },
}


RATING_PATH = [
    (800, "Implementation basics"),
    (1000, "Arrays and sorting"),
    (1200, "Greedy basics"),
    (1400, "Binary search and two pointers"),
    (1600, "Basic dynamic programming"),
    (1800, "Graphs, DFS, BFS, and stronger DP"),
    (2000, "Advanced DP and data structures"),
    (2200, "Trees, DSU, shortest paths"),
    (2400, "Segment tree, math, and harder graph ideas"),
    (2600, "Advanced techniques and proof-heavy problems"),
]


def parse_tags(tags):
    return [
        tag.strip().lower()
        for tag in str(tags).split(";")
        if tag.strip()
    ]


def unique_items(items):
    seen = set()
    result = []

    for item in items:
        normalized = item.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(item)

    return result


def knowledge_for_tags(tags):
    parsed_tags = parse_tags(tags)
    topics = []
    concepts = []
    techniques = []
    must_know = []
    should_know = []
    advanced = []

    for tag in parsed_tags:
        info = TOPIC_KNOWLEDGE.get(tag)

        if not info:
            topics.append(tag.title())
            continue

        topics.append(info["topic"])
        concepts.extend(info["concepts"])
        techniques.extend(info["techniques"])
        must_know.extend(info["must_know"])
        should_know.extend(info["should_know"])
        advanced.extend(info["advanced"])

    return {
        "topics": unique_items(topics),
        "concepts": unique_items(concepts),
        "techniques": unique_items(techniques),
        "must_know": unique_items(must_know),
        "should_know": unique_items(should_know),
        "advanced": unique_items(advanced),
    }


def extract_constraints(statement):
    text = str(statement)
    pattern = re.compile(
        r"(?P<var>[a-zA-Z][a-zA-Z0-9_]*)\s*(?:<=|≤|<|\\leq|\\le)\s*"
        r"(?P<value>\d+(?:\s*\^\s*\d+)?|\d+e\d+)",
        re.IGNORECASE,
    )
    constraints = []

    for match in pattern.finditer(text):
        raw_value = match.group("value").replace(" ", "")
        numeric_value = parse_constraint_value(raw_value)

        constraints.append({
            "variable": match.group("var"),
            "raw_value": raw_value,
            "value": numeric_value,
        })

    return constraints


def parse_constraint_value(raw_value):
    if "^" in raw_value:
        base, exponent = raw_value.split("^", 1)
        return int(base) ** int(exponent)

    if "e" in raw_value.lower():
        return int(float(raw_value))

    return int(raw_value)


def complexity_advice(constraints):
    if not constraints:
        return {
            "largest_constraint": None,
            "recommended": ["Estimate complexity from input size manually"],
            "avoid": [],
            "notes": ["No simple upper-bound constraint was detected."],
        }

    largest = max(constraints, key=lambda item: item["value"])
    value = largest["value"]

    if value <= 100:
        recommended = ["O(n^3)", "O(n^2)", "O(n log n)"]
        avoid = []
        notes = ["Small constraints allow heavier brute force."]
    elif value <= 3000:
        recommended = ["O(n^2)", "O(n log n)", "O(n)"]
        avoid = ["O(n^3)"]
        notes = ["Quadratic solutions may pass if constants are reasonable."]
    elif value <= 200000:
        recommended = ["O(n log n)", "O(n)"]
        avoid = ["O(n^2)"]
        notes = ["Look for sorting, greedy, binary search, or data structures."]
    elif value <= 10000000:
        recommended = ["O(n)", "O(n log log n)"]
        avoid = ["O(n log n) with high constants", "O(n^2)"]
        notes = ["Linear or near-linear solutions are usually required."]
    else:
        recommended = ["O(log n)", "O(1)", "O(sqrt n) when acceptable"]
        avoid = ["O(n)", "O(n log n)", "O(n^2)"]
        notes = ["Very large values often require math, binary search, or number theory."]

    return {
        "largest_constraint": largest,
        "recommended": recommended,
        "avoid": avoid,
        "notes": notes,
    }


def readiness_assessment(problem_rating, user_rating, solved_count, known_topics, tags):
    knowledge = knowledge_for_tags(tags)
    required_topics = {topic.lower() for topic in knowledge["topics"]}
    known = {topic.strip().lower() for topic in known_topics if topic.strip()}

    topic_coverage = 1.0
    if required_topics:
        topic_coverage = len(required_topics & known) / len(required_topics)

    rating_gap = problem_rating - user_rating
    gap_score = 1 / (1 + math.exp((rating_gap - 100) / 250))
    solved_bonus = min(max(solved_count, 0), 300) / 300 * 0.15
    topic_score = topic_coverage * 0.35
    success_probability = max(
        0.05,
        min(0.95, gap_score * 0.50 + solved_bonus + topic_score),
    )

    if rating_gap <= 100 and topic_coverage >= 0.65:
        verdict = "Good attempt"
    elif rating_gap <= 300 and topic_coverage >= 0.40:
        verdict = "Challenging but useful"
    else:
        verdict = "Too difficult right now"

    missing_topics = sorted(required_topics - known)

    return {
        "problem_rating": problem_rating,
        "user_rating": user_rating,
        "difficulty_gap": rating_gap,
        "topic_coverage": topic_coverage,
        "success_probability": round(success_probability * 100),
        "verdict": verdict,
        "missing_topics": [topic.title() for topic in missing_topics],
    }


def build_learning_path(current_rating, target_rating, known_topics, tags):
    known = {topic.strip().lower() for topic in known_topics if topic.strip()}
    knowledge = knowledge_for_tags(tags)
    required_topics = [
        topic
        for topic in knowledge["topics"]
        if topic.lower() not in known
    ]

    rating_steps = [
        label
        for rating, label in RATING_PATH
        if current_rating < rating <= target_rating
    ]

    return unique_items(rating_steps + required_topics)


def difficulty_explanation(statement, tags, rating):
    knowledge = knowledge_for_tags(tags)
    constraints = extract_constraints(statement)
    advice = complexity_advice(constraints)

    reasons = []
    tag_count = max(1, len(knowledge["topics"]))

    for topic in knowledge["topics"]:
        reasons.append({
            "factor": topic,
            "weight": round(60 / tag_count),
        })

    if advice["largest_constraint"]:
        reasons.append({
            "factor": "Large constraints",
            "weight": 20 if advice["largest_constraint"]["value"] > 200000 else 10,
        })

    if rating >= 2000:
        reasons.append({"factor": "Advanced implementation/proof", "weight": 20})
    elif rating >= 1600:
        reasons.append({"factor": "Requires combining ideas", "weight": 15})
    else:
        reasons.append({"factor": "Basic implementation pressure", "weight": 10})

    return {
        "rating": rating,
        "knowledge": knowledge,
        "constraints": constraints,
        "complexity_advice": advice,
        "difficulty_factors": reasons,
    }


def format_learning_report(
    statement,
    tags,
    problem_rating,
    user_rating=None,
    solved_count=0,
    known_topics=None,
):
    known_topics = known_topics or []
    knowledge = knowledge_for_tags(tags)
    explanation = difficulty_explanation(statement, tags, problem_rating)

    lines = [
        "Learning analysis",
        f"Problem rating: {problem_rating}",
        "",
        "Topics you should know:",
        *[f"- {topic}" for topic in knowledge["topics"]],
        "",
        "Important concepts:",
        *[f"- {concept}" for concept in knowledge["concepts"]],
        "",
        "Techniques:",
        *[f"- {technique}" for technique in knowledge["techniques"]],
        "",
        "Prerequisites - must know:",
        *[f"- {item}" for item in knowledge["must_know"]],
        "",
        "Prerequisites - should know:",
        *[f"- {item}" for item in knowledge["should_know"]],
        "",
        "Advanced prerequisites:",
        *[f"- {item}" for item in knowledge["advanced"]],
    ]

    if explanation["constraints"]:
        lines.extend([
            "",
            "Constraint analysis:",
        ])
        for constraint in explanation["constraints"][:5]:
            lines.append(
                f"- {constraint['variable']} <= {constraint['raw_value']}"
            )
        lines.extend([
            "Recommended complexity:",
            *[f"- {item}" for item in explanation["complexity_advice"]["recommended"]],
        ])
        if explanation["complexity_advice"]["avoid"]:
            lines.extend([
                "Avoid:",
                *[f"- {item}" for item in explanation["complexity_advice"]["avoid"]],
            ])

    lines.extend([
        "",
        "Why this problem is difficult:",
        *[
            f"- {item['factor']}: {item['weight']}%"
            for item in explanation["difficulty_factors"]
        ],
    ])

    if user_rating is not None:
        readiness = readiness_assessment(
            problem_rating=problem_rating,
            user_rating=user_rating,
            solved_count=solved_count,
            known_topics=known_topics,
            tags=tags,
        )
        path = build_learning_path(
            current_rating=user_rating,
            target_rating=problem_rating,
            known_topics=known_topics,
            tags=tags,
        )

        lines.extend([
            "",
            "Can I solve this?",
            f"- Your rating: {readiness['user_rating']}",
            f"- Difficulty gap: {readiness['difficulty_gap']:+d}",
            f"- Estimated success: {readiness['success_probability']}%",
            f"- Verdict: {readiness['verdict']}",
        ])

        if readiness["missing_topics"]:
            lines.extend([
                "- Missing topics: " + ", ".join(readiness["missing_topics"]),
            ])

        if path:
            lines.extend([
                "",
                "Learning path:",
                *[f"{index}. {step}" for index, step in enumerate(path, start=1)],
            ])

    return "\n".join(lines)
