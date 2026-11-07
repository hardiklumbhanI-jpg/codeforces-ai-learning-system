"""
Codeforces Learning Assistant - Streamlit UI.

Run from the project root with:

    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from predict_rating import predict_rating  # noqa: E402
from predict_topics import predict_topics  # noqa: E402
from problem_similarity import similar_problems  # noqa: E402
from recommend_problems import recommend_problems, recommend_next_problem  # noqa: E402
from learning_analysis import (  # noqa: E402
    knowledge_for_tags,
    extract_constraints,
    complexity_advice,
    difficulty_explanation,
    readiness_assessment,
    build_learning_path,
)


st.set_page_config(
    page_title="Codeforces Learning Assistant",
    page_icon="\U0001F9E9",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tags_to_semicolon(tags_text):
    """Normalize a comma/semicolon separated tags string to 'tag1;tag2'."""
    if not tags_text:
        return ""
    separators = [";", ","]
    text = tags_text
    for sep in separators[1:]:
        text = text.replace(sep, ";")
    return ";".join(
        t.strip() for t in text.split(";") if t.strip()
    )


def show_error(exc):
    st.error(f"Something went wrong: {exc}")


def knowledge_section(tags_semicolon):
    knowledge = knowledge_for_tags(tags_semicolon)

    if not any(knowledge.values()):
        st.info("No tags to analyze yet.")
        return

    st.markdown("**Topics**")
    st.write(", ".join(knowledge["topics"]) or "None")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Concepts**")
        for item in knowledge["concepts"]:
            st.markdown(f"- {item}")
        st.markdown("**Techniques**")
        for item in knowledge["techniques"]:
            st.markdown(f"- {item}")

    with col2:
        st.markdown("**Prerequisites**")
        st.markdown("*Must know*")
        for item in knowledge["must_know"]:
            st.markdown(f"- {item}")
        st.markdown("*Should know*")
        for item in knowledge["should_know"]:
            st.markdown(f"- {item}")
        st.markdown("*Advanced*")
        for item in knowledge["advanced"]:
            st.markdown(f"- {item}")

    return knowledge


def constraints_section(statement):
    constraints = extract_constraints(statement)
    advice = complexity_advice(constraints)

    if constraints:
        rows = [
            {"Variable": c["variable"], "Constraint": f"<= {c['raw_value']}"}
            for c in constraints
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No explicit numeric constraints were detected in the statement.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommended complexity**")
        st.markdown(", ".join(advice["recommended"]))
    with col2:
        if advice["avoid"]:
            st.markdown("**Avoid**")
            st.markdown(", ".join(advice["avoid"]))

    for note in advice["notes"]:
        st.caption(note)


def difficulty_section(statement, tags_semicolon, rating):
    explanation = difficulty_explanation(statement, tags_semicolon, rating)
    st.markdown("**Why this problem is difficult**")
    for factor in explanation["difficulty_factors"]:
        st.markdown(f"{factor['factor']}")
        st.progress(min(max(factor["weight"], 0), 100) / 100)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Codeforces Learning Assistant")
st.caption(
    "Predict problem difficulty, discover topics, and get a personalized "
    "path to your next Codeforces problem."
)

tab_rating, tab_topics, tab_analysis, tab_readiness, tab_recs, tab_path = st.tabs(
    [
        "Rating Prediction",
        "Topic Prediction",
        "Learning Analysis",
        "Readiness Check",
        "Recommendations",
        "Learning Path",
    ]
)


# ---------------------------------------------------------------------------
# Tab 1: Rating Prediction
# ---------------------------------------------------------------------------

with tab_rating:
    st.header("Rating Prediction")
    st.write("Paste a problem statement to predict its Codeforces rating.")

    statement = st.text_area(
        "Problem statement",
        height=220,
        key="rating_statement",
        placeholder="You are given an array of n integers. Find the maximum possible sum...",
    )
    tags_text = st.text_input(
        "Tags (optional, comma or semicolon separated)",
        key="rating_tags",
        placeholder="dp, greedy",
    )

    if st.button("Predict rating", type="primary"):
        if not statement.strip():
            st.warning("Please enter a problem statement.")
        else:
            try:
                tags_semicolon = tags_to_semicolon(tags_text)
                rating = predict_rating(statement, tags=tags_semicolon)
                st.metric("Predicted Codeforces rating", rating)
            except Exception as exc:  # noqa: BLE001
                show_error(exc)


# ---------------------------------------------------------------------------
# Tab 2: Topic Prediction
# ---------------------------------------------------------------------------

with tab_topics:
    st.header("Topic Prediction")
    st.write("Paste a problem statement to predict its likely Codeforces tags.")

    topic_statement = st.text_area(
        "Problem statement",
        height=220,
        key="topic_statement",
        placeholder="You are given an array of n integers...",
    )

    if st.button("Predict topics", type="primary"):
        if not topic_statement.strip():
            st.warning("Please enter a problem statement.")
        else:
            try:
                predictions = predict_topics(topic_statement)
                for item in predictions:
                    st.markdown(f"**{item['topic']}** — {round(item['confidence'] * 100, 1)}%")
                    st.progress(min(max(item["confidence"], 0.0), 1.0))
            except Exception as exc:  # noqa: BLE001
                show_error(exc)


# ---------------------------------------------------------------------------
# Tab 3: Learning Analysis
# ---------------------------------------------------------------------------

with tab_analysis:
    st.header("Learning Analysis")
    st.write(
        "See the topics, concepts, prerequisites, constraints, and difficulty "
        "breakdown for a problem."
    )

    analysis_statement = st.text_area(
        "Problem statement",
        height=200,
        key="analysis_statement",
    )
    analysis_tags = st.text_input(
        "Tags (optional — predicted automatically if left blank)",
        key="analysis_tags",
    )

    if st.button("Analyze", type="primary"):
        if not analysis_statement.strip():
            st.warning("Please enter a problem statement.")
        else:
            try:
                tags_semicolon = tags_to_semicolon(analysis_tags)
                if not tags_semicolon:
                    predicted = predict_topics(analysis_statement)
                    tags_semicolon = ";".join(item["topic"] for item in predicted)
                    st.caption(f"Using predicted tags: {tags_semicolon or 'none detected'}")

                rating = predict_rating(analysis_statement, tags=tags_semicolon)
                st.metric("Predicted rating", rating)

                st.subheader("Topics, Concepts & Prerequisites")
                knowledge_section(tags_semicolon)

                st.subheader("Constraint Analysis")
                constraints_section(analysis_statement)

                st.subheader("Difficulty Breakdown")
                difficulty_section(analysis_statement, tags_semicolon, rating)
            except Exception as exc:  # noqa: BLE001
                show_error(exc)


# ---------------------------------------------------------------------------
# Tab 4: Readiness Check
# ---------------------------------------------------------------------------

with tab_readiness:
    st.header("Readiness Check")
    st.write("Find out if you're ready to attempt a problem.")

    col1, col2 = st.columns(2)
    with col1:
        current_rating = st.number_input(
            "Your current rating", min_value=0, max_value=4000, value=1200, step=50
        )
        solved_count = st.number_input(
            "Problems solved so far", min_value=0, max_value=100000, value=100, step=10
        )
    with col2:
        known_topics_text = st.text_input(
            "Known topics (comma or semicolon separated)",
            placeholder="Arrays, Sorting, Greedy Algorithms",
        )

    readiness_statement = st.text_area(
        "Problem statement to check",
        height=180,
        key="readiness_statement",
    )
    readiness_tags = st.text_input(
        "Tags (optional — predicted automatically if left blank)",
        key="readiness_tags",
    )

    if st.button("Check readiness", type="primary"):
        if not readiness_statement.strip():
            st.warning("Please enter a problem statement.")
        else:
            try:
                tags_semicolon = tags_to_semicolon(readiness_tags)
                if not tags_semicolon:
                    predicted = predict_topics(readiness_statement)
                    tags_semicolon = ";".join(item["topic"] for item in predicted)

                problem_rating = predict_rating(readiness_statement, tags=tags_semicolon)
                known_topics = [
                    t.strip()
                    for t in tags_to_semicolon(known_topics_text).split(";")
                    if t.strip()
                ]

                readiness = readiness_assessment(
                    problem_rating=problem_rating,
                    user_rating=int(current_rating),
                    solved_count=int(solved_count),
                    known_topics=known_topics,
                    tags=tags_semicolon,
                )

                st.metric("Predicted problem rating", problem_rating)
                st.metric("Difficulty gap", f"{readiness['difficulty_gap']:+d}")

                st.markdown("**Estimated success probability**")
                st.progress(readiness["success_probability"] / 100)
                st.caption(f"{readiness['success_probability']}%")

                verdict_color = {
                    "Good attempt": "success",
                    "Challenging but useful": "warning",
                    "Too difficult right now": "error",
                }.get(readiness["verdict"], "info")
                getattr(st, verdict_color)(f"Verdict: {readiness['verdict']}")

                if readiness["missing_topics"]:
                    st.markdown("**Missing topics**")
                    st.markdown(", ".join(readiness["missing_topics"]))
                else:
                    st.markdown("You already know all the required topics.")
            except Exception as exc:  # noqa: BLE001
                show_error(exc)


# ---------------------------------------------------------------------------
# Tab 5: Recommendations
# ---------------------------------------------------------------------------

with tab_recs:
    st.header("Recommendations")

    rec_type = st.radio(
        "What kind of recommendation do you want?",
        ["Practice problems", "Similar problems", "Next problem after a solved one"],
        horizontal=True,
    )

    if rec_type == "Practice problems":
        col1, col2, col3 = st.columns(3)
        with col1:
            target_rating = st.number_input(
                "Target rating", min_value=800, max_value=3500, value=1600, step=100
            )
        with col2:
            weak_tags_text = st.text_input(
                "Focus tags (optional)", placeholder="dp, greedy"
            )
        with col3:
            count = st.slider("Number of results", 1, 20, 10)

        if st.button("Recommend practice problems", type="primary"):
            try:
                weak_tags = [
                    t.strip()
                    for t in tags_to_semicolon(weak_tags_text).split(";")
                    if t.strip()
                ]
                recs = recommend_problems(
                    target_rating=int(target_rating),
                    weak_tags=weak_tags,
                    count=int(count),
                )
                st.dataframe(
                    recs[["name", "rating", "tags", "url"]],
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                show_error(exc)

    elif rec_type == "Similar problems":
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Problem name or URL", placeholder="Odd-Even Subsequence"
            )
        with col2:
            similar_count = st.slider("Number of results", 1, 20, 10, key="similar_count")

        if st.button("Find similar problems", type="primary"):
            if not query.strip():
                st.warning("Please enter a problem name or URL.")
            else:
                try:
                    results = similar_problems(query, count=int(similar_count))
                    display = results.copy()
                    display["similarity"] = (display["similarity"] * 100).round(1).astype(str) + "%"
                    st.dataframe(
                        display[["name", "rating", "similarity", "tags", "url"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    show_error(exc)

    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            solved_query = st.text_input(
                "Solved problem name or URL", placeholder="Odd-Even Subsequence"
            )
        with col2:
            next_count = st.slider("Number of results", 1, 20, 5, key="next_count")

        if st.button("Recommend next problem", type="primary"):
            if not solved_query.strip():
                st.warning("Please enter a solved problem's name or URL.")
            else:
                try:
                    solved_problem, recs = recommend_next_problem(
                        solved_query, count=int(next_count)
                    )
                    st.success(
                        f"After solving: {solved_problem['name']} "
                        f"({int(solved_problem['rating'])})"
                    )
                    st.dataframe(
                        recs[["name", "rating", "tags", "url"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    show_error(exc)


# ---------------------------------------------------------------------------
# Tab 6: Learning Path
# ---------------------------------------------------------------------------

with tab_path:
    st.header("Learning Path")
    st.write("Get a step-by-step path from your current level to a target problem.")

    col1, col2 = st.columns(2)
    with col1:
        path_current_rating = st.number_input(
            "Current rating", min_value=0, max_value=4000, value=1200, step=50, key="path_current"
        )
        path_known_topics_text = st.text_input(
            "Known topics (comma or semicolon separated)",
            key="path_known_topics",
            placeholder="Arrays, Sorting",
        )
    with col2:
        target_mode = st.radio(
            "Target",
            ["Target rating", "Target problem statement"],
            horizontal=True,
        )

    target_rating_value = None
    path_tags = ""
    path_statement = ""

    if target_mode == "Target rating":
        target_rating_value = st.number_input(
            "Target rating", min_value=800, max_value=3500, value=1800, step=100, key="path_target_rating"
        )
        path_tags = st.text_input("Tags for this target (optional)", key="path_tags_only")
    else:
        path_statement = st.text_area("Target problem statement", height=180, key="path_statement")
        path_tags = st.text_input("Tags (optional — predicted automatically if left blank)", key="path_tags")

    if st.button("Build learning path", type="primary"):
        try:
            tags_semicolon = tags_to_semicolon(path_tags)

            if target_mode == "Target problem statement":
                if not path_statement.strip():
                    st.warning("Please enter a target problem statement.")
                    st.stop()
                if not tags_semicolon:
                    predicted = predict_topics(path_statement)
                    tags_semicolon = ";".join(item["topic"] for item in predicted)
                target_rating_value = predict_rating(path_statement, tags=tags_semicolon)
                st.metric("Predicted target rating", target_rating_value)

            known_topics = [
                t.strip()
                for t in tags_to_semicolon(path_known_topics_text).split(";")
                if t.strip()
            ]

            path = build_learning_path(
                current_rating=int(path_current_rating),
                target_rating=int(target_rating_value),
                known_topics=known_topics,
                tags=tags_semicolon,
            )

            if not path:
                st.info("You're already prepared — no additional steps needed.")
            else:
                for index, step in enumerate(path, start=1):
                    st.markdown(f"{index}. {step}")
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
