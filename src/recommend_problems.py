from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "cleaned" / "codeforces_cleaned.csv"


def parse_tags(tags):
    return {
        tag.strip().lower()
        for tag in str(tags).split(";")
        if tag.strip()
    }


def load_problem_bank(data_file=DATA_FILE):
    df = pd.read_csv(data_file)
    df["tags"] = df["tags"].fillna("")
    df["name"] = df["name"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    return df


def recommend_problems(
    target_rating,
    weak_tags=None,
    count=10,
    tolerance=300,
    data_file=DATA_FILE,
):
    weak_tags = parse_tags(";".join(weak_tags or []))
    df = load_problem_bank(data_file)

    candidates = df[
        df["rating"].between(
            target_rating - tolerance,
            target_rating + tolerance,
        )
    ].copy()

    if candidates.empty:
        candidates = df.copy()

    candidates["rating_gap"] = (candidates["rating"] - target_rating).abs()
    candidates["tag_overlap"] = candidates["tags"].apply(
        lambda tags: len(parse_tags(tags) & weak_tags)
    )

    if weak_tags:
        tagged_candidates = candidates[candidates["tag_overlap"] > 0]
        if not tagged_candidates.empty:
            candidates = tagged_candidates

    recommendations = candidates.sort_values(
        ["tag_overlap", "rating_gap", "rating", "name"],
        ascending=[False, True, True, True],
    ).head(count)

    return recommendations[
        ["name", "rating", "tags", "url", "rating_gap", "tag_overlap"]
    ]


def format_recommendations(recommendations):
    lines = []

    for index, row in enumerate(recommendations.itertuples(), start=1):
        lines.append(
            f"{index}. {row.name} | {row.rating} | {row.tags} | {row.url}"
        )

    return "\n".join(lines)


def find_problem(df, query):
    query = str(query).strip().lower()

    if not query:
        raise ValueError("Problem query cannot be empty.")

    url_match = df[df["url"].str.lower() == query]
    if not url_match.empty:
        return url_match.iloc[0]

    exact_name = df[df["name"].str.lower() == query]
    if not exact_name.empty:
        return exact_name.iloc[0]

    contains_name = df[df["name"].str.lower().str.contains(query, regex=False)]
    if not contains_name.empty:
        return contains_name.iloc[0]

    raise ValueError(f"No problem found for query: {query}")


def recommend_next_problem(
    solved_problem_query,
    count=5,
    step=100,
    tolerance=200,
    data_file=DATA_FILE,
):
    df = load_problem_bank(data_file)
    solved_problem = find_problem(df, solved_problem_query)
    solved_tags = parse_tags(solved_problem["tags"])
    solved_rating = int(solved_problem["rating"])
    target_rating = int(solved_problem["rating"]) + step

    candidates = df[
        df["rating"].between(
            target_rating - tolerance,
            target_rating + tolerance,
        )
    ].copy()
    candidates = candidates[
        (candidates["rating"] >= solved_rating)
        & (candidates["url"] != solved_problem["url"])
    ]

    if candidates.empty:
        candidates = df[
            (df["rating"] >= solved_rating)
            & (df["url"] != solved_problem["url"])
        ].copy()

    candidates["rating_gap"] = (candidates["rating"] - target_rating).abs()
    candidates["tag_overlap"] = candidates["tags"].apply(
        lambda tags: len(parse_tags(tags) & solved_tags)
    )

    tagged_candidates = candidates[candidates["tag_overlap"] > 0]
    if not tagged_candidates.empty:
        candidates = tagged_candidates

    recommendations = candidates.sort_values(
        ["rating_gap", "tag_overlap", "rating", "name"],
        ascending=[True, False, True, True],
    ).head(count)
    recommendations = recommendations[
        ["name", "rating", "tags", "url", "rating_gap", "tag_overlap"]
    ]

    return solved_problem, recommendations


if __name__ == "__main__":
    recs = recommend_problems(
        target_rating=1800,
        weak_tags=["dp", "greedy"],
        count=5,
    )
    print(format_recommendations(recs))
