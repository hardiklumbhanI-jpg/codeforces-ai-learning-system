from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "cleaned" / "codeforces_cleaned.csv"


def load_problem_bank(data_file=DATA_FILE):
    df = pd.read_csv(data_file)
    df["statement"] = df["statement"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["name"] = df["name"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).reset_index(drop=True)
    df["rating"] = df["rating"].astype(int)
    df["search_text"] = (
        df["name"] + " " + df["tags"] + " " + df["statement"]
    )
    return df


def find_problem(df, query):
    query = str(query).strip().lower()

    if not query:
        raise ValueError("Problem query cannot be empty.")

    url_match = df[df["url"].str.lower() == query]
    if not url_match.empty:
        return url_match.index[0]

    exact_name = df[df["name"].str.lower() == query]
    if not exact_name.empty:
        return exact_name.index[0]

    contains_name = df[df["name"].str.lower().str.contains(query, regex=False)]
    if not contains_name.empty:
        return contains_name.index[0]

    raise ValueError(f"No problem found for query: {query}")


def similar_problems(query, count=10, data_file=DATA_FILE):
    df = load_problem_bank(data_file)
    problem_index = find_problem(df, query)

    vectorizer = TfidfVectorizer(
        max_features=12000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(df["search_text"])

    similarities = cosine_similarity(
        matrix[problem_index],
        matrix,
    ).ravel()

    result = df.copy()
    result["similarity"] = similarities
    result = result.drop(index=problem_index)
    result = result.sort_values(
        ["similarity", "rating"],
        ascending=[False, True],
    ).head(count)

    return result[
        ["name", "rating", "tags", "url", "similarity"]
    ]


def format_similar_problems(problems):
    lines = []

    for index, row in enumerate(problems.itertuples(), start=1):
        percentage = round(row.similarity * 100)
        lines.append(
            f"{index}. {row.name} | {row.rating} | {percentage}% similar | {row.url}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print(
        format_similar_problems(
            similar_problems("Odd-Even Subsequence", count=5)
        )
    )
