"""
Data cleaning script for the Codeforces Learning Assistant.

Loads the raw scraped Codeforces problems dataset, removes rows with
missing/invalid statements or ratings, normalizes the tags column, and
saves the result to data/cleaned/codeforces_cleaned.csv.

Run from the project root:

    python src/data_cleaning.py
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "codeforces_raw.csv"
CLEANED_FILE = BASE_DIR / "data" / "cleaned" / "codeforces_cleaned.csv"

REQUIRED_COLUMNS = [
    "contestId",
    "index",
    "name",
    "rating",
    "tags",
    "url",
    "statement",
]

# A statement shorter than this is almost certainly a scrape artifact
# rather than a real problem statement.
MIN_STATEMENT_LENGTH = 30


def normalize_tags(tags):
    """Lowercase, strip, de-duplicate, and re-join semicolon separated tags."""
    if pd.isna(tags):
        return ""

    seen = []
    for tag in str(tags).split(";"):
        cleaned = tag.strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)

    return ";".join(seen)


def load_raw(raw_file=RAW_FILE):
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_file}. "
            "Place the scraped Codeforces CSV there before running this script."
        )

    df = pd.read_csv(raw_file)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )

    return df


def clean_dataframe(df):
    df = df.copy()

    # ---------------------------------
    # 1. Remove rows with missing/invalid statements
    # ---------------------------------
    df["statement"] = df["statement"].astype(str)
    df = df[df["statement"].str.strip().str.lower() != "nan"]
    df = df[df["statement"].str.len() >= MIN_STATEMENT_LENGTH]

    # ---------------------------------
    # 2. Remove rows with missing/invalid ratings
    # ---------------------------------
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    df = df[(df["rating"] >= 800) & (df["rating"] <= 3500)]

    # ---------------------------------
    # 3. Clean tags
    # ---------------------------------
    df["tags"] = df["tags"].apply(normalize_tags)

    # ---------------------------------
    # 4. Basic hygiene
    # ---------------------------------
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=["url"])
    df = df.drop_duplicates(subset=["statement"])

    df = df.reset_index(drop=True)

    return df[REQUIRED_COLUMNS]


def main():
    print(f"Loading raw dataset from {RAW_FILE} ...")
    raw_df = load_raw()
    print(f"Raw rows: {len(raw_df)}")

    cleaned_df = clean_dataframe(raw_df)
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Rows removed: {len(raw_df) - len(cleaned_df)}")

    CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEANED_FILE, index=False)
    print(f"Saved cleaned dataset to {CLEANED_FILE}")


if __name__ == "__main__":
    main()
