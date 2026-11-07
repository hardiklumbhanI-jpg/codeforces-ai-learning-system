from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from feature_engineering import add_structural_features


# ---------------------------------------
# PROJECT PATHS
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------
# LOAD SAVED OBJECTS
# ---------------------------------------

model = joblib.load(
    MODELS_DIR / "final_rating_model.pkl"
)

tfidf = joblib.load(
    MODELS_DIR / "final_tfidf_vectorizer.pkl"
)

scaler = joblib.load(
    MODELS_DIR / "structural_scaler.pkl"
)

tag_encoder = joblib.load(
    MODELS_DIR / "tag_encoder.pkl"
)


STRUCTURAL_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "number_count",
    "math_symbol_count",
    "avg_word_length"
]


def parse_tags(tags):
    return [
        tag.strip()
        for tag in str(tags).split(";")
        if tag.strip()
    ]


def predict_rating(statement, tags=""):
    """Predict the Codeforces rating for one problem statement."""
    row = pd.DataFrame({
        "statement": [statement],
        "tags": [tags],
    })

    row = add_structural_features(row)

    text_features = tfidf.transform(row["statement"].fillna(""))

    structural_features = scaler.transform(row[STRUCTURAL_COLUMNS])

    tag_classes = set(tag_encoder.classes_)
    parsed_tags = [
        tag
        for tag in parse_tags(tags)
        if tag in tag_classes
    ]
    tag_features = csr_matrix(
        tag_encoder.transform([parsed_tags])
    )

    features = hstack([
        text_features,
        structural_features,
        tag_features
    ])

    prediction = model.predict(features.toarray())

    return int(round(prediction[0]))


if __name__ == "__main__":
    sample_statement = """
    You are given an array of n integers. Find the maximum possible sum after
    performing operations.
    """

    print(
        "Predicted Rating:",
        predict_rating(sample_statement, tags="greedy;dp")
    )
