"""
Trains and compares baseline rating-prediction models:
Linear Regression, Ridge Regression, and Gradient Boosting Regressor.

Features used: TF-IDF text features + structural features + MultiLabelBinarizer
tag features.

This script is for experimentation/comparison. The winning configuration is
retrained on the full dataset and saved for production use by
src/train_final_rating_model.py.

Run from the project root:

    python src/train_rating_model.py
"""

from pathlib import Path
import sys

import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "cleaned" / "codeforces_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(BASE_DIR / "src"))

from feature_engineering import add_structural_features

STRUCTURAL_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "number_count",
    "math_symbol_count",
    "avg_word_length",
]

RANDOM_STATE = 42


def parse_tags(tags):
    return [
        tag.strip()
        for tag in str(tags).split(";")
        if tag.strip()
    ]


def build_features(train_df, test_df):
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_text = tfidf.fit_transform(train_df["statement"])
    X_test_text = tfidf.transform(test_df["statement"])

    scaler = StandardScaler()
    X_train_struct = scaler.fit_transform(train_df[STRUCTURAL_COLUMNS])
    X_test_struct = scaler.transform(test_df[STRUCTURAL_COLUMNS])

    mlb = MultiLabelBinarizer()
    X_train_tags = csr_matrix(mlb.fit_transform(train_df["tags"].apply(parse_tags)))
    X_test_tags = csr_matrix(mlb.transform(test_df["tags"].apply(parse_tags)))

    X_train = hstack([X_train_text, X_train_struct, X_train_tags]).toarray()
    X_test = hstack([X_test_text, X_test_struct, X_test_tags]).toarray()

    return X_train, X_test


def evaluate(model_name, model, X_train, y_train, X_test, y_test):
    print(f"Training {model_name} ...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.4f}")

    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df["tags"] = df["tags"].fillna("")
    df = add_structural_features(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    X_train, X_test = build_features(train_df, test_df)
    y_train = train_df["rating"]
    y_test = test_df["rating"]

    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=5,
                n_estimators=300,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    results = [
        evaluate(name, model, X_train, y_train, X_test, y_test)
        for name, model in models
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    print(f"\nSaved model comparison to {RESULTS_DIR / 'model_results.csv'}")
    print(results_df)


if __name__ == "__main__":
    main()
