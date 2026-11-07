from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "cleaned" / "codeforces_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def parse_tags(tags):
    return [
        tag.strip()
        for tag in str(tags).split(";")
        if tag.strip()
    ]


def evaluate_predictions(y_true, y_pred):
    return {
        "Micro_F1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "Macro_F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "Hamming_Loss": hamming_loss(y_true, y_pred),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df["statement"] = df["statement"].fillna("")
    df["tags"] = df["tags"].fillna("")

    tag_lists = df["tags"].apply(parse_tags)

    topic_encoder = MultiLabelBinarizer()
    y_topics = topic_encoder.fit_transform(tag_lists)

    X_temp, X_test, y_temp, y_test = train_test_split(
        df["statement"],
        y_topics,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,
        random_state=RANDOM_STATE,
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )
    )

    print("Training topic model...")
    model.fit(X_train_vec, y_train)

    val_probabilities = model.predict_proba(X_val_vec)

    threshold_results = []
    for threshold in THRESHOLDS:
        predictions = (val_probabilities >= threshold).astype(int)
        threshold_results.append({
            "Threshold": threshold,
            **evaluate_predictions(y_val, predictions),
        })

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(
        RESULTS_DIR / "topic_threshold_comparison.csv",
        index=False,
    )

    best_threshold_row = (
        threshold_df
        .sort_values("Micro_F1", ascending=False)
        .iloc[0]
    )
    best_threshold = float(best_threshold_row["Threshold"])

    test_probabilities = model.predict_proba(X_test_vec)
    test_predictions = (test_probabilities >= best_threshold).astype(int)
    final_metrics = evaluate_predictions(y_test, test_predictions)

    pd.DataFrame([{
        "Model": "Logistic Regression",
        "Threshold": best_threshold,
        **final_metrics,
    }]).to_csv(
        RESULTS_DIR / "topic_model_comparison.csv",
        index=False,
    )

    pd.DataFrame([{
        "Metric": "Best threshold",
        "Value": best_threshold,
    }, *[
        {"Metric": metric, "Value": value}
        for metric, value in final_metrics.items()
    ]]).to_csv(
        RESULTS_DIR / "topic_experiment_02_threshold_tuning.csv",
        index=False,
    )

    joblib.dump(model, MODELS_DIR / "topic_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "topic_tfidf_vectorizer.pkl")
    joblib.dump(topic_encoder, MODELS_DIR / "topic_label_encoder.pkl")

    with open(MODELS_DIR / "topic_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_threshold,
            "model": "OneVsRestClassifier(LogisticRegression)",
            "num_topics": len(topic_encoder.classes_),
            "topics": list(topic_encoder.classes_),
        }, f, indent=2)

    print("Topic model saved.")
    print("Best threshold:", best_threshold)
    print("Final metrics:", final_metrics)


if __name__ == "__main__":
    main()
