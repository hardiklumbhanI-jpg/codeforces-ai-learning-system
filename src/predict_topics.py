from pathlib import Path
import json
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# -----------------------------------
# LOAD MODEL FILES
# -----------------------------------

topic_model = joblib.load(
    MODELS_DIR / "topic_model.pkl"
)

topic_tfidf = joblib.load(
    MODELS_DIR / "topic_tfidf_vectorizer.pkl"
)

topic_encoder = joblib.load(
    MODELS_DIR / "topic_label_encoder.pkl"
)


with open(
    MODELS_DIR / "topic_config.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)


TOPIC_THRESHOLD = config.get("threshold", 0.35)


def predict_topics(statement, threshold=None, fallback_count=3):

    if threshold is None:
        threshold = TOPIC_THRESHOLD

    # Convert statement to TF-IDF
    statement_vector = topic_tfidf.transform(
        [statement]
    )

    # Get probability for every topic
    probabilities = topic_model.predict_proba(
        statement_vector
    )[0]

    all_results = []

    for topic, probability in zip(
        topic_encoder.classes_,
        probabilities
    ):
        all_results.append({
            "topic": topic,
            "confidence": float(probability)
        })

    results = [
        item
        for item in all_results
        if item["confidence"] >= threshold
    ]

    # Highest confidence first
    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    if results:
        return results

    all_results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return all_results[:fallback_count]


if __name__ == "__main__":

    problem = """
    You are given an array of n integers.

    You need to find the minimum number of operations
    required to make the array sorted.

    In one operation you may choose an index and modify
    the value.

    n can be up to 200000.
    """

    predictions = predict_topics(
        problem
    )

    print("\nPredicted Topics")
    print("------------------")

    for item in predictions:

        print(
            item["topic"],
            "→",
            round(
                item["confidence"] * 100,
                2
            ),
            "%"
        )
