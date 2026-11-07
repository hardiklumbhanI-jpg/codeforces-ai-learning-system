"""Train and evaluate Model 2's future-problem success probability model.

Every split is chronological within each user: earlier interactions train the
model and later interactions validate/test it.  Model 1 predictions are never
loaded and are not present in ``FEATURE_COLUMNS``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "data" / "user_model" / "cache" / "matplotlib"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .build_user_dataset import build_dataset, write_dataset
from .config import DB_PATH, MODEL_DIR, RESULTS_DIR, USER_DATA_DIR, ensure_directories
from .feature_engineering import FEATURE_COLUMNS


RANDOM_STATE = 42


def time_split(data: pd.DataFrame) -> pd.Series:
    labels = pd.Series(index=data.index, dtype="object")
    for _, group in data.groupby("handle", sort=False):
        ordered = group.sort_values("timestamp")
        count = len(ordered)
        train_end = max(1, int(count * 0.70))
        validation_end = max(train_end + 1, int(count * 0.85))
        validation_end = min(validation_end, max(1, count - 1))
        labels.loc[ordered.index[:train_end]] = "train"
        labels.loc[ordered.index[train_end:validation_end]] = "validation"
        labels.loc[ordered.index[validation_end:]] = "test"
    return labels


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if not mask.any():
            continue
        error += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(error if total else 0.0)


def metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    y = y_true.to_numpy(dtype=int)
    both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if y.sum() else None,
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "expected_calibration_error": expected_calibration_error(y, probabilities),
        "positive_rate": float(y.mean()),
        "rows": int(len(y)),
    }


def candidate_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.04, max_depth=2,
            min_samples_leaf=3, random_state=RANDOM_STATE,
        ),
    }


def _plot_calibration(y_true: pd.Series, probabilities: np.ndarray, output: Path) -> None:
    fraction, mean_probability = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(mean_probability, fraction, marker="o", label="Model 2")
    axis.set(xlabel="Predicted acceptance probability", ylabel="Observed acceptance rate",
             title="Model 2 calibration on chronological test data")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train(db_path: Path | str = DB_PATH) -> dict[str, Any]:
    ensure_directories()
    rows = build_dataset(db_path)
    if len(rows) < 20:
        raise ValueError("At least 20 user/problem interactions are required to train Model 2.")
    data = pd.DataFrame(rows).sort_values(["handle", "timestamp"]).reset_index(drop=True)
    if data["accepted"].nunique() < 2:
        raise ValueError("Training data needs both accepted and unsuccessful interactions.")
    data["split"] = time_split(data)
    if not {"train", "validation", "test"}.issubset(set(data["split"])):
        raise ValueError("Each chronological split must contain data; collect more history.")
    write_dataset(rows, USER_DATA_DIR / "user_interactions.csv")

    train_df = data[data["split"] == "train"]
    validation_df = data[data["split"] == "validation"]
    test_df = data[data["split"] == "test"]
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["accepted"].astype(int)
    X_validation, y_validation = validation_df[FEATURE_COLUMNS], validation_df["accepted"].astype(int)

    comparisons = []
    fitted = {}
    for name, model in candidate_models().items():
        try:
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_validation)[:, 1]
            result = metrics(y_validation, probabilities)
            comparisons.append({"model": name, **result})
            fitted[name] = model
        except ValueError as exc:
            comparisons.append({"model": name, "error": str(exc), "log_loss": float("inf")})
    valid = [item for item in comparisons if np.isfinite(item.get("log_loss", float("inf")))]
    if not valid:
        raise ValueError("No candidate model could be trained on the available chronological data.")
    # Probability quality is primary; Brier is the deterministic tie-breaker.
    selected_name = min(valid, key=lambda item: (item["log_loss"], item["brier_score"]))["model"]
    final_model = candidate_models()[selected_name]
    train_validation = data[data["split"].isin(["train", "validation"])]
    final_model.fit(train_validation[FEATURE_COLUMNS], train_validation["accepted"].astype(int))
    test_probabilities = final_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    test_metrics = metrics(test_df["accepted"].astype(int), test_probabilities)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "success_probability_model.pkl")
    if isinstance(final_model, Pipeline) and "scaler" in final_model.named_steps:
        joblib.dump(final_model.named_steps["scaler"], MODEL_DIR / "scaler.pkl")
    else:
        auxiliary_scaler = StandardScaler().fit(train_validation[FEATURE_COLUMNS])
        joblib.dump(auxiliary_scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(
        {"feature_columns": FEATURE_COLUMNS, "topic_encoding": "knowledge_graph topic ids"},
        MODEL_DIR / "encoders.pkl",
    )

    metadata = {
        "model_version": "user-success-v1",
        "selected_model": selected_name,
        "target": "accepted at least once for a future user/problem interaction",
        "split": "chronological 70/15/15 within each handle",
        "feature_columns": FEATURE_COLUMNS,
        "model1_predicted_rating_used": False,
        "official_problem_rating_used": True,
        "validation_comparison": comparisons,
        "test_metrics": test_metrics,
        "calibration": {
            "method": "reliability diagram and expected calibration error",
            "warning": "Calibration reliability depends on representative real interaction volume.",
        },
        "row_counts": data["split"].value_counts().to_dict(),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(comparisons).replace([np.inf, -np.inf], np.nan).to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )
    pd.DataFrame({
        "handle": test_df["handle"], "problem_id": test_df["problem_id"],
        "timestamp": test_df["timestamp"], "actual": test_df["accepted"],
        "predicted_probability": test_probabilities,
    }).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _plot_calibration(test_df["accepted"].astype(int), test_probabilities, RESULTS_DIR / "calibration_curve.png")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 2 success prediction.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    metadata = train(args.db)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
