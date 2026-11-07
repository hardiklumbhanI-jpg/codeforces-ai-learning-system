from pathlib import Path
import sys

import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "cleaned" / "codeforces_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR / "src"))

from feature_engineering import add_structural_features

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


# ------------------------------------
# 1. LOAD DATA
# ------------------------------------

df = pd.read_csv(DATA_FILE)

df["tags"] = df["tags"].fillna("")

df = add_structural_features(df)


# ------------------------------------
# 2. TRAIN TEST SPLIT
# ------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)


# ------------------------------------
# 3. TEXT FEATURES
# ------------------------------------

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_text = tfidf.fit_transform(
    train_df["statement"]
)


# ------------------------------------
# 4. STRUCTURAL FEATURES
# ------------------------------------

scaler = StandardScaler()

X_train_struct = scaler.fit_transform(
    train_df[STRUCTURAL_COLUMNS]
)


# ------------------------------------
# 5. TAG FEATURES
# ------------------------------------

train_tags = (
    train_df["tags"]
    .apply(parse_tags)
)

mlb = MultiLabelBinarizer()

X_train_tags = mlb.fit_transform(
    train_tags
)

X_train_tags = csr_matrix(
    X_train_tags
)


# ------------------------------------
# 6. COMBINE FEATURES
# ------------------------------------

X_train = hstack([
    X_train_text,
    X_train_struct,
    X_train_tags
])

X_train = X_train.toarray()

y_train = train_df["rating"]


# ------------------------------------
# 7. FINAL MODEL
# ------------------------------------

model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=5,
    n_estimators=300,
    random_state=42
)

print("Training final model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ------------------------------------
# 8. SAVE EVERYTHING
# ------------------------------------

MODELS_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODELS_DIR / "final_rating_model.pkl"
)

joblib.dump(
    tfidf,
    MODELS_DIR / "final_tfidf_vectorizer.pkl"
)

joblib.dump(
    scaler,
    MODELS_DIR / "structural_scaler.pkl"
)

joblib.dump(
    mlb,
    MODELS_DIR / "tag_encoder.pkl"
)

print("All model files saved.")
