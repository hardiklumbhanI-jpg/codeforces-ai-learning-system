# Codeforces Problem Difficulty and Personalized Learning System

This project contains two independently trainable and independently testable pipelines:

- **Model 1 — Problem Difficulty Prediction:** the preserved supplied model predicts a Codeforces rating from a problem statement, TF-IDF text, six structural features, and Codeforces tags.
- **Model 2 — User Skill and Personalized Learning:** a new user-history pipeline infers topic skill, predicts acceptance probability, measures skill gaps, estimates personalized difficulty/readiness, and recommends a progression.

The normal experience is terminal-first:

```powershell
python main.py
```

The existing Streamlit application is retained, but it is optional and is not required by either terminal pipeline.

## Why the models are separate

```text
MODEL 1: problem statement ──> TF-IDF/structure/tags ──> predicted rating/range
                                                        │
                                                        │ presentation only
                                                        ▼
MODEL 2: user/API history + official problem metadata ──> success/skill/gaps/path
```

Model 2 never imports Model 1, never reads Model 1 artifacts, and never uses Model 1's predicted rating as a feature. Its `problem_rating` feature is always the official Codeforces rating. `main.py` calls the pipelines separately and combines their already-computed results only for display/explanation.

The original Model 1 source, notebooks, data, results, plots, and root `models/*.pkl` artifacts are preserved. Byte-identical compatibility copies are also placed in `models/problem_difficulty/`. See `docs/ARCHITECTURE_AUDIT.md` for the pre-implementation inspection and production/experimental classification.

## Installation

Python 3.11 or 3.12 is recommended. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Docker, database server, website, API credential, or LLM key is required. SQLite is part of Python.

## Quick start and offline acceptance flow

Interactive:

```powershell
python main.py
```

Reproducible bundled offline fixture:

```powershell
python main.py --offline --non-interactive --handle sample_user --problem-index 45 --result WRONG_ANSWER
```

`sample_user` is a reserved, explicitly synthetic demonstration profile, not a real Codeforces account and not evidence of production model quality. It always selects the local fixture even if a public account happens to share that handle. The command displays the statement before revealing official rating/tags, then reports Model 1 rating information, required-topic relevance, inferred skills/confidence, gaps, Model 2 probability, personalized difficulty/readiness, an explanation, similar problems, and personalized next problems. The observed verdict is persisted and the skill estimator is recomputed from all evidence.

An optional solution path supplies basic, non-judging metadata:

```powershell
python main.py --offline --handle sample_user --problem-index 45 --solution solution.cpp
```

The program does not submit code or impersonate an online judge. If no judge result is available, enter `ACCEPTED`, `WRONG_ANSWER`, `TLE`, `RUNTIME_ERROR`, `COMPILATION_ERROR`, or `MEMORY_LIMIT_EXCEEDED`.

## Example terminal result

```text
PROBLEM ANALYSIS
Problem Index: #45
Official Rating: ...
AI Predicted Rating (Model 1): ...
Model 1 Reliability: ...
Prediction Range: ...

PERSONALIZED ANALYSIS (MODEL 2)
Your Current Rating: ...
Your Estimated Difficulty: ...
Acceptance Probability: ...
Probability Confidence: ...
Readiness: .../100 (...)
```

Values are calculated from the shipped artifacts/database at runtime; this README intentionally does not present invented example scores.

## Model 1: preserved problem-difficulty pipeline

Production flow:

```text
statement -> 10,000 word/bigram TF-IDF features
statement -> char/word/sentence/number/math-symbol/average-word-length features
official/manual tags -> 37-way multi-hot encoding
combined dense features -> GradientBoostingRegressor -> predicted Codeforces rating
```

Run Model 1 independently:

```powershell
python -m src.problem_model.predict --statement "Given an array, find its maximum." --tags "implementation"
```

Train the preserved production Model 1 independently:

```powershell
python src/train_final_rating_model.py
```

The wrapper adds uncertainty metadata from the preserved held-out `results/prediction_samples.csv`: reliability is one minus normalized held-out MAE, and the range uses the empirical 80th-percentile absolute residual. This makes uncertainty data-derived and also exposes the supplied model's limitations.

## Model 2: data collection and SQLite

The public [Codeforces API](https://codeforces.com/apiHelp) requires no credentials for the endpoints used here. The collector calls:

- `user.info` for profile/rank/rating fields;
- `user.rating` for chronological contest/rating changes;
- `user.status` for submissions/verdicts/languages/timing;
- optionally `problemset.problems` for the public problem master.

Codeforces documents a maximum of one call per two seconds; the client waits 2.1 seconds, applies a timeout, stores raw response caches under `data/user_model/cache`, and falls back to cache/sample data on failure.

Collect or update one real handle:

```powershell
python -m src.user_model.collect_user_data tourist
```

Also refresh the full problemset metadata (large API response):

```powershell
python -m src.user_model.collect_user_data tourist --refresh-problemset
```

Force local-only behavior:

```powershell
python -m src.user_model.collect_user_data sample_user --offline
```

The database is `data/user_model/codeforces_users.db`. Primary/unique keys make collection resumable and prevent duplicate users, problems, tags, submissions, and rating events. Composite indexes cover user/time, problem/time, verdict, rating, tag, and prediction queries. Missing API fields remain nullable.

Tables:

`users`, `problems`, `problem_tags`, `submissions`, `rating_history`, `user_problem_history`, `attempt_history`, `topic_performance`, `skill_profiles`, `recommendations`, and `model_predictions`.

Build a CSV research export:

```powershell
python -m src.user_model.build_user_dataset
```

## Topic taxonomy and knowledge graph

`data/user_model/knowledge_graph.json` contains extensible nodes and two edge types:

- `is_a` builds hierarchy such as 1D DP -> Dynamic Programming -> Algorithms.
- `prerequisite` points from prerequisite to dependent skill, such as Queue -> BFS, Heap/Priority Queue -> Dijkstra, DP State -> State Transition -> Memoization, and Arrays/Trees/Recursion/Range Queries -> Segment Tree.

Official tags give strong topic evidence. Transparent statement-keyword matches and prerequisites give weaker evidence. Required-topic percentages always mean **estimated relevance/importance for solving the problem**. They do not mean that a fixed percentage of knowledge is objectively required.

## Skill estimator formulas

For every topic, Model 2 fits a regularized one-parameter item-response model:

```text
P(solve | topic ability a, official rating d) = 1 / (1 + 10^((d-a)/400))
```

This means success on a hard problem provides stronger evidence than success on an easy problem. The optimizer uses a user's rating as a weak prior, not as a target label.

Each observation weight is:

```text
topic_relevance * [0.35 + 0.65 * 2^(-age_days / 90)]
```

Long-term evidence therefore never disappears, while recent evidence receives more weight. Topic skill maps fitted ability from the 800–2800 research scale into 0–100.

Controlled inactivity decay begins only after a 90-day grace period:

```text
retention = 0.65 + 0.35 * 2^(-(inactive_days - 90) / 365)
```

Retention is floored at 65%. All parameters are in `src/user_model/config.py`.

Confidence is independent of success:

```text
confidence = 1 - exp(-effective_evidence / 8)
```

Two observations therefore cannot receive the same confidence as dozens. Per-topic records also contain attempted/solved relevance mass, recent and long-term performance, average/max solved rating, WA/TLE rates, average attempts, consistency, last practice, and evidence metadata. `topic_performance` stores smoothed 200-rating-bucket curves.

After a terminal result, the complete topic posterior is recomputed. There is no fixed `+5%` reward: difficulty, topic relevance, verdict, evidence count, recency, and confidence determine the change.

## Model 2 success prediction and leakage control

Target: whether the user eventually accepted a newly encountered problem (`accepted=1`) or did not (`accepted=0`). Features cover rating history, activity, earlier solved/attempted counts, recent/overall rates, prior topic performance/confidence/recency, same-rating performance, verdict patterns, attempts, consistency, official problem rating, rating gap, and official topic metadata.

Feature generation walks each handle's history chronologically. The current target and all future interactions are invisible. Splits are chronological **within each handle**:

```text
earliest 70% -> train
next 15%     -> validation/model selection
latest 15%   -> final test
```

Compared candidates:

- scaled, class-balanced Logistic Regression (interpretable baseline);
- Random Forest;
- Gradient Boosting.

Validation log loss selects the final model, with Brier score as tie-breaker. Evaluation saves Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, Brier Score, expected calibration error, test probabilities, and a calibration curve.

Train both Model 2 artifacts:

```powershell
python -m src.user_model.train_skill_model
python -m src.user_model.train_success_model
```

Outputs remain separate:

```text
models/user_model/success_probability_model.pkl
models/user_model/skill_model.pkl
models/user_model/scaler.pkl
models/user_model/encoders.pkl
models/user_model/metadata.json
results/user_model/metrics.json
results/user_model/model_comparison.csv
results/user_model/test_predictions.csv
results/user_model/calibration_curve.png
```

If no trained success artifact exists, Model 2 uses a documented offline baseline combining rating-curve probability (50%), topic skill (25%), recent performance (15%), and same-rating history (10%). The application labels this fallback explicitly.

## Skill gap, personalized difficulty, and readiness

For each required topic:

```text
gap = max(0, required_relevance - inferred_user_skill)
Strong Match: gap <= 8 points
Moderate Gap: 8 < gap <= 25 points
Major Gap: gap > 25 points
```

Personalized difficulty is not `official rating + arbitrary constant`. First, Model 2 acceptance probability is inverted through the same Elo curve:

```text
probability_equivalent_rating = user_rating + 400*log10((1-p)/p)
personalized_rating = (1-confidence)*official + confidence*probability_equivalent
```

The result is rounded to 25 rating points. Low-evidence predictions shrink toward official metadata.

Readiness is:

```text
100 * (0.45*success_probability
     + 0.20*topic_match
     + 0.15*official_rating_match
     + 0.10*recent_performance
     + 0.10*prediction_confidence)
```

## Recommendation engine

Two result categories are always separate:

1. **Similar Problems:** statement/name/tag TF-IDF cosine similarity.
2. **Personalized Next Problems:** unsolved problems ranked by measured gaps, official topic overlap, target overlap, and a controlled step from the user's rating.

Modes implemented in `recommendation_engine.py`:

- Skill Gap Mode
- Similarity Mode
- Difficulty Progression Mode
- Topic Practice Mode
- Revision Mode
- Mixed Topic Mode

Every personalized item includes name, stable problem ID, local display index, official rating, tags, expected difficulty, and “Why this problem?”. Very hard targets are approached through a capped progression window rather than immediately repeated.

## Optional LLM explanation

All numbers are produced before the LLM layer. The optional layer receives structured JSON and may only explain it. Without credentials, or on any API/timeout/error, deterministic explanation is automatic.

Do not put secrets in the repository. In PowerShell for the current terminal:

```powershell
$env:OPENAI_API_KEY = "your-project-api-key"
$env:OPENAI_MODEL = "a-model-enabled-for-your-project"
python main.py
```

In Command Prompt:

```bat
set OPENAI_API_KEY=your-project-api-key
set OPENAI_MODEL=a-model-enabled-for-your-project
python main.py
```

The implementation uses `POST /v1/responses` with `store: false`, a structured JSON input, and an explanation-only instruction. See the official [OpenAI Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create). `.env.example` is documentation only; environment variables are intentionally not auto-loaded, which keeps dependencies and secret handling simple.

## Tests

```powershell
pytest -q
```

Tests cover schema/indexes, insertion, duplicate handling, API failure/cache/sample behavior, invalid and missing users, empty histories, unknown problem indexes, feature leakage, skill calculation/confidence/decay/difficulty weighting, success fallback, skill gaps, personalized difficulty, recommendations, controlled updates, independent Model 1 inference, and the full offline terminal flow.

## Notebooks

Original Model 1 notebooks remain at `notebooks/01_*` through `05_*`.

Model 2 research notebooks are isolated in `notebooks/model2/`:

1. `01_user_data_collection.ipynb`
2. `02_user_eda.ipynb`
3. `03_topic_skill_analysis.ipynb`
4. `04_skill_feature_engineering.ipynb`
5. `05_success_prediction.ipynb`
6. `06_skill_model_evaluation.ipynb`
7. `07_recommendation_system.ipynb`
8. `08_personalized_difficulty.ipynb`

## Major Model 2 files

- `collect_user_data.py` — rate-limited API, cache, sample fallback, database ingestion.
- `database.py` — SQLite schema, indexes, UPSERTs, stable IDs, aggregation.
- `build_user_dataset.py` — chronological interaction export.
- `feature_engineering.py` — leakage-safe training and live inference features.
- `topic_skill.py` — item-response skill model, recency/decay/confidence/curves.
- `skill_estimator.py` — builds and persists topic profiles.
- `train_skill_model.py` — saves configured statistical skill artifact.
- `train_success_model.py` — compares, evaluates, and saves classifiers.
- `predict_success.py` — trained inference and documented fallback.
- `problem_topics.py` — independent topic relevance representation.
- `skill_gap.py` — topic gap/status report.
- `personalized_difficulty.py` — probability-equivalent rating and readiness.
- `recommendation_engine.py` — similar and personalized modes.
- `profile_manager.py` — aggregates and controlled result updates.
- `knowledge_graph.py` — validation, hierarchy, prerequisite traversal.
- `llm_explanation.py` — explanation-only Responses API with fallback.

## Folder structure

```text
app/                              retained Streamlit UI
data/
  raw/                            original Model 1 data
  cleaned/                        original Model 1 data
  user_model/                     Model 2 SQLite, taxonomy, cache, sample
models/
  *.pkl                           retained legacy Model 1 compatibility
  problem_difficulty/             byte-identical Model 1 artifact copies
  user_model/                     independent Model 2 artifacts
notebooks/
  01_* ... 05_*                   retained Model 1 notebooks
  model2/                         eight independent Model 2 notebooks
results/
  existing files                  retained Model 1 experiments
  user_model/                     Model 2 metrics/calibration
src/
  existing modules                retained Model 1 code
  problem_model/                  Model 1 namespace/wrapper
  user_model/                     Model 2 package
tests/
main.py                           presentation/matching layer only
requirements.txt
README.md
```

## Updating and retraining

Typical real-data cycle:

```powershell
python -m src.user_model.collect_user_data YOUR_HANDLE
python -m src.user_model.build_user_dataset
python -m src.user_model.train_skill_model
python -m src.user_model.train_success_model
pytest -q
python main.py
```

Collect multiple consenting/public handles before drawing research conclusions. Re-running collection is safe and incremental. Back up `data/user_model/codeforces_users.db` if the local learning history matters to you.

## Known limitations

- Codeforces API problem objects do not provide full statements; this project links API metadata to the supplied local statement bank when available. A newly added API problem may show “statement unavailable” until local statement data is updated.
- Public `user.status` returns at most the requested history window used here (10,000). Very large histories may need a future paginated/resumable strategy.
- Codeforces verdicts show submissions, not whether a user independently understood an editorial or abandoned an unsubmitted attempt.
- Official tags can reveal solution ideas, so they are hidden in the pre-attempt terminal display.
- Topic relevance from tags/keywords/prerequisites is transparent but imperfect, especially for fine-grained subtopics absent from official tags.
- Skill estimates for sparse topics correctly have low confidence; they are not diagnoses of ability.
- The shipped Model 2 artifact is trained only to make offline execution possible on the explicitly synthetic fixture. Replace it with representative real histories before evaluating or deploying.
- Calibration depends on population and time. Re-check the saved reliability curve after retraining or distribution changes.
- Model 1's supplied dataset is small (794 problems) and its preserved held-out error is substantial; the terminal displays this uncertainty rather than hiding it.
- No automatic submission/judging integration is included.
