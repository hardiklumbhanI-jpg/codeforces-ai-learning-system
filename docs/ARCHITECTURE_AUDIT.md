# Pre-implementation architecture audit

This audit was completed before Model 2 files were added. The supplied ZIP contained 794 raw and 794 cleaned problems, 12 Python modules, one Streamlit app, five notebooks, nine pickle artifacts, one JSON model configuration, two plots, and 13 experiment/result files.

## Existing production Model 1

`src/predict_rating.py` is the production rating path. It loads `models/final_rating_model.pkl`, `final_tfidf_vectorizer.pkl`, `structural_scaler.pkl`, and `tag_encoder.pkl`. A statement is transformed into 10,000-feature word/bigram TF-IDF, six standardized structural counts, and 37 multi-hot Codeforces tags. The dense combined matrix feeds a `GradientBoostingRegressor`. The preserved held-out file reports MAE around 516 and R² around 0.37 for the selected experiment.

`src/predict_topics.py` is a separate Model 1 helper: one-vs-rest logistic regression over statement TF-IDF with a saved 0.45 threshold. `src/recommend_problems.py`, `problem_similarity.py`, and `learning_analysis.py` are deterministic Model 1-era helpers. `app/streamlit_app.py` consumes these modules in six tabs.

## Existing experimental or legacy files

- `src/train_rating_model.py` compares Linear, Ridge, and Gradient Boosting models; it is not the production inference path.
- `models/ridge_rating_model.pkl` and `models/tfidf_vectorizer.pkl` are older statement-only experiment artifacts.
- `src/train_final_rating_model.py` is the production Model 1 trainer.
- `src/train_topic_model.py` is the topic-model trainer.
- `src/test_prediction.py` is a small manual smoke test.
- `notebooks/01_*` through `05_*`, `results/*`, and `plots/*` are retained experiment/research history.
- The cleaned CSV differs from raw primarily through name/tag normalization; both have 794 unique URLs and statements.

## Preservation decision

Every original Model 1 Python file, notebook, data file, result, plot, and root model artifact was retained. Model 2 lives only under `src/user_model`, `models/user_model`, `data/user_model`, `results/user_model`, and `notebooks/model2`. `main.py` is the only matching/presentation layer. Model 2 feature code has no import of Model 1 and explicitly excludes Model 1 predicted ratings.
