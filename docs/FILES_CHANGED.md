# Implementation file summary

## Existing files modified

- `README.md` — comprehensive two-model architecture, setup, formulas, commands, API/LLM/offline guidance, limitations, and retraining documentation.
- `requirements.txt` — version-compatible ML/runtime/test dependencies; `ipykernel` replaces the Windows-long-path-prone Jupyter meta-package.
- `.gitignore` — local environments, caches, SQLite sidecars, and test caches.

No supplied Model 1 Python, notebook, data, result, plot, or serialized model file was modified.

## Major files/directories added

- `main.py`
- `.env.example`
- `docs/ARCHITECTURE_AUDIT.md`
- `docs/FILES_CHANGED.md`
- `src/problem_model/`
- `src/user_model/` (collection, database, taxonomy, skills, features, training, prediction, gaps, personalization, recommendations, explanations)
- `data/user_model/` (SQLite, knowledge graph, cache directory, sample CSVs)
- `models/problem_difficulty/` (byte-identical compatibility copies)
- `models/user_model/` (trained sample artifacts and metadata)
- `notebooks/model1/` (retained-notebook compatibility copies)
- `notebooks/model2/` (eight research notebooks)
- `results/model1/` (retained-result compatibility copies)
- `results/user_model/` (metrics, comparison, predictions, calibration)
- `tests/` (17 automated tests)
