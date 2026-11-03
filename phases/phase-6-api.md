# Phase 6: FastAPI Service

**Status:** Done and tested.
**Code:** [api/main.py](../api/main.py), [serving.py](../src/rec/serving.py), [lite.py](../src/rec/lite.py), tests in [tests/](../tests)
**Run locally:** `PYTHONPATH=src uvicorn api.main:app --port 8000`, then open `http://127.0.0.1:8000/docs` for the interactive API docs.

## 1. What the service does

It loads about 210 KB of precomputed artifacts once at startup and answers requests from memory. There is no database. A request builds a feature dict, scores 48 items with a numpy model, applies the business re-ranker, and returns explained results.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, model list, training size and split date |
| `GET /options` | Dropdown values and default profile for the UI |
| `GET /customers`, `GET /customers/summary` | 300 real test-period customers (value tier, cluster, CLV proxy), and the Phase 3 summary |
| `POST /recommendations` | Ranked, explained list for a profile. Parameters: model, K, profit weight (λ), return, fraud and sentiment terms, risk-withholding switch and threshold, category filter |
| `GET /items`, `GET /items/{item}/related` | Item table, and cross-sell and upsell tiers for an item |
| `GET /insights`, `GET /insights/{section}` | Phase 4 insights |
| `GET /models/metrics` | Offline evaluation and the λ trade-off table |

The `/recommendations` response includes, per item: relevance, final score, net profit per order, margin, return rate, rating, reason codes, and an upsell hint. It also includes a comparison against the relevance-only list (profit uplift, relevance change, overlap).

**Bonus features built into this endpoint:** profit-aware ranking (λ), fraud-aware withholding and an item-level fraud penalty, a sentiment boost, and template-based explanations.

## 2. Validation and error handling

Pydantic rejects bad input before it reaches the model: age 18 to 100, loyalty 0 to 100, K 1 to 48, λ 0 to 1, hour 0 to 23, month 1 to 12. Unknown model names return 400, unknown items and insight sections return 404. CORS is configurable through `CORS_ORIGINS`, and responses over 1 KB are gzipped.

## 3. Tests (17, all passing)

- **API behavior:** response shape and probabilities, scores sorted, λ=0 gives zero uplift, λ=1 gives an all-Electronics list, every model works, fraud withholding, category filter, validation and error codes, related items, insights, upsell always pointing to a higher-profit tier.
- **Feature consistency:** 500 real rows, with the API's single-profile bucketing compared against the training pipeline's bucketing. This guards against training/serving skew.
- **Lite-model equivalence:** the numpy models match the scikit-learn models to 1e-9 on 1,500 real rows for each of the 6 served models.
- **Dependency check:** a subprocess confirms the serving code never imports scikit-learn or scipy.

Beyond the test suite, all 300 sample customers were run through the live API at λ=0 and λ=1 (600 calls, 0 failures).

## 4. Optimization: numpy-only inference

The first working version used the pickled scikit-learn models. Profiling showed two problems:

| Measurement | First version | After optimization |
|---|---|---|
| Request time (median, in process) | 16.8 to 28.6 ms | **0.06 to 0.08 ms** (about 250× faster) |
| Artifact load | 9.8 s | **0.04 s** |
| Python import time | 6.4 s | **1.1 s** |
| Server memory | needs scikit-learn and scipy | **85 MB** |
| Runtime packages | 7 | **4** (fastapi, uvicorn, polars, numpy) |
| Test suite time | 33 s | 12 s |

**What was done:**
1. About 94% of request time was polars filter and sort calls inside the per-item upsell lookup. Upsell tiers, option lists and cross-sell tables are now precomputed at startup.
2. Scoring no longer builds a DataFrame per request. It works from a plain feature dict.
3. `lite.py` holds numpy-only classes (constant, segment lookup, naive Bayes, logistic regression, hybrid). `train.py` converts each fitted scikit-learn model into one of these and saves them as a single 74 KB `lite_models.pkl`.
4. The scikit-learn originals stay in `data/processed/models_sklearn/` (local only) so the equivalence test can compare against them.

## 5. Problems encountered and fixes

| Problem | Fix |
|---|---|
| `split_date` in the API read `2025-11-01 00:00:00.alias("datetime")`, a Polars expression turned into a string and written into the metadata | Added a plain `SPLIT_DATE_STR` constant. Regenerated `meta.json` and `metrics.json`. Metrics are unchanged. |
| Request latency of 16 to 28 ms for even a trivial model | Profiled with cProfile and precomputed the upsell lookups (section 4). |
| Slow, heavy startup from importing scikit-learn | numpy-only lite models, proven equivalent by tests. |
| The sample customers carried `coupon_used`, `abandoned_cart_before` and `is_weekend` as the strings "true"/"false", so the UI checkboxes showed the wrong state (the API parsed them correctly) | Exported them as booleans in `samples.py`, and added a defensive conversion in the UI. Verified in a real browser. |
| `train.py` ran twice after an edit left a duplicate `__main__` block | Removed the duplicate. |
| `config.py` creates directories on import, which could crash on a read-only host | Wrapped it in a `try/except OSError`. |

## 6. Next

Phase 7: the React frontend. Phase 8: deployment and optimization results.
