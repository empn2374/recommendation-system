# Intelligent Product Recommendation Platform: End-to-End Plan

**Deadline:** Submission and showcase on **Oct 28** (plan written Oct 24, so about 4 working days).
**Stack:** Python (data and ML) + **FastAPI** (serving) + **React (Vite + TypeScript)** (interactive UI).
**Status:** **Built.** This file is the original plan. See [README.md](README.md) and the per-phase documents in [phases/](phases) for what was actually delivered. Deviations from this plan, all driven by profiling the real data:
- The data has **one order per customer**, so collaborative filtering, co-purchase rules and RFM/BG-NBD CLV were not possible. The recommender is profile-based with a profit-aware re-ranker, and CLV is a scenario model (Phase 1 to 3).
- **DuckDB, embeddings and any database were dropped.** The API serves about 210 KB of precomputed artifacts from memory.
- **Docker was skipped.** Deployment is a native Python web service plus a static site (Phase 8). Tailwind was replaced with plain CSS.
- The original column-level assumptions below were checked in Stage 1, and several were wrong (for example `product_id` is not a product key).

---

## 1. Goal and Scope

Build a recommender that scores for **customer relevance** and **business value** (revenue, margin, inventory movement, retention), and present it as an interactive app plus a clear story.

### Deliverables mapped to the assignment

| Assignment phase | What we produce | Where it shows in the demo |
|---|---|---|
| P1: Data analysis and prep | Profiling report, quality assessment, feature engineering doc, assumptions | "Data Health" page and report |
| P2: Recommendation design | Popularity, content-based, collaborative, hybrid, business-rule re-ranking, cross-sell/upsell | "Recommendations" page |
| P3: Customer intelligence | RFM segmentation, high-value customers, behavior analysis, CLV | "Customer 360" page |
| P4: Business insights | Product, marketing, behavior, revenue opportunity and returns insights | "Insights" dashboard |
| P5: Architecture | Scalable design (batch + real-time), diagrams, monitoring and governance | "Architecture" page and slides |
| Bonus | Profit-aware ranking, sentiment, fraud-aware filter, real-time design, GenAI explanations | Toggles in the Recommendations page |

### Scope tiers (so we don't miss the deadline)

- **Must have:** P1 to P5, with popularity, content-based, collaborative, hybrid and business-rule re-ranking, plus the working React + FastAPI demo.
- **Should have:** profit-aware ranking, fraud-aware filter and sentiment signal (cheap to add as re-ranking features), evaluation comparison.
- **Nice to have:** GenAI explanations (template-based first, LLM optional), real-time design (architecture diagram only, not built).

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Data processing | **Polars** (pandas as fallback) | 1M rows fits in memory on a laptop. Polars alone covers cleaning, aggregation and feature work, so **DuckDB is dropped**. |
| Storage | CSV → **Parquet** files. **No database server.** | Parquet is compact and fast. Serving reads small precomputed files into memory, so no DB is needed (see 2.1). |
| Profiling | `ydata-profiling` (sampled) + custom checks | Quick report with documented gaps. |
| Modeling | `implicit` (ALS) or `scikit-learn` TruncatedSVD, `scikit-learn` TF-IDF/cosine, optional **LightGBM** ranker | CPU-friendly, free, and fast to iterate. |
| Text similarity | **TF-IDF** (scikit-learn). **Neural embeddings dropped.** | Free and tiny. Product text is likely short and synthetic, so embeddings add little. |
| Explanations | Template reason codes. LLM is optional and off by default. | Free, deterministic, and no data leaves the machine. |
| API | **FastAPI** + Pydantic + Uvicorn | Typed, auto Swagger docs (useful in the demo). |
| Frontend | **React + Vite + TypeScript**, TanStack Query, Recharts, Tailwind | Fast to build, clean dashboards. |
| Experiments | Jupyter notebooks for EDA; scripts for pipeline | Notebooks for the story, scripts for reproducibility. |
| Packaging | One `Dockerfile` for the API (used for deployment). Local run without Docker is fine. | Docker is only needed for the free host. |
| Tests | `pytest` for pipeline and API contract, light component tests | Enough to prove it works. |

### 2.1 Zero-cost design: what runs where

**Key idea: train offline, serve small precomputed files.** The 1M-row dataset and the heavy ML libraries never go to production.

```
[Your laptop]  raw CSV → Polars cleaning → features → train models → export artifacts/ (a few MB to ~50 MB)
                                                                           │ committed to Git (or HF repo)
[Free host]    FastAPI loads artifacts/ into memory at startup ───────────► [Free static host] React app
```

**What `artifacts/` contains** (all Parquet or JSON):
- Per customer: segment, RFM, CLV, and the top ~50 candidate products with relevance scores (from the hybrid model)
- Per product: price, margin, rating, sentiment, return rate, fraud-risk share, similar items and co-purchase pairs
- Aggregated insight tables for the dashboard and the offline metrics table

**Why this works:**
- The **profit / fraud / sentiment sliders re-rank those cached candidates live** in the API. This is cheap arithmetic, so no model runs at request time.
- The API image only needs `fastapi`, `uvicorn` and `polars` (or `pandas`). It does not need `implicit`, `scikit-learn` or a GPU. That keeps memory low (about 150 to 300 MB) and fits free tiers.
- Predictions for customers that exist in the data are precomputed. New or anonymous users use the segment-popularity or basket-based fallback, which is also precomputed.

**Cost table**

| Item | Choice | Cost |
|---|---|---|
| Model training | Local CPU (`implicit` / scikit-learn) | Free |
| Text similarity | TF-IDF (scikit-learn) | Free (embeddings avoided) |
| Database | **None.** Parquet/JSON loaded into API memory | Free |
| Artifact storage | Git repo (keep each file under 100 MB and the total under ~100 MB; otherwise a Hugging Face repo) | Free |
| Frontend hosting | **Vercel**, Netlify, Cloudflare Pages or GitHub Pages (static Vite build) | Free |
| Backend hosting | **Hugging Face Spaces (Docker)** as first choice, **Render free web service** as the alternative | Free |
| Domain / SSL | Provided subdomain, HTTPS included | Free |
| GenAI explanations | Templates | Free |

**Backend host trade-offs (verify current limits before deploying, as free tiers change)**
- **Hugging Face Spaces (Docker, free CPU):** most generous RAM, good for Python APIs. It may sleep after inactivity and wake on the first request.
- **Render (free):** simple GitHub deploys, but about 512 MB RAM and the service sleeps after ~15 min idle, so the first request takes tens of seconds. Warm it up before the demo.
- **Fallback if both fail:** run the API locally during the showcase and keep the React app pointing at it. A recorded backup demo covers the worst case.

**When a database would actually be justified:** only if we need to persist new data at runtime, such as logging user feedback or clicks. In that case use a free hosted Postgres (Neon or Supabase free tier) or a single SQLite file. It is not needed for the core assignment, so it is out of scope by default.

---

## 3. Proposed Repository Structure

```
recommendation-system/
├─ requirements/                 # given brief (docx, image)
├─ data/
│  ├─ raw/                       # original CSV (gitignored)
│  ├─ interim/                   # cleaned Parquet
│  └─ processed/                 # features, interaction matrices, model artifacts
├─ notebooks/
│  ├─ 01_profiling_quality.ipynb
│  ├─ 02_eda_business_insights.ipynb
│  ├─ 03_customer_segmentation_clv.ipynb
│  └─ 04_recommender_experiments.ipynb
├─ src/rec/                      # python package (shared by notebooks and API)
│  ├─ config.py
│  ├─ ingest.py                  # CSV → Parquet, schema enforcement
│  ├─ quality.py                 # DQ checks and cleaning rules
│  ├─ features.py                # customer/product/interaction features
│  ├─ segmentation.py            # RFM, clustering, CLV
│  ├─ models/
│  │  ├─ popularity.py
│  │  ├─ content.py
│  │  ├─ collaborative.py
│  │  ├─ hybrid.py
│  │  └─ crosssell.py            # association rules / co-purchase
│  ├─ rerank.py                  # business rules, profit/fraud/sentiment/inventory
│  ├─ evaluate.py                # offline metrics
│  └─ explain.py                 # reason codes / GenAI explanation
├─ api/                          # FastAPI app
│  ├─ main.py
│  ├─ routers/ (recommendations, customers, products, insights, health)
│  ├─ schemas.py
│  └─ deps.py                    # loads artifacts at startup
├─ web/                          # React app (Vite + TS)
│  └─ src/ (pages, components, api client, charts)
├─ docs/
│  ├─ data_profiling_report.md
│  ├─ data_quality_assessment.md
│  ├─ feature_engineering.md
│  ├─ assumptions_and_observations.md
│  ├─ architecture.md            # + diagrams (Mermaid)
│  └─ showcase_script.md
├─ tests/
├─ artifacts/                    # small precomputed files shipped to the API (deployed)
├─ Dockerfile                    # API image for the free host
├─ pyproject.toml / requirements.txt
└─ README.md
```

---

## 4. Stages

### Stage 0: Setup and Data Acquisition (Day 1, about 1h)
- Finish the dataset download (the `.crdownload` file is partial). Move it to `data/raw/` and verify the row count (~1M) and column count (62).
- Create the Python environment, the repo skeleton, and the Vite + FastAPI scaffolds.
- Set a **config file** for paths, random seed, and the train/test split date.
- **Exit:** `python -m rec.ingest` converts CSV to Parquet and prints the schema.

### Stage 1: Data Profiling and Quality (Day 1) → Phase 1
- **Profile:** dtypes, cardinality, null rates, distributions, date ranges (2024–2026), numeric ranges, top categories.
- **Identify entity grain:** is each row an order line? Find the keys (order_id, customer_id, product_id), and check duplicates and the uniqueness of each.
- **Quality checks:**
  - Missing values, duplicates, invalid or negative prices/quantities, future dates
  - Inconsistent categories (casing, spelling), outliers, discount/price/profit consistency (price × qty − discount ≈ total)
  - Referential sanity (same product with multiple prices or names, same customer with multiple demographics)
  - Rating and sentiment consistency (low rating with positive sentiment?)
  - Synthetic-data artifacts (uniform distributions, no real seasonality) to flag honestly
- **Decide cleaning rules** and log every change (rows affected) for the quality report.
- **Outputs:** `01_profiling_quality.ipynb`, `docs/data_profiling_report.md`, `docs/data_quality_assessment.md`, cleaned Parquet.

### Stage 2: Feature Engineering (Day 1 to 2) → Phase 1
- **Customer features:** recency, frequency, monetary value (RFM), AOV, tenure, category affinity vector, preferred brand/price band, return rate, discount sensitivity, channel/campaign responsiveness, web-behavior metrics (sessions, pages, cart events if present), average rating given.
- **Product features:** popularity (units, orders, unique buyers), revenue, margin %, return rate, average rating, sentiment score, price band, category/brand, TF-IDF vector of name/category text, stock/inventory proxy (if available).
- **Interaction features:** implicit feedback weight, for example `purchase = 1 + log(qty)`, boosted by rating, with a return penalty and time decay (recent counts more).
- **Leakage control:** build all features using only data **before the split date**.
- **Output:** `docs/feature_engineering.md`, `src/rec/features.py`, feature Parquet tables.

### Stage 3: Customer Intelligence (Day 2) → Phase 3
- **RFM scoring** and named segments (Champions, Loyal, At-risk, Hibernating, New).
- **Clustering** (KMeans or GMM on scaled features) with profiling of each cluster, compared against the RFM segments.
- **High-value customers:** top percentile by revenue and profit, with Pareto/Lorenz analysis.
- **CLV estimation:** start with a simple, defensible model (historical margin × expected frequency × retention). If the data supports it, use BG/NBD + Gamma-Gamma (`lifetimes`/`pymc-marketing`). Document the assumptions.
- **Behavior analysis:** purchase cadence, category journeys, channel mix, discount dependence.
- **Output:** `03_customer_segmentation_clv.ipynb`, segment/CLV tables used by the API.

### Stage 4: Business Insights (Day 2) → Phase 4
Answer with charts and numbers:
- **Product performance:** top/bottom by revenue, profit, margin, growth, and "high rating but low sales" gems.
- **Marketing effectiveness:** conversion, revenue and ROI by channel/campaign, discount depth vs. profit (does discounting erode margin?).
- **Customer behavior:** cohorts and retention, repeat rate, time-to-second-purchase, seasonality.
- **Revenue opportunities:** cross-sell attach rates, under-penetrated categories per segment, win-back potential of at-risk customers.
- **Return patterns:** return rate by category, product, price band, channel, and segment, and the profit leakage from returns.
- **Fraud/payment risk:** the share of risky orders and their effect on revenue.
- **Output:** `02_eda_business_insights.ipynb` and the "Insights" API endpoints and dashboard.

### Stage 5: Recommendation Models (Day 2 to 3) → Phase 2
Build in this order, so a working baseline always exists:

1. **Popularity baseline:** global, per category, per segment, and time-decayed trending.
2. **Content-based:** product similarity from category/brand/price band/text (TF-IDF). The user profile is a weighted average of purchased items. It handles cold-start for products.
3. **Collaborative filtering:** implicit ALS (or SVD) on the user × item matrix. Item-item cosine as a simpler second option and for "customers also bought".
4. **Hybrid:** weighted blend of the CF, content, and popularity scores, with a **fallback ladder**:
   - A new user (cold) gets segment popularity or demographic popularity.
   - A light user gets a content-heavy blend.
   - A heavy user gets a CF-heavy blend.
   - Optionally train a **LightGBM ranker** on candidate features (stretch).
5. **Cross-sell / upsell:**
   - Cross-sell: association rules or co-purchase lift (support, confidence, lift) at the basket or customer level.
   - Upsell: same category, higher price band, with a better rating and an acceptable margin.
6. **Business-rule re-ranking layer** (`rerank.py`), configurable weights:

   ```
   final_score = w_rel · relevance
               + w_profit · normalized_margin
               + w_inv · inventory_movement_boost
               + w_sent · sentiment/rating signal
               - penalties (high return rate, fraud risk, already purchased, out of stock)
   ```
   Hard filters include already purchased (unless consumable), low rating, and high-fraud-risk exclusions.
7. **Explanations:** reason codes first ("Because you bought X", "Top rated in Y", "Frequently bought together"). The GenAI bonus is optional and only if a free-tier LLM is acceptable. The default is templates.

### Stage 6: Offline Evaluation (Day 3)
- **Temporal split:** train on the earlier period, validate on the next window, test on the last.
- **Relevance metrics:** Precision@K, Recall@K, MAP@K, NDCG@K, hit rate.
- **Beyond accuracy:** coverage, diversity, novelty, popularity bias.
- **Business metrics (simulated):** expected revenue and profit of the top-K list, margin uplift of the re-ranker vs. the pure relevance list, and the relevance trade-off shown as a **Pareto curve** (relevance vs. profit).
- **Comparison table:** popularity vs. content vs. CF vs. hybrid vs. hybrid + business rules. This table is a key slide.
- **Be honest:** the data is synthetic, so CF signal may be weak. We report that and explain how the hybrid and rules compensate.

### Stage 7: FastAPI Serving (Day 3) → Phase 5 (serving)
Load the artifacts once at startup (parquet, model files) and serve from memory.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and artifact version |
| `GET /customers?query=` and `GET /customers/{id}` | Search and profile (segment, RFM, CLV, history) |
| `GET /recommendations/{customer_id}?model=&k=&profit_weight=&exclude_fraud=&use_sentiment=` | Personalized recommendations plus a per-item reason and score breakdown |
| `GET /products/{id}` and `GET /products/{id}/similar` | Product detail and similar items |
| `GET /products/{id}/also-bought` | Cross-sell and upsell |
| `GET /insights/products` / `marketing` / `customers` / `returns` / `revenue-opportunities` | Dashboard data |
| `GET /segments` | Segment summary |
| `GET /models/metrics` | Offline evaluation table |
| `POST /recommendations/simulate` | What-if: custom weights and a basket-based (anonymous) recommendation |

Include Pydantic schemas, CORS, caching of hot queries, pagination, and an OpenAPI spec. Add pytest contract tests.

### Stage 8: React Frontend (Day 3 to 4)
**Pages**
1. **Overview/Home:** KPI cards (revenue, profit, customers, orders, return rate) and the project story.
2. **Data Health:** the profiling and quality summary (null rates, issues found, cleaning actions).
3. **Recommendations Playground (hero page):**
   - Pick or search a customer and choose the model (popularity / content / CF / hybrid)
   - **Sliders:** relevance vs. profit weight; toggles for fraud filter, sentiment boost and inventory boost
   - Product cards with score, margin, rating, and a "Why this?" explanation
   - A **side-by-side compare** of the baseline vs. business-aware list, with revenue/profit deltas
4. **Customer 360:** segment, RFM, CLV, purchase history, category affinity.
5. **Segments:** segment sizes and value, with a profile of each.
6. **Insights Dashboard:** tabs for Products, Marketing, Behavior, Returns, Revenue opportunities.
7. **Model Evaluation:** the metrics comparison and the relevance-vs-profit trade-off chart.
8. **Architecture:** the rendered diagram and component notes.

**UI quality:** loading and error states, responsive layout, consistent charts, and a typed API client generated from the OpenAPI spec.

### Stage 9: Architecture and Governance (Day 4) → Phase 5
Documented in `docs/architecture.md` with Mermaid diagrams, covering **what we built (local)** and **how it scales (production)**:

| Concern | Demo implementation | Production design |
|---|---|---|
| Ingestion | CSV → Parquet script | Batch via ADF/Airflow and CDC/streaming via Kafka/Event Hubs |
| Storage | Parquet files in the repo | Lakehouse (bronze/silver/gold on ADLS/S3 + Delta), a warehouse for BI |
| Feature engineering | Python scripts | Spark/Databricks jobs plus a **feature store** (Feast) for online/offline parity |
| Training | Local scripts | Scheduled retraining (daily/weekly), MLflow tracking and registry |
| Serving | FastAPI in memory | Candidate generation (ANN, e.g. FAISS) → ranking → re-ranking, behind an API gateway, with Redis cache |
| Real-time (bonus) | Design only | Event stream → streaming feature updates → session-based recs (low-latency path) |
| Monitoring | Basic logging and metrics endpoint | Data drift, model drift, CTR/conversion tracking, latency/error SLOs |
| Governance | Documented | PII handling, data lineage, access control, fairness/diversity checks, A/B testing, audit logs, GDPR delete/opt-out |

Also cover cold-start strategy, A/B testing plan, feedback loop, and cost considerations.

### Stage 10: Packaging, Testing and Showcase (Day 4, Oct 27 to 28)
- Two documented commands start the API and web app locally. Deploy the API to Hugging Face Spaces or Render and the React build to Vercel or Netlify (see 2.1).
- `README` with setup, architecture, results, and screenshots.
- Smoke-test the whole flow with a **sample of 3 to 4 demo customers** (heavy buyer, new buyer, at-risk, high-value) prepared in advance.
- Prepare a **10 to 12 minute showcase** (`docs/showcase_script.md`):
  1. Problem and goal (1 min)
  2. Data findings and quality issues (2 min)
  3. Customer segments and CLV (2 min)
  4. Business insights, the top 3 (2 min)
  5. Live demo of the playground, including the profit slider and explanations (3 min)
  6. Architecture and scaling (1 min)
  7. Results, limitations and next steps (1 min)
- Record a short backup demo video in case the live demo fails.

---

## 5. Schedule (Oct 24 to Oct 28)

| Day | Focus | Done when |
|---|---|---|
| **Oct 24 (Fri)** | Stage 0 to 2: setup, profiling, quality, start of features | Clean Parquet, profiling and DQ report drafted |
| **Oct 25 (Sat)** | Stage 2 to 5: features, segmentation/CLV, popularity + content + CF baselines | Segments and CLV tables, 3 working models |
| **Oct 26 (Sun)** | Stage 5 to 7: hybrid, re-ranker, evaluation, FastAPI endpoints | API serves recommendations and insights |
| **Oct 27 (Mon)** | Stage 8 to 9: React pages, charts, architecture docs | End-to-end demo works locally |
| **Oct 28 (Tue)** | Stage 10: polish, tests, rehearsal, submit and showcase | Submitted and the showcase is ready |

If time is short, cut in this order: GenAI explanations → LightGBM ranker → CLV probabilistic model → Architecture page rendering (keep the doc).

---

## 6. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Dataset download is incomplete or corrupt | Re-download now and verify the row count. Don't start modeling until it is confirmed. |
| Synthetic data has weak collaborative signal | Lean on the hybrid and content models and the business re-ranking. Report the evaluation honestly. |
| Column meanings are unclear | Document assumptions in `assumptions_and_observations.md` early. Confirm the grain and keys at Stage 1. |
| Memory and speed on 1M rows | Use Parquet, Polars, sparse matrices, and precompute top-N lists. |
| Scope creep | Follow the must/should/nice tiers. Keep a working baseline at each stage. |
| Live demo failure | Precomputed artifacts, local only, and a backup recording. |

---

## 7. Open Decisions (to confirm before building)

1. **Python tooling:** pandas vs. Polars (default: Polars, no DuckDB).
2. **UI styling:** Tailwind + shadcn (default) or another component library.
3. **GenAI explanations:** template reason codes only (default, free), or a free-tier LLM call (needs an API key, and has data-sharing implications)?
4. **Submission format:** repository plus slides, a notebook, or just the live demo?
5. **Real-time:** architecture diagram only (default) or a small simulated event stream?
6. **Hosting:** Hugging Face Spaces (default) or Render for the API, and Vercel or Netlify for the React app. Do you have accounts, or should we plan for local-only?

## 8. Definition of Done

- All 5 phases documented and backed by code or notebooks.
- At least 4 recommendation approaches compared offline with a metrics table.
- Profit-aware re-ranking demonstrably moves margin with a measured relevance trade-off.
- React app lets a user pick a customer, tune the weights and see explained recommendations.
- Architecture document with diagrams covering all six required areas.
- A rehearsed demo plus a backup recording.
