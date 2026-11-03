# Intelligent Product Recommendation Platform

End-to-end analysis and recommender for a 1M-order synthetic e-commerce dataset: data profiling, customer intelligence, business insights, a profit-aware recommender, a FastAPI service, and a React app.

> **Key finding:** the dataset is synthetic and almost entirely random (one order per customer, product choice independent of customer attributes), so personalization cannot beat popularity. The measurable value is in the business layer: profit-aware re-ranking (+22% profit captured at the same hit rate), category mix, and discount policy. The documentation explains and proves this.

## Documentation (one file per phase)

| Phase | Document |
|---|---|
| 1. Data profiling, quality, preparation | [phases/phase-1-data-profiling-and-quality.md](phases/phase-1-data-profiling-and-quality.md) |
| 2. Features and recommendation design | [phases/phase-2-recommendation-design.md](phases/phase-2-recommendation-design.md) |
| 3. Customer intelligence | [phases/phase-3-customer-intelligence.md](phases/phase-3-customer-intelligence.md) |
| 4. Business insights | [phases/phase-4-business-insights.md](phases/phase-4-business-insights.md) |
| 5. Solution architecture | [phases/phase-5-solution-architecture.md](phases/phase-5-solution-architecture.md) |
| 6. FastAPI service | [phases/phase-6-api.md](phases/phase-6-api.md) |
| 7. React frontend | [phases/phase-7-frontend.md](phases/phase-7-frontend.md) |
| 8. Deployment and optimization | [phases/phase-8-deployment.md](phases/phase-8-deployment.md) |

Each file records what was done, what went wrong, and how it was fixed.

## Quick start (local)

Requires Python 3.13 and Node 20+. The dataset CSV goes in `data/` as `ecommerce_dataset_+1m.csv`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt fastapi "uvicorn[standard]" httpx pytest
cd web; npm install; cd ..

# build every artifact from the raw CSV (run from the repo root)
$env:PYTHONPATH="src"
foreach ($m in "ingest","clean","features","customers","train","evaluate","crosssell","insights","samples") { .\.venv\Scripts\python -m rec.$m }

# serve
.\.venv\Scripts\python -m uvicorn api.main:app --port 8000     # terminal 1, docs at /docs
cd web; npm run dev                                             # terminal 2, open http://localhost:5173
```

The repository already contains the generated `artifacts/` (about 210 KB), so you can skip the build step and just serve.

## Deploy only the API to Render (Docker)

The backend works independently of the React frontend. The image includes `api/`,
`src/`, all saved `artifacts/` (including `models/lite_models.pkl`), and the four
runtime dependencies from `requirements-api.txt`. The raw CSV, training data,
scikit-learn, a database, and retraining are not required to answer requests.
Do not upload only `api/main.py`: it imports the serving code and needs the saved
models, item statistics, and supporting JSON/Parquet artifacts at startup.

1. Push the project, including `artifacts/`, `Dockerfile`, and `render.yaml`, to GitHub.
2. In Render, create a **Blueprint** from the repository to use `render.yaml`.
   Alternatively, create a **Web Service**, select **Docker**, leave the root
   directory at the repository root, and use `./Dockerfile`.
3. Keep the Docker command override empty: the image starts Uvicorn on
   `0.0.0.0` and uses Render's `PORT` (default `10000` locally).
4. Verify `https://<service-name>.onrender.com/health`, then try `/docs`.
   `POST /recommendations` with `{}` returns the default Hybrid recommendations.

`CORS_ORIGINS=*` is set in both the Docker image and Render configuration, allowing
browser requests from any origin. Credentialed cookie access is not enabled.
CORS is not authentication; the endpoints remain publicly accessible.

To try the same image locally, Docker must be installed and running:

```powershell
docker build -t recommendation-api .
docker run --rm -p 8000:10000 recommendation-api
# Open http://localhost:8000/docs
```

`.dockerignore` limits the build context to serving files and artifacts, excluding
the raw dataset, frontend, local environment, and reports. The image runs as a
non-root user. Use only trusted model artifacts because the model file is loaded
with Python pickle.

The frontend can be deployed separately later; set its `VITE_API_URL` to the
Render API URL before building it. Creating these files does not itself provision
a Render service or verify a cloud deployment.

## Tests

```powershell
$env:CORS_ORIGINS="*"
.\.venv\Scripts\python -m pytest tests -q      # API, CORS, feature consistency, lite-model equivalence
cd web; npm run build                          # type-check and production build
```

## Layout

```
src/rec/      pipeline and serving code (ingest, clean, features, models, lite, train, evaluate, crosssell, customers, insights, samples, serving)
api/          FastAPI app
web/          React + Vite + TypeScript app
artifacts/    small precomputed files the API loads (models, item table, insights, metrics)
phases/       documentation, one file per phase
reports/      raw profiling and analysis output
tests/        pytest suite
```

## Suggested showcase flow (about 10 minutes)

1. **Problem and data** (Overview, Data Health): 1M orders, the ten findings, and why the data is effectively random.
2. **Customers** (Customers): value tiers, the 42% of profit from the top 10%, why they cannot be predicted in advance, and the CLV scenarios with their stated assumptions.
3. **Insights** (Insights): category mix, discount depth destroying margin without lifting conversion, returns, and the sized opportunities.
4. **Evaluation** (Evaluation): honest model comparison and the relevance-vs-profit trade-off.
5. **Live demo** (Recommendations): pick a customer, move the profit weight from 0 to 0.25 to 1, show the reasons and upsell hints, then switch on the fraud withholding.
6. **Architecture** (Architecture): as built and the production design.
