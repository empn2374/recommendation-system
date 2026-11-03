# Phase 8: Deployment and Optimization Summary

**Status:** Native local deployment was tested end to end. Docker support has now
been added at the user's request for Render. Cloud deployment has **not** been
executed. Building/running the container requires Docker; the current local
environment does not have the Docker CLI.

## 1. What was tested locally

A realistic deployment rehearsal, without Docker:

1. Copied only the deployable files (`api/`, `src/`, `artifacts/`, `requirements-api.txt`) into a clean temp folder.
2. Created a brand-new virtualenv and installed **only** `requirements-api.txt` (fastapi, uvicorn, polars, numpy).
3. Started the API from that copy and called every endpoint.

| Check | Result |
|---|---|
| Packages installed | 4 runtime packages. No scikit-learn, scipy or joblib. |
| Endpoints | `/health`, `/options`, `/items`, `/insights`, `/models/metrics`, `/customers/summary` all 200. `POST /recommendations` returned a correct list (+34% profit uplift for that profile). |
| Memory | about 85 MB |
| Cold start (fresh venv on this Windows machine) | 7.9 s to the first healthy response. Expect less on Linux. |
| Cleanup | Temp copy and server removed afterwards. |

The React build (`npm run build`) was also verified and the dev UI was exercised in a real browser against the API (Phase 7).

## 2. What you need to deploy (all free tiers; verify current limits before relying on them)

| Part | Where | Notes |
|---|---|---|
| API | **Render**, Docker web service | [Dockerfile](../Dockerfile) and [render.yaml](../render.yaml) are included. Render builds the image and uses its default command to start Uvicorn on `0.0.0.0:$PORT`. |
| Frontend | **Vercel**, Netlify, Cloudflare Pages or GitHub Pages | Static build of `web/`. Set `VITE_API_URL` to the API's public URL **at build time**. |
| Data and models | In the Git repo | `artifacts/` is only about 210 KB, including the 74 KB model file. The 391 MB raw CSV and the processed Parquet files stay local (see `.gitignore`). |
| Database | None | Not needed. |

**Steps:**
1. Push the repo to GitHub. Make sure `artifacts/` is committed. `data/` is ignored.
2. Render: create a **Blueprint** from the repo to apply `render.yaml`, or create a **Web Service** with language **Docker**, root directory left at the repo root, and Dockerfile path `./Dockerfile`. Leave the Docker command override empty. After it is live, open `/health` and `/docs`.
3. Vercel or Netlify: import the repo, set the root directory to `web`, build command `npm run build`, output directory `dist`, and environment variable `VITE_API_URL=https://<your-render-service>.onrender.com`.
4. `CORS_ORIGINS=*` allows all browser origins as requested. Credentialed cookie access is not enabled. This does not add authentication.

### Backend-only requirements

The API can answer requests without a frontend. The image copies `api/`, `src/`,
and the complete saved `artifacts/`, and installs only `requirements-api.txt`.
The original CSV, processed training rows, and training libraries are not needed.
Startup loads the trained Lite models and supporting summaries once; requests
score and re-rank products without retraining. The frontend is a separate optional
deployment for users who want a visual interface.

The image uses Python 3.13, runs as a non-root user, and defaults to port 10000.
`.dockerignore` restricts the context to serving files. Only trusted pickle model
artifacts should be included.

```powershell
docker build -t recommendation-api .
docker run --rm -p 8000:10000 recommendation-api
```

**Cold starts:** Render's free tier sleeps after a period of inactivity, so the first request can take tens of seconds. Open the app a couple of minutes before the showcase. The fallback is to run locally (section 4), which is also what a recorded demo would use.

## 3. Optimization results (summary)

| Area | Before | After | How |
|---|---|---|---|
| API request time (in process, median) | 16.8 to 28.6 ms | 0.06 to 0.08 ms | Precomputed upsell tables, dict-based scoring, no DataFrame per request |
| End-to-end HTTP request (median, includes curl process start) | not measured | about 4 ms | |
| Artifact load | 9.8 s | 0.04 s | numpy-only lite models, 74 KB single file |
| Python import time | 6.4 s | 1.1 s | No scikit-learn or scipy at serve time |
| API memory | needed scikit-learn + scipy | about 85 MB | |
| API runtime packages | 7 | 4 | |
| JSON payload (`/insights`) | 54 KB | 12.9 KB | gzip middleware (−76%) |
| Frontend initial JS | 647 KB (192 KB gzipped) | 224 KB (70 KB gzipped) | Lazy-loaded pages, chart library loaded on demand |
| Test suite | 33 s | 12 s | Faster serving code |

Correctness was protected throughout: lite models match scikit-learn to 1e-9 on 1,500 real rows for each served model, and the evaluation metrics were re-generated and are unchanged.

## 4. Running everything locally (what you will do for the showcase)

```powershell
# terminal 1: API
$env:PYTHONPATH="src"; .\.venv\Scripts\python -m uvicorn api.main:app --port 8000

# terminal 2: web
cd web; npm run dev     # open http://localhost:5173
```

To rebuild every artifact from the raw CSV, run from the repo root with `PYTHONPATH=src`:
`python -m rec.ingest`, `rec.clean`, `rec.features`, `rec.customers`, `rec.train`, `rec.evaluate`, `rec.crosssell`, `rec.insights`, `rec.samples`.

## 5. Decisions and limits

- **Docker added** at your updated request. The earlier rehearsal used native Python, not a container. Do not treat those historical timings as Docker benchmarks.
- **Hugging Face Spaces** is no longer the primary choice, because it needs Docker or a Gradio/Streamlit app for a FastAPI backend.
- **No cloud deployment was executed.** I did not touch any account or credentials. The local rehearsal in section 1 is the evidence that the packaging works.
- **Free-tier limits change.** Re-check Render's current memory and sleep behavior before relying on them. The API needs about 85 MB, well within a 512 MB class limit.

## 6. Problems encountered and fixes

| Problem | Fix |
|---|---|
| Deployment preference changed from native Python to Docker | Added a serving-only Dockerfile and restricted build context; switched `render.yaml` to Docker runtime. |
| `requirements-api.txt` listed scikit-learn, scipy and joblib that the API no longer needs | Trimmed to four packages and verified in a clean virtualenv. |
| `config.py` creates directories at import, risky on a read-only host | Made it tolerant of `OSError`. |
| The `docker --version` check hung | Stopped it. It was not needed. |
