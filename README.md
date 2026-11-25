# Project Handa — Backend API

FastAPI service that serves the class-suspension recommendation model (Random Forest, regularized) for Olongapo City, plus supporting weather data (PAGASA, TCWS, OpenWeatherMap) and a labeling/retraining workflow. Consumed by the [project-handa](../project-handa) Next.js frontend.

## Prerequisites

- Python 3.11+ (a `.venv` may already exist locally; recreate it if it's on an older Python)
- Postgres (via Docker, or the instance already started by `project-handa`'s `docker-compose.yml`)
- Docker, if you'd rather run the whole stack via Compose from `project-handa/`

## Running the app

### Option A: Docker Compose (recommended — runs Postgres + this API + the web app together)

From the **project-handa** folder (not this one):

```bash
cd ../project-handa
docker compose up --build
```

This starts `postgres` (5432), `handa-api` — this backend (8000), and `web` (3000).

### Option B: Run locally

1. Copy the env file and add your OpenWeatherMap key (optional — the API falls back to climatology data if unset):

   ```bash
   cp .env.example .env
   ```

2. Create a virtualenv and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-api.txt
   ```

3. Make sure Postgres is running and reachable. Easiest way is to reuse the frontend's Compose service:

   ```bash
   cd ../project-handa && docker compose up -d postgres && cd -
   ```

   By default the app connects to `postgresql://postgres:postgres@localhost:5432/project_handa` (override with a `DATABASE_URL` env var if needed).

4. Start the API:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   On startup it creates its tables (`weather_snapshots`, `suspension_labels`, `model_versions`) if they don't exist yet, and starts the nightly retraining scheduler.

5. Confirm it's up:

   ```bash
   curl http://localhost:8000/health
   ```

   Interactive API docs are available at `http://localhost:8000/docs`.

## Data & modeling scripts

These are exploratory/offline scripts, not part of the running API:

```bash
python analysis.py          # quick console EDA summary over the raw weather CSV
python suspension_model.py  # trains + evaluates the Random Forest / LogReg / GBoost models, prints metrics
```

Or open [`analysis_notebook.ipynb`](./analysis_notebook.ipynb) for the same exploratory data analysis with charts, graphs, and tables (recommended for a walkthrough of the data and modeling results).

## Project layout

- `app/main.py` — FastAPI routes (predictions, PAGASA/TCWS/live weather, labels, model versions)
- `app/db.py` — SQLAlchemy models + engine (reads `DATABASE_URL`)
- `app/ml_model.py` — trains/caches the live prediction model, staging retrain + promote workflow
- `app/scheduler.py` — nightly staging-retrain job
- `suspension_model.py` — model training/evaluation library (RF, LogReg, GBoost, CV strategies)
- `analysis.py` — early EDA scratch space over the raw weather CSV
- `Cubi Point Daily Data.csv` / `May-November Suspensions - Sheet1.csv` — source weather and suspension-label data
