# Ghost Lap

**Predicting F1 race outcomes with machine learning and real data.**

Ghost Lap takes a starting grid + track conditions and predicts the finishing
order of a Formula 1 race. It is built on real race data (via the FastF1 API),
a leakage-free feature pipeline, a single gradient-boosted ranking model, and an
immersive pit-wall style React UI.

> Honest accuracy (trained on 2022–2024, tested on the **unseen** 2025 season):
> **50% winner accuracy · 61% podium overlap · 3.55 position MAE.**
> No data leakage, no inflated numbers — see [`INTERVIEW.md`](INTERVIEW.md) for the full reasoning.

---

## How it works (at a glance)

```
 FastF1 API
     │   raw race results (2022→now)
     ▼
 f1_data_collector.py ── incremental parquet store  (data/races/*.parquet)
     │
     ▼
 f1_features.py ── time-ordered, leakage-free features
     │              (driver form, team/car pace, track history, era, grid…)
     ▼
 train.py ── temporal validation + walk-forward, then fit final model
     │            saves → models/position_model.pkl, scaler.pkl, driver_stats.json
     ▼
 f1_api_backend.py ── FastAPI, predict-only (loads model + snapshot)
     │            POST /predict → ranked finishing order
     ▼
 f1-frontend ── React UI (pit-wall design, lights-out predict, timing tower)
```

**One model, one source of truth.** A single `XGBRegressor` predicts each
driver's *expected finishing position*. Ranking, win % and podium % are all
derived from that one score, so outputs can never contradict each other.

---

## Why it's different

- **No data leakage.** Driver/team stats for race *R* are computed only from
  races *before* R. The old "80% accuracy" projects leak the answer into the
  features; Ghost Lap doesn't — its numbers are real.
- **Car pace is a feature.** Team/constructor form is modelled explicitly — the
  car is ~80% of F1 pace, and it's the thing that resets every rules era.
- **2026-aware.** An `era` flag + recency-weighted training + rolling recent-form
  features let the model adapt to the 2026 regulation reset as new races arrive.
- **Validated honestly.** Temporal hold-out (train past → test future season) and
  walk-forward validation, not random splits on leaked features.

---

## Tech stack

| Layer | Tech |
|-------|------|
| Data | FastF1, pandas, pyarrow (parquet) |
| ML | XGBoost, scikit-learn (scaling), joblib |
| API | FastAPI, Uvicorn, Pydantic |
| UI | React 19, lucide-react, pure-CSS animation (Anton / Archivo / JetBrains Mono) |

---

## Project structure

```
f1_predictor/
├── f1_data_collector.py   # incremental FastF1 → parquet raw store
├── f1_features.py         # leakage-free feature builder (train + serve share it)
├── f1_ml_predictor.py     # the single ranking model (fit / rank / save / load)
├── train.py               # offline pipeline: validate + fit + save snapshot
├── f1_api_backend.py      # FastAPI predict-only service
├── requirements.txt
├── models/                # position_model.pkl, scaler.pkl, *.json (built by train.py)
├── data/races/            # parquet raw store (built by collector, git-ignored)
├── cache/                 # FastF1 HTTP cache (git-ignored)
├── INTERVIEW.md           # ML interview deep-dive + Q&A
└── f1-frontend/           # React app
    ├── src/App.js         # main UI
    ├── src/App.css        # design system
    └── src/index.css      # tokens + background
```

---

## 🚀 Run it locally (step by step)

Prerequisites: **Python 3.11+** and **Node 18+**.

### 1. Backend setup

```bash
# from the project root
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Build the data store

```bash
# Pulls race results into data/races/*.parquet.
# Uses the FastF1 cache if present; otherwise downloads (incremental + rate-limit safe).
python f1_data_collector.py
```

> First run on a fresh machine downloads data and can take a while. Re-runs are
> instant — already-stored races are skipped. To pull the **current season**
> (e.g. 2026), just run it again; it fetches only the new, already-completed races.

### 3. Train the model

```bash
python train.py
```

This prints honest validation metrics, then saves `models/position_model.pkl`,
`models/scaler.pkl`, `models/feature_columns.json`, `models/driver_stats.json`.

### 4. Start the API

```bash
python f1_api_backend.py
# → http://localhost:8000   (docs at /docs)
```

### 5. Start the frontend (new terminal)

```bash
cd f1-frontend
npm install
npm start
# → http://localhost:3000
```

Open **http://localhost:3000**, set the grid + conditions, hit **Run Prediction**. 🏎️

> **Retraining later:** re-run `python f1_data_collector.py` (gets new races) then
> `python train.py`, and restart the API. The API never trains — it only serves.

---

## 📦 Git — push it (one clean commit)

Some artifacts used to be committed but are now generated locally (models, data,
training csv). Untrack them, keep the source + the small JSON the app needs.

```bash
# 1. See what's going on
git status

# 2. Stop tracking generated/large files (keep them on disk)
git rm -r --cached cache data 2>/dev/null
git rm --cached f1_training_data.csv 2>/dev/null
git rm --cached models/position_model.pkl models/scaler.pkl 2>/dev/null
git rm --cached models/win_model.pkl models/podium_model.pkl models/label_encoders.pkl 2>/dev/null

# 3. Stage everything (new code, docs, frontend, .gitignore)
git add -A

# 4. Commit
git commit -m "Rebuild Ghost Lap: leakage-free pipeline, single ranking model, new UI"

# 5. Push
git push origin main
```

> `.gitignore` already excludes `cache/`, `data/`, `*.csv`, and `models/*.pkl`.
> The trained model is regenerated with `python train.py`, so it doesn't need to
> live in git. `models/feature_columns.json` and `models/driver_stats.json` stay
> tracked (small, and handy to inspect).

---

## ⚠️ Disclaimer

For learning and fun. F1 is chaotic — these are probabilistic predictions, not
betting advice.

**Built by Guru with tons of coffee and brainstroming lolll**