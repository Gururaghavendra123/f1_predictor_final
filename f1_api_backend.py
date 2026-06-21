"""
F1 Race Predictor API — predict-only.

Loads the model + driver/team stats snapshot produced by train.py and serves
ranked race predictions. Training/data-collection is offline (run train.py); the
API never does heavy work in a request.
"""
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from f1_ml_predictor import F1Predictor
from f1_features import features_for_prediction

predictor: F1Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = F1Predictor()
    if predictor.load():
        snap = predictor.snapshot or {}
        print(f"Model loaded. Snapshot as_of {snap.get('as_of')}, "
              f"{len(snap.get('drivers', {}))} drivers.")
    else:
        print("No model found. Run `python train.py` then restart.")
    yield
    print("Shutting down F1 Predictor API")


app = FastAPI(
    title="F1 Race Predictor API",
    description="Predict F1 race outcomes with a single ranking model",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — local-first. Add prod origins back when deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- request/response models ----
class RaceConditions(BaseModel):
    track_name: str
    country: str = ""
    track_temp: float = 25.0
    air_temp: float = 20.0
    humidity: float = 50.0
    rainfall: bool = False
    track_type: str = "normal"   # kept for UI compatibility; track type is derived from track_name
    era: int = 1                 # 0 = 2022-2025 rules, 1 = 2026+ rules


class DriverInput(BaseModel):
    driver_code: str
    grid_position: int = 10


class PredictionRequest(BaseModel):
    race_conditions: RaceConditions
    drivers: List[DriverInput]


class PredictionResult(BaseModel):
    driver: str
    predicted_position: int
    expected_position: float
    win_probability: float
    podium_probability: float
    confidence: float


def _require_model():
    if predictor is None or not predictor.trained:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py`, then restart the API.",
        )
    if not predictor.snapshot:
        raise HTTPException(
            status_code=503,
            detail="Stats snapshot missing. Re-run `python train.py`.",
        )


# ---- routes ----
@app.get("/")
async def root():
    return {"message": "F1 Race Predictor API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": bool(predictor and predictor.trained)}


@app.post("/predict", response_model=List[PredictionResult])
async def predict_race(request: PredictionRequest):
    _require_model()
    rc = request.race_conditions
    if not request.drivers:
        raise HTTPException(status_code=400, detail="No drivers supplied.")

    rows = []
    for d in request.drivers:
        feats = features_for_prediction(
            predictor.snapshot,
            driver=d.driver_code,
            grid_position=d.grid_position,
            event=rc.track_name,
            era=rc.era,
            track_temp=rc.track_temp,
            air_temp=rc.air_temp,
            humidity=rc.humidity,
            rainfall=rc.rainfall,
        )
        feats['driver'] = d.driver_code
        rows.append(feats)

    try:
        ranked = predictor.rank_race(rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return [PredictionResult(**r) for r in ranked]


@app.get("/drivers")
async def get_available_drivers():
    """Drivers the model actually knows (from the trained snapshot)."""
    if predictor and predictor.snapshot:
        return {"drivers": sorted(predictor.snapshot.get('drivers', {}).keys())}
    return {"drivers": []}


@app.get("/tracks")
async def get_track_info():
    """Tracks for the UI. Names match FastF1 event names so track history resolves."""
    return {
        "tracks": [
            {"name": "Bahrain Grand Prix", "country": "Bahrain", "type": "normal"},
            {"name": "Saudi Arabian Grand Prix", "country": "Saudi Arabia", "type": "street"},
            {"name": "Australian Grand Prix", "country": "Australia", "type": "normal"},
            {"name": "Japanese Grand Prix", "country": "Japan", "type": "technical"},
            {"name": "Chinese Grand Prix", "country": "China", "type": "normal"},
            {"name": "Miami Grand Prix", "country": "United States", "type": "street"},
            {"name": "Emilia Romagna Grand Prix", "country": "Italy", "type": "technical"},
            {"name": "Monaco Grand Prix", "country": "Monaco", "type": "street"},
            {"name": "Canadian Grand Prix", "country": "Canada", "type": "normal"},
            {"name": "Spanish Grand Prix", "country": "Spain", "type": "normal"},
            {"name": "Austrian Grand Prix", "country": "Austria", "type": "highspeed"},
            {"name": "British Grand Prix", "country": "United Kingdom", "type": "highspeed"},
            {"name": "Hungarian Grand Prix", "country": "Hungary", "type": "technical"},
            {"name": "Belgian Grand Prix", "country": "Belgium", "type": "highspeed"},
            {"name": "Dutch Grand Prix", "country": "Netherlands", "type": "technical"},
            {"name": "Italian Grand Prix", "country": "Italy", "type": "highspeed"},
            {"name": "Azerbaijan Grand Prix", "country": "Azerbaijan", "type": "street"},
            {"name": "Singapore Grand Prix", "country": "Singapore", "type": "street"},
            {"name": "United States Grand Prix", "country": "United States", "type": "normal"},
            {"name": "Mexico City Grand Prix", "country": "Mexico", "type": "normal"},
            {"name": "São Paulo Grand Prix", "country": "Brazil", "type": "normal"},
            {"name": "Las Vegas Grand Prix", "country": "United States", "type": "street"},
            {"name": "Qatar Grand Prix", "country": "Qatar", "type": "normal"},
            {"name": "Abu Dhabi Grand Prix", "country": "United Arab Emirates", "type": "normal"},
        ]
    }


@app.get("/model-info")
async def get_model_info():
    if not predictor or not predictor.trained:
        return {"status": "not_trained", "message": "Run python train.py"}
    snap = predictor.snapshot or {}
    return {
        "status": "trained",
        "features_count": len(predictor.feature_columns),
        "available_drivers": len(snap.get('drivers', {})),
        "trained_through": snap.get('as_of'),
        "model_type": "position_ranking",
    }


@app.post("/train")
async def train_models():
    raise HTTPException(
        status_code=501,
        detail="Training is offline. Run `python train.py`, then restart the API.",
    )


@app.post("/retrain")
async def retrain_models():
    raise HTTPException(
        status_code=501,
        detail="Retraining is offline. Run `python train.py`, then restart the API.",
    )


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
