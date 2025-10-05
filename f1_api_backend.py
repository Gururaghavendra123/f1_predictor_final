from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import asyncio
from contextlib import asynccontextmanager

# Import our custom modules
from f1_data_collector import F1DataCollector
from f1_ml_predictor import F1Predictor

# Global variables
predictor = None
driver_stats = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor, driver_stats
    predictor = F1Predictor()
    
    # Try to load existing models
    if not predictor.load_models():
        print("No existing models found. Ready for training.")
    
    yield
    # Shutdown
    print("Shutting down F1 Predictor API")

app = FastAPI(
    title="F1 Race Predictor API",
    description="Predict F1 race outcomes using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class RaceConditions(BaseModel):
    track_name: str
    country: str
    track_temp: float = 25.0
    air_temp: float = 20.0
    humidity: float = 50.0
    rainfall: bool = False
    track_type: str = "normal"  # "street", "highspeed", "technical", "normal"

class DriverInput(BaseModel):
    driver_code: str  # e.g., "VER", "HAM"
    grid_position: int = 10

class PredictionRequest(BaseModel):
    race_conditions: RaceConditions
    drivers: List[DriverInput]

class PredictionResult(BaseModel):
    driver: str
    win_probability: float
    podium_probability: float
    predicted_position: int
    confidence: float

class TrainingRequest(BaseModel):
    years: List[int] = [2022, 2023, 2024]
    retrain: bool = False

# API Routes
@app.get("/")
async def root():
    return {"message": "F1 Race Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": predictor.trained if predictor else False
    }

@app.post("/train")
async def train_models(request: TrainingRequest):
    """Train or retrain the prediction models"""
    try:
        global predictor, driver_stats
        
        # Check if models already exist and retrain is not requested
        if predictor and predictor.trained and not request.retrain:
            return {
                "message": "Models already trained. Use retrain=true to force retrain.",
                "status": "already_trained"
            }
        
        # Initialize data collector
        collector = F1DataCollector()
        
        # Download session data
        print(f"Starting data collection for years: {request.years}")
        sessions = collector.get_race_sessions(request.years)
        
        if not sessions:
            raise HTTPException(status_code=400, detail="No session data collected")
        
        # Create driver features
        driver_stats = collector.create_driver_features()
        
        # Create training dataset
        training_df = collector.create_training_dataset()
        
        # Save training data
        training_df.to_csv('f1_training_data.csv', index=False)
        
        # Train models
        predictor = F1Predictor()
        training_results = predictor.train_models('f1_training_data.csv')
        
        return {
            "message": "Models trained successfully",
            "status": "success",
            "data": {
                "sessions_collected": len(sessions),
                "training_samples": len(training_df),
                "drivers_analyzed": len(driver_stats),
                "model_performance": training_results
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=List[PredictionResult])
async def predict_race(request: PredictionRequest):
    """Predict race outcomes for given conditions and drivers"""
    try:
        if not predictor or not predictor.trained:
            # Try to load models first
            if not predictor or not predictor.load_models():
                raise HTTPException(
                    status_code=400, 
                    detail="Models not trained. Please call /train endpoint first."
                )
        
        # Prepare race data for prediction
        race_data = []
        
        # Determine track characteristics
        track_features = get_track_features(
            request.race_conditions.track_name,
            request.race_conditions.track_type
        )
        
        for driver in request.drivers:
            # Get driver historical stats
            driver_perf = get_driver_performance(driver.driver_code)
            
            race_entry = {
                'driver': driver.driver_code,
                'country': request.race_conditions.country,
                'grid_position': driver.grid_position,
                'track_temp': request.race_conditions.track_temp,
                'air_temp': request.race_conditions.air_temp,
                'humidity': request.race_conditions.humidity,
                'rainfall': int(request.race_conditions.rainfall),
                **track_features,
                **driver_perf
            }
            
            race_data.append(race_entry)
        
        # Make predictions
        predictions = predictor.predict_race_outcome(race_data)
        
        # Convert to response format
        results = []
        for pred in predictions:
            results.append(PredictionResult(
                driver=pred['driver'],
                win_probability=pred['win_probability'],
                podium_probability=pred['podium_probability'],
                predicted_position=pred['predicted_position'],
                confidence=pred['confidence']
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/drivers")
async def get_available_drivers():
    """Get list of available drivers for prediction"""
    if predictor and predictor.trained:
        drivers = predictor.get_driver_list()
        return {"drivers": drivers}
    else:
        # Return common F1 drivers if models not trained
        return {
            "drivers": [
                "VER", "HAM", "RUS", "LEC", "SAI", "NOR", "PIA", "ALO", "STR",
                "PER", "GAS", "OCO", "ALB", "SAR", "MAG", "HUL", "RIC", "TSU", "ZHO", "BOT"
            ]
        }

@app.get("/tracks")
async def get_track_info():
    """Get information about F1 tracks"""
    return {
        "tracks": [
            {"name": "Monaco", "country": "Monaco", "type": "street"},
            {"name": "Silverstone", "country": "United Kingdom", "type": "highspeed"},
            {"name": "Monza", "country": "Italy", "type": "highspeed"},
            {"name": "Spa", "country": "Belgium", "type": "highspeed"},
            {"name": "Suzuka", "country": "Japan", "type": "technical"},
            {"name": "Interlagos", "country": "Brazil", "type": "normal"},
            {"name": "Circuit of the Americas", "country": "United States", "type": "normal"},
            {"name": "Albert Park", "country": "Australia", "type": "normal"},
            {"name": "Imola", "country": "Italy", "type": "technical"},
            {"name": "Miami", "country": "United States", "type": "street"},
            {"name": "Barcelona", "country": "Spain", "type": "normal"},
            {"name": "Red Bull Ring", "country": "Austria", "type": "normal"},
            {"name": "Paul Ricard", "country": "France", "type": "normal"},
            {"name": "Hungaroring", "country": "Hungary", "type": "technical"},
            {"name": "Zandvoort", "country": "Netherlands", "type": "normal"},
            {"name": "Monza", "country": "Italy", "type": "highspeed"},
            {"name": "Singapore", "country": "Singapore", "type": "street"},
            {"name": "Baku", "country": "Azerbaijan", "type": "street"},
            {"name": "Jeddah", "country": "Saudi Arabia", "type": "street"},
            {"name": "Abu Dhabi", "country": "United Arab Emirates", "type": "normal"},
            {"name": "Las Vegas", "country": "United States", "type": "street"}
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained models"""
    if not predictor or not predictor.trained:
        return {"status": "not_trained", "message": "Models not trained yet"}
    
    return {
        "status": "trained",
        "features_count": len(predictor.feature_columns),
        "available_drivers": len(predictor.get_driver_list()) if predictor.get_driver_list() else 0,
        "model_types": ["win_prediction", "podium_prediction", "position_prediction"]
    }

# Helper functions
def get_track_features(track_name: str, track_type: str):
    """Get track feature encoding"""
    track_features = {
        'track_is_street': 0,
        'track_is_highspeed': 0,
        'track_is_technical': 0
    }
    
    if track_type == "street":
        track_features['track_is_street'] = 1
    elif track_type == "highspeed":
        track_features['track_is_highspeed'] = 1
    elif track_type == "technical":
        track_features['track_is_technical'] = 1
    
    return track_features

def get_driver_performance(driver_code: str):
    """Get driver performance stats (simplified for demo)"""
    # Default performance stats
    default_stats = {
        'driver_win_rate': 0.1,
        'driver_podium_rate': 0.3,
        'driver_avg_finish': 10.0,
        'driver_avg_grid': 10.0,
        'driver_dnf_rate': 0.1,
        'driver_avg_points': 5.0
    }
    
    # Top drivers with better stats
    top_drivers = {
        'VER': {
            'driver_win_rate': 0.6,
            'driver_podium_rate': 0.8,
            'driver_avg_finish': 2.5,
            'driver_avg_grid': 2.0,
            'driver_dnf_rate': 0.05,
            'driver_avg_points': 20.0
        },
        'HAM': {
            'driver_win_rate': 0.3,
            'driver_podium_rate': 0.6,
            'driver_avg_finish': 4.0,
            'driver_avg_grid': 3.5,
            'driver_dnf_rate': 0.03,
            'driver_avg_points': 15.0
        },
        'LEC': {
            'driver_win_rate': 0.2,
            'driver_podium_rate': 0.5,
            'driver_avg_finish': 5.0,
            'driver_avg_grid': 4.0,
            'driver_dnf_rate': 0.08,
            'driver_avg_points': 12.0
        },
        'RUS': {
            'driver_win_rate': 0.1,
            'driver_podium_rate': 0.4,
            'driver_avg_finish': 6.0,
            'driver_avg_grid': 5.0,
            'driver_dnf_rate': 0.04,
            'driver_avg_points': 10.0
        }
    }
    
    return top_drivers.get(driver_code, default_stats)

# Additional utility endpoints
@app.post("/retrain")
async def retrain_models():
    """Force retrain models with existing data"""
    try:
        if not os.path.exists('f1_training_data.csv'):
            raise HTTPException(status_code=400, detail="No training data found. Please call /train first.")
        
        global predictor
        predictor = F1Predictor()
        training_results = predictor.train_models('f1_training_data.csv')
        
        return {
            "message": "Models retrained successfully",
            "status": "success",
            "performance": training_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/prediction-template")
async def get_prediction_template():
    """Get a template for making predictions"""
    return {
        "race_conditions": {
            "track_name": "Monaco",
            "country": "Monaco",
            "track_temp": 25.0,
            "air_temp": 20.0,
            "humidity": 50.0,
            "rainfall": False,
            "track_type": "street"
        },
        "drivers": [
            {"driver_code": "VER", "grid_position": 1},
            {"driver_code": "HAM", "grid_position": 2},
            {"driver_code": "LEC", "grid_position": 3}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)