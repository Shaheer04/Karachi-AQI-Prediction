import os
import time
import requests
import pandas as pd
import numpy as np
import joblib
import hopsworks
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from src.utils import feature_engineering
from src.api.prediction_service import AQIPredictionService

# Load environment variables
load_dotenv()

HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Constants
LAT = 24.8607
LON = 67.0011
MODEL_DIR = "models_inference"

# Global variables to hold model artifacts
model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print("Loading model and scalers...")
    download_model()
    yield
    # Shutdown: Clean up if needed
    pass

app = FastAPI(lifespan=lifespan)

def download_model():
    """Downloads the best model and scalers from Hopsworks Model Registry."""
    global model_artifacts
    
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        print("Hopsworks credentials not found. Skipping model download.")
        return

    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        # Get best model (force latest version for now to ensure we get the autoregressive one)
        # mr.get_best_model might return v1 if metrics were incomparable
        # Update: We now want to select by RMSE if available.
        # But during dev/migration, explicit versioning is safer if we just pushed distinct changes.
        # However, for production robustness:
        # model = mr.get_best_model(name="aqi_predictor_best", metric="rmse", direction="min")
        
        models = mr.get_models("aqi_predictor_best")
        if not models:
            raise Exception("No models found")
        # Sort by version
        best_model = max(models, key=lambda m: m.version)
        print(f"Downloading model version: {best_model.version}")
        
        # Download to local directory
        model_path = best_model.download()
        
        print(f"Model downloaded to: {model_path}")
        # Debug contents
        # for root, dirs, files in os.walk(model_path):
        #     for file in files:
        #         print(os.path.join(root, file))
        
        # Load artifacts
        # Check if saved as keras or pkl
        if os.path.exists(os.path.join(model_path, "model.keras")):
             model_artifacts['model'] = tf.keras.models.load_model(os.path.join(model_path, "model.keras"))
             model_artifacts['type'] = 'lstm'
        else:
             model_artifacts['model'] = joblib.load(os.path.join(model_path, "model.pkl"))
             model_artifacts['type'] = 'ml'
             
        model_artifacts['scaler_X'] = joblib.load(os.path.join(model_path, "scaler_X.pkl"))
        model_artifacts['scaler_y'] = joblib.load(os.path.join(model_path, "scaler_y.pkl"))
        
        print(f"Model loaded successfully. Type: {model_artifacts['type']}")

    except Exception as e:
        print(f"Failed to load model from Hopsworks: {e}")
        # We don't raise here to allow app to start even if model dl fails (dev mode), 
        # but predict endpoint will fail.
        print("Warning: Model download failed. /predict will return 503.")

def fetch_fs_history():
    """Fetches last 3 days of history from Hopsworks Feature Store (v2 Raw) to prime lag features."""
    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        # USE VERSION 2 (Raw data)
        fg = fs.get_feature_group(name="aqi_features_karachi", version=2)
        
        now = datetime.now(timezone.utc)
        # Getting last 72 hours (3 days) to be safe for lags and missing rows
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(days=3)).timestamp() * 1000)
        
        # Query for range
        query = fg.select_all()
        
        print("Fetching recent history from Hopsworks Feature Store (v2)...")
        # We fetch RAW data
        df_raw = query.filter(fg.datetime_id >= start_time).read()
        
        if df_raw.empty:
            return df_raw
            
        return df_raw # Return RAW, we will engineer it in predict_aqi using main function logic or inference_utils
        
    except Exception as e:
        print(f"Failed to fetch history from Feature Store: {e}")
        raise e

@app.get("/predict")
def predict_aqi():
    if 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Fetch History
        fs_df = pd.DataFrame()
        try:
            fs_df = fetch_fs_history() 
            if fs_df is not None and not fs_df.empty:
                print(f"History fetched: {len(fs_df)} rows.")
            else:
                print("History fetched: 0 rows.")
        except Exception as e:
            print(f"Warning: FS fetch failed ({e}).")

        if fs_df.empty or len(fs_df) < 24:
             msg = f"Not enough history data (found {len(fs_df)} rows). Need > 24h context."
             raise HTTPException(status_code=500, detail=msg)

        # 2. Predict Future   
        service = AQIPredictionService(model_artifacts)
        future_predictions = service.predict_future(fs_df, steps=72)
            
        # 3. Response Construction
        response_data = []
        daily_buckets = {}
        
        for item in future_predictions:
            dt = item['datetime']
            val = item['predicted_aqi']
            
            response_data.append({
                "datetime": dt.isoformat(),
                "predicted_aqi": round(val, 2),
                "pm2_5": round(item['pm2_5'], 2)
            })
            
            day_key = dt.strftime('%Y-%m-%d')
            if day_key not in daily_buckets: daily_buckets[day_key] = []
            daily_buckets[day_key].append(val)
            
        daily_summary = []
        for day, values in daily_buckets.items():
            daily_summary.append({
                "date": day,
                "avg_aqi": round(float(np.mean(values)), 2),
                "min_aqi": round(float(np.min(values)), 2),
                "max_aqi": round(float(np.max(values)), 2)
            })
            
        return {
            "hourly_predictions": response_data,
            "daily_summary": daily_summary
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
