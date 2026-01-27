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
        
        # Get best model (force latest version for now)
        models = mr.get_models("aqi_predictor_best")
        if not models:
            raise Exception("No models found")
        
        # Sort by version to get the absolute latest
        best_model = max(models, key=lambda m: m.version)
        print(f"Downloading model version: {best_model.version}")
        
        # --- 1. Dynamic Metrics Extraction ---
        try:
            # Hopsworks stores metrics in .training_metrics (dict)
            metrics = best_model.training_metrics if best_model.training_metrics else {}
            
            # If empty, try .metrics (older SDK versions might use this)
            if not metrics and hasattr(best_model, 'metrics'):
                metrics = best_model.metrics
            
            # Ensure we have valid data for the UI
            clean_metrics = {}
            if metrics:
                for k, v in metrics.items():
                    # Check if value is a number and format it
                    try:
                        if isinstance(v, (int, float)):
                            clean_metrics[k] = round(float(v), 4)
                        else:
                            clean_metrics[k] = v # Keep string/other as is
                    except:
                        clean_metrics[k] = str(v)
            else:
                print("Warning: No metrics found in model metadata.")
                clean_metrics = {"Status": "No metrics available"}
                
            model_artifacts['metrics'] = clean_metrics
            
        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
            model_artifacts['metrics'] = {"Error": "Failed to load metrics"}
        
        model_artifacts['name'] = best_model.name
        model_artifacts['version'] = best_model.version
        
        # Download to local directory
        model_path = best_model.download()
        print(f"Model downloaded to: {model_path}")
        
        # --- 2. Dynamic Model Loading & Type Detection ---
        if os.path.exists(os.path.join(model_path, "model.keras")):
             loaded_model = tf.keras.models.load_model(os.path.join(model_path, "model.keras"))
             model_artifacts['model'] = loaded_model
             # User requested specific naming like "LSTM"
             # We can try to inspect layers or just default to "LSTM" for Keras models in this project context
             model_artifacts['type'] = 'LSTM' 
        else:
             loaded_model = joblib.load(os.path.join(model_path, "model.pkl"))
             model_artifacts['model'] = loaded_model
             
             # Dynamic Type Detection for Scikit-Learn / XGBoost / CatBoost
             type_name = type(loaded_model).__name__
             
             if 'XGB' in type_name:
                 model_artifacts['type'] = 'XGBoost'
             elif 'LGBM' in type_name:
                 model_artifacts['type'] = 'LightGBM'
             elif 'CatBoost' in type_name:
                 model_artifacts['type'] = 'CatBoost'
             elif 'RandomForest' in type_name:
                 model_artifacts['type'] = 'Random Forest'
             elif 'LinearRegression' in type_name:
                 model_artifacts['type'] = 'Linear Regression'
             else:
                 model_artifacts['type'] = type_name # Fallback to class name
             
        model_artifacts['scaler_X'] = joblib.load(os.path.join(model_path, "scaler_X.pkl"))
        model_artifacts['scaler_y'] = joblib.load(os.path.join(model_path, "scaler_y.pkl"))
        
        print(f"Model loaded successfully. Type: {model_artifacts['type']}")
        print(f"Loaded Metrics: {model_artifacts['metrics']}")

    except Exception as e:
        print(f"Failed to load model from Hopsworks: {e}")
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
            
        # Prepare metadata response
        metrics = model_artifacts.get('metrics', {})
        # Ensure we have defaults if specific keys missing
        
        return {
            "model_metadata": {
                "name": model_artifacts.get('name', 'AQI-Predictor'),
                "version": f"v{model_artifacts.get('version', '1.0')}",
                "type": model_artifacts.get('type', 'Unknown'),
                "metrics": metrics,
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
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
