import os
import time
import pandas as pd
import numpy as np
import joblib
import hopsworks
import tensorflow as tf
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from src.utils import feature_engineering

# Load environment variables
load_dotenv()

HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Constants ---
MODEL_DIR = "models_inference"

# --- Helper Logic (from prediction_service.py) ---
def estimate_pm25_from_aqi(aqi):
    """
    Estimates PM2.5 concentration (ug/m3) from AQI using inverted EPA breakpoints.
    """
    aqi = max(0, aqi)
    # Breakpoints: (I_lo, I_hi, C_lo, C_hi)
    breakpoints = [
        (0, 50, 0.0, 12.0),
        (51, 100, 12.1, 35.4),
        (101, 150, 35.5, 55.4),
        (151, 200, 55.5, 150.4),
        (201, 300, 150.5, 250.4),
        (301, 400, 250.5, 350.4),
        (401, 500, 350.5, 500.4)
    ]
    
    for (i_lo, i_hi, c_lo, c_hi) in breakpoints:
        if i_lo <= aqi <= i_hi:
            if i_hi == i_lo: return c_lo
            c = ((aqi - i_lo) * (c_hi - c_lo)) / (i_hi - i_lo) + c_lo
            return c
            
    # If beyond 500, extrapolate from last segment slope
    i_lo, i_hi, c_lo, c_hi = breakpoints[-1]
    c = ((aqi - i_lo) * (c_hi - c_lo)) / (i_hi - i_lo) + c_lo
    return c

class AQIPredictionService:
    def __init__(self, model_artifacts):
        self.model = model_artifacts['model']
        self.scaler_X = model_artifacts['scaler_X']
        self.scaler_y = model_artifacts['scaler_y']
        self.model_type = model_artifacts['type']

    def predict_future(self, history_df, steps=72):
        """
        Predicts future AQI for 'steps' hours using recursive autoregression.
        Optimized for latency by using list-based buffering.
        """
        # --- 1. Preparation ---
        # Ensure history is sorted and has proper types
        history_df = history_df.sort_values("datetime_utc")
        history_df['datetime_utc'] = pd.to_datetime(history_df['datetime_utc'])
        
        # Check if PM2.5 exists, else impute (simple forward fill or mean if totally missing, but we expect it)
        if 'pm2_5' not in history_df.columns:
            print("PM2.5 missing in history!")
            
        # Convert to list of dicts for O(1) appending and reading (much faster than DataFrame)
        buffer = history_df.to_dict('records')
        
        future_predictions = []
        
        # Last known row for persistence
        if not buffer:
            return []

        last_row = buffer[-1]
        last_dt = last_row['datetime_utc']
        
        print(f"Starting recursion from: {last_dt}")
        
        for i in range(steps):
            next_dt = last_dt + timedelta(hours=i+1)
            
            # --- 2. Seasonal Input (The "Guess") ---
            # We need the row from 24 hours ago to provide base values (NO2, PM10, etc)
            # and to serve as the source for 'lag_24' features.
            
            # Since we predict 1 hour at a time and append to buffer, 
            # the value 24 hours ago is simply at index `len(buffer) - 24`.
            if len(buffer) >= 24:
                hist_row = buffer[-24]
            else:
                hist_row = buffer[-1] # Fallback for very short history
            
            # Copy seasonal features (weather/pollutants form yesterday)
            next_row = hist_row.copy()
            next_row['datetime_utc'] = next_dt
            
            # --- 3. Incremental Feature Calculation ---
            # We calculate only what is needed for the model.
            
            # Time Features
            month = next_dt.month
            hour = next_dt.hour
            
            feat_hour_sin = np.sin(2 * np.pi * hour / 24)
            feat_hour_cos = np.cos(2 * np.pi * hour / 24)
            feat_month_sin = np.sin(2 * np.pi * month / 12)
            feat_month_cos = np.cos(2 * np.pi * month / 12)
            
            # Lags (24h)
            # We already fetched 'hist_row' which IS the 24h lag record.
            feat_pm25_lag_24 = hist_row['pm2_5']
            feat_aqi_lag_24 = hist_row.get('calculated_aqi', 0)
            
            # Rolling (6h) - Average of last 6 items in buffer
            rolling_window = buffer[-6:]
            feat_pm25_rolling_6h = np.mean([r['pm2_5'] for r in rolling_window])
            feat_aqi_6hr_avg = np.mean([r.get('calculated_aqi', 0) for r in rolling_window])
            
            # --- 4. Assemble Input Row ---
            # Create a localized dict for the model input
            input_features = next_row.copy() 
            
            # Update/Add engineered features
            input_features.update({
                'month': month,
                'hour': hour,
                'hour_sin': feat_hour_sin,
                'hour_cos': feat_hour_cos,
                'month_sin': feat_month_sin,
                'month_cos': feat_month_cos,
                'pm2_5_lag_24': feat_pm25_lag_24,
                'aqi_lag_24': feat_aqi_lag_24,
                'pm2_5_rolling_6h': feat_pm25_rolling_6h,
                'aqi_6hr_avg': feat_aqi_6hr_avg
            })
            
            # Create DataFrame for Scaler (1 row)
            df_input = pd.DataFrame([input_features])
            
            # --- 5. Scale & Predict ---
            cols_to_exclude = ['datetime_utc', 'datetime_id', 'calculated_aqi']
            X_cols = [c for c in df_input.columns if c not in cols_to_exclude]
            
            X = df_input[X_cols]
            
            # Scale
            cols_to_scale = [c for c in X.columns if not c.endswith('_sin') and not c.endswith('_cos')]
            
            X_scaled = X.copy().astype(float)
            try:
                # Fallback implementation: Assume columns match
                X_scaled[cols_to_scale] = self.scaler_X.transform(X[cols_to_scale])
                final_input = X_scaled
            except Exception as e:
                print(f"Scaling error at step {i}: {e}")
                break
                
            # Predict
            if self.model_type == 'lstm':
                X_val = final_input.values.reshape((1, 1, final_input.shape[1])).astype(np.float32)
                pred_scaled = self.model.predict(X_val, verbose=0)
                pred_val = self.scaler_y.inverse_transform(pred_scaled).ravel()[0]
            else:
                pred_scaled = self.model.predict(final_input)
                # Reshape for inverse transform if needed
                pred_val = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
            
            raw_pred = max(0, float(pred_val))
            
            # --- 6. Smoothing Strategy ---
            smoothing_factor = 0.7
            previous_val = last_row.get('calculated_aqi', raw_pred)
            smoothed_val = (smoothing_factor * raw_pred) + ((1 - smoothing_factor) * previous_val)
            pred_val = smoothed_val 
            
            # --- 7. Update Buffer ---
            est_pm25 = estimate_pm25_from_aqi(pred_val)
            
            input_features['calculated_aqi'] = pred_val
            input_features['pm2_5'] = est_pm25
            
            buffer.append(input_features)
            future_predictions.append({
                'datetime': next_dt,
                'predicted_aqi': pred_val,
                'pm2_5': est_pm25,
                'raw_aqi': raw_pred 
            })
            
            last_row = input_features
            
        return future_predictions

# --- Core Functions ---

def load_model_artifacts():
    """Downloads (if needed) and loads the best model and scalers from Hopsworks Model Registry."""
    model_artifacts = {}
    
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        print("Hopsworks credentials not found. Cannot load model.")
        return None

    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        # Get best model
        models = mr.get_models("aqi_predictor_best")
        if not models:
            raise Exception("No models found")
        
        # Sort by version
        best_model = max(models, key=lambda m: m.version)
        print(f"Loading model version: {best_model.version}")
        
        # --- Metrics Extraction ---
        try:
            metrics = best_model.training_metrics if best_model.training_metrics else {}
            if not metrics and hasattr(best_model, 'metrics'):
                metrics = best_model.metrics
            
            clean_metrics = {}
            if metrics:
                for k, v in metrics.items():
                    try:
                        if isinstance(v, (int, float)):
                            clean_metrics[k] = round(float(v), 4)
                        else:
                            clean_metrics[k] = v 
                    except:
                        clean_metrics[k] = str(v)
            else:
                clean_metrics = {"Status": "No metrics available"}
                
            model_artifacts['metrics'] = clean_metrics
        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
            model_artifacts['metrics'] = {"Error": "Failed to load metrics"}
        
        model_artifacts['name'] = best_model.name
        model_artifacts['version'] = best_model.version
        
        # Download
        model_path = best_model.download()
        
        # --- Model Loading ---
        if os.path.exists(os.path.join(model_path, "model.keras")):
             loaded_model = tf.keras.models.load_model(os.path.join(model_path, "model.keras"))
             model_artifacts['model'] = loaded_model
             model_artifacts['type'] = 'LSTM' 
        else:
             loaded_model = joblib.load(os.path.join(model_path, "model.pkl"))
             model_artifacts['model'] = loaded_model
             type_name = type(loaded_model).__name__
             if 'XGB' in type_name: model_artifacts['type'] = 'XGBoost'
             elif 'LGBM' in type_name: model_artifacts['type'] = 'LightGBM'
             elif 'CatBoost' in type_name: model_artifacts['type'] = 'CatBoost'
             else: model_artifacts['type'] = type_name
             
        model_artifacts['scaler_X'] = joblib.load(os.path.join(model_path, "scaler_X.pkl"))
        model_artifacts['scaler_y'] = joblib.load(os.path.join(model_path, "scaler_y.pkl"))
        
        print(f"Model loaded successfully. Type: {model_artifacts['type']}")
        return model_artifacts

    except Exception as e:
        print(f"Failed to load model from Hopsworks: {e}")
        return None

def fetch_recent_history():
    """Fetches last 3 days of history from Hopsworks Feature Store (v2 Raw)."""
    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features_karachi", version=2)
        
        now = datetime.now(timezone.utc)
        start_time = int((now - timedelta(days=3)).timestamp() * 1000)
        
        query = fg.select_all()
        print("Fetching recent history from Hopsworks Feature Store (v2)...")
        df_raw = query.filter(fg.datetime_id >= start_time).read()
        
        return df_raw
    except Exception as e:
        print(f"Failed to fetch history from Feature Store: {e}")
        return None

def get_predictions(model_artifacts):
    """Orchestrates the prediction process."""
    if not model_artifacts:
        return None

    try:
        # 1. Fetch History
        fs_df = fetch_recent_history()
        
        if fs_df is None or fs_df.empty or len(fs_df) < 24:
             print(f"Not enough history data (found {len(fs_df) if fs_df is not None else 0} rows).")
             # Try to return something if we have at least 1 row? 
             # For now, strict check like main.py
             if fs_df is None or len(fs_df) < 24:
                return None

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
            "model_metadata": {
                "name": model_artifacts.get('name', 'AQI-Predictor'),
                "version": f"v{model_artifacts.get('version', '1.0')}",
                "type": model_artifacts.get('type', 'Unknown'),
                "metrics": model_artifacts.get('metrics', {}),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "hourly_predictions": response_data,
            "daily_summary": daily_summary
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None
