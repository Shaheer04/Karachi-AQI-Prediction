import os
import os
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import argparse
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.utils import feature_engineering

# Load environment variables
load_dotenv()

HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def train_model():
    """
    Fetches data from Hopsworks, trains models, and saves the best one.
    """
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        print("Missing Hopsworks credentials.")
        return

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    try:
        # Fetch data from Feature Group v2 (Raw Data)
        print("Retrieving Feature View...")
        # We might need to create a Feature View if not using one, or just read FG directly for simplicity in this script
        # For robustness, reading FG directly and filtering in pandas is fine for this scale.
        fg = fs.get_feature_group(name="aqi_features_karachi", version=2)
        
        print("Fetching training data (Raw)...")
        df_raw = fg.read()
        
        # Sort by time to ensure lags are correct
        df_raw = df_raw.sort_values("datetime_utc")
        
        print("Applying dynamic Feature Engineering (Lags/Rolling)...")
        # Generate lags/rolling features on the fly
        df = feature_engineering(df_raw)
        
        # Drop initial rows with NaNs caused by lags (first 24h)
        print(f"Raw shape: {df_raw.shape}, Engineered shape: {df.shape}")
        df = df.dropna()
        print(f"Training shape (after dropna): {df.shape}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
            


    # Ensure datetime is in correct format
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Sort by time
    df = df.sort_values(by="datetime_utc")
    
    # --- Split & Scale ---
    from scripts.model_training import prepare_data, train_lightgbm, train_catboost, train_lstm
    
    data = prepare_data(df)
    
    metrics = {}
    models = {}
    
    # 1. LightGBM
    lgbm_model, lgbm_metrics = train_lightgbm(data)
    metrics['lightgbm'] = lgbm_metrics
    models['lightgbm'] = lgbm_model

    # 2. CatBoost
    cat_model, cat_metrics = train_catboost(data)
    metrics['catboost'] = cat_metrics
    models['catboost'] = cat_model

    # 3. LSTM
    lstm_model, lstm_metrics = train_lstm(data)
    metrics['lstm'] = lstm_metrics
    models['lstm'] = lstm_model

    # --- Selection ---
    # Select best model based on RMSE (Primary)
    best_model_name = min(metrics, key=lambda k: metrics[k]['rmse'])
    best_metrics = metrics[best_model_name]
    print(f"\nBest Model: {best_model_name} with RMSE: {best_metrics['rmse']:.4f}")
    
    best_model = models[best_model_name]

    # --- Register Best Model ---
    print("Registering Best Model to Hopsworks...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save Model and Scalers
    model_dir = "models"
    joblib.dump(data['scaler_X'], f"{model_dir}/scaler_X.pkl")
    joblib.dump(data['scaler_y'], f"{model_dir}/scaler_y.pkl")
    
    if best_model_name == 'lstm':
        best_model.save(f"{model_dir}/model.keras")
    else:
        joblib.dump(best_model, f"{model_dir}/model.pkl")

    mr = project.get_model_registry()
    
    # Create input example for schema
    input_example = data['X_train'].sample(1).to_dict(orient="records")[0]

    aqi_model = mr.python.create_model(
        name="aqi_predictor_best",
        # Log all metrics, but usage depends on get_best_model call
        metrics={
            "rmse": best_metrics['rmse'],
            "mae": best_metrics['mae'],
            "r2": best_metrics['r2']
        },
        description=f"Best Daily AQI Model. Type: {best_model_name}. RMSE: {best_metrics['rmse']:.4f}",
        input_example=input_example
    )
    
    # Upload the entire directory to ensure all artifacts (model, scalers) are included
    aqi_model.save(model_dir)
    
    print("Model successfully registered!")

if __name__ == "__main__":
    train_model()
