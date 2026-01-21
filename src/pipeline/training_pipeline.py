import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet
import shutil

# Load environment variables
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

def training_pipeline():
    if not HOPSWORKS_API_KEY:
        print("Error: HOPSWORKS_API_KEY not found.")
        return

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # 1. Retrieve Data
    print("Retrieving Feature View...")
    feature_view = None
    
    # 1. Try to get Feature View version 1
    try:
        feature_view = fs.get_feature_view(name="aqi_features_karachi", version=1)
    except Exception as e:
        print(f"Feature View retrieval failed (expected if first run): {e}")

    # 2. If it exists, we might want to ensure it's pointing to the fresh Feature Group.
    # However, deleting a Feature View is safer to ensure it picks up the recreated FG.
    if feature_view:
        print("Found existing Feature View version 1. Recreating to ensure consistency...")
        try:
            feature_view.delete()
            print("Deleted existing Feature View.")
            feature_view = None
        except Exception as e:
            print(f"Error deleting feature view: {e}")

    # 3. Create Feature View
    if feature_view is None:
        print("Creating Feature View version 1...")
        try:
            fg = fs.get_feature_group(name="aqi_features_karachi", version=1)
            query = fg.select_all()
            feature_view = fs.get_or_create_feature_view(
                name="aqi_features_karachi",
                version=1,
                description="Feature View for AQI prediction",
                labels=["calculated_aqi"],
                query=query
            )
        except Exception as e:
            print(f"Failed to create feature view: {e}")
            return
            

    print("Fetching training data...")
    # Fetch all data and split manually to ensure chronological order
    df, _ = feature_view.training_data(description="All data")
    
    # Sort by time
    df = df.sort_values(by="datetime_utc")
    
    # --- Split Data ---
    target_col = 'calculated_aqi'
    
    # Separate datetime for indexing
    datetime_col = df['datetime_utc']
    y = df[target_col]
    
    # Define features (exclude datetime, datetime_id, and target)
    feature_cols = [col for col in df.columns 
                    if col not in ['datetime_utc', 'datetime_id', target_col]]
    
    X = df[feature_cols]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Calculate split index (80-20 split)
    split_index = int(len(df) * 0.8)

    # Split chronologically
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    datetime_train = datetime_col.iloc[:split_index]
    datetime_test = datetime_col.iloc[split_index:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # --- Scaling ---
    print("Scaling Features and Target...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Identify columns to scale (exclude binary and cyclical if preferred, strictly following user snippet)
    columns_to_scale = [col for col in X_train.columns 
                        if not col.endswith('_sin')
                        and not col.endswith('_cos')]
    
    scaler_X.fit(X_train[columns_to_scale])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns_to_scale] = scaler_X.transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler_X.transform(X_test[columns_to_scale])

    # Scale target
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # y_test_scaled is not needed for training, we compare against raw y_test for MAE

    # --- Model Training ---
    metrics = {}
    models = {}

    # 1. LightGBM
    print("\nTraining LightGBM...")
    lgbm_model = lgb.LGBMRegressor(random_state=42)
    lgbm_model.fit(X_train_scaled, y_train_scaled)
    
    y_pred_lgbm_scaled = lgbm_model.predict(X_test_scaled)
    y_pred_lgbm = scaler_y.inverse_transform(y_pred_lgbm_scaled.reshape(-1, 1)).ravel()
    
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    metrics['lightgbm'] = mae_lgbm
    models['lightgbm'] = lgbm_model
    print(f"LightGBM MAE: {mae_lgbm}")

    # 2. CatBoost
    print("\nTraining CatBoost...")
    cat_model = CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    cat_model.fit(X_train_scaled, y_train_scaled)
    
    y_pred_cat_scaled = cat_model.predict(X_test_scaled)
    y_pred_cat = scaler_y.inverse_transform(y_pred_cat_scaled.reshape(-1, 1)).ravel()
    
    mae_cat = mean_absolute_error(y_test, y_pred_cat)
    metrics['catboost'] = mae_cat
    models['catboost'] = cat_model
    print(f"CatBoost MAE: {mae_cat}")

    # 3. Prophet
    print("\nTraining Prophet...")
    # Prepare data for Prophet: needs 'ds' and 'y'. We use unscaled target/features usually, 
    # but to be fair and use the same engineered features, we can add them as regressors.
    # Prophet is a bit different; strictly it's a time series model. 
    # Using the raw split data for Prophet as it handles its own scaling/seasonality usually.
    
    df_prophet_train = pd.DataFrame()
    df_prophet_train['ds'] = datetime_train.dt.tz_localize(None) # Prophet often prefers naive ts
    df_prophet_train['y'] = y_train.values

    # Adding regressors (features)
    for col in X_train.columns:
        df_prophet_train[col] = X_train[col].values

    prophet_model = Prophet()
    for col in X_train.columns:
        prophet_model.add_regressor(col)
        
    prophet_model.fit(df_prophet_train)

    # Make future dataframe for test
    df_prophet_test = pd.DataFrame()
    df_prophet_test['ds'] = datetime_test.dt.tz_localize(None)
    for col in X_test.columns:
        df_prophet_test[col] = X_test[col].values
        
    forecast = prophet_model.predict(df_prophet_test)
    y_pred_prophet = forecast['yhat'].values
    
    mae_prophet = mean_absolute_error(y_test, y_pred_prophet)
    metrics['prophet'] = mae_prophet
    models['prophet'] = prophet_model
    print(f"Prophet MAE: {mae_prophet}")

    # --- Selection ---
    best_model_name = min(metrics, key=metrics.get)
    best_mae = metrics[best_model_name]
    print(f"\nBest Model: {best_model_name} with MAE: {best_mae}")
    
    best_model = models[best_model_name]

    # --- Register Best Model ---
    print("Registering Best Model to Hopsworks...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save Model and Scalers
    model_dir = "models"
    joblib.dump(scaler_X, f"{model_dir}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{model_dir}/scaler_y.pkl")
    
    if best_model_name == 'prophet':
        # Prophet serialization is specific but pickle usually works
        joblib.dump(best_model, f"{model_dir}/model.pkl")
    else:
        joblib.dump(best_model, f"{model_dir}/model.pkl")

    mr = project.get_model_registry()
    
    # Create input example for schema
    input_example = X_train_scaled.sample(1).to_dict(orient="records")[0] if best_model_name != 'prophet' else df_prophet_train.head(1).to_dict(orient="records")[0]

    aqi_model = mr.python.create_model(
        name="aqi_predictor_best",
        metrics={"mae": best_mae},
        description=f"Best Daily AQI Model. Type: {best_model_name}",
        input_example=input_example
    )
    
    aqi_model.save(f"{model_dir}/model.pkl")
    aqi_model.save(f"{model_dir}/scaler_X.pkl")
    aqi_model.save(f"{model_dir}/scaler_y.pkl")
    
    print("Model successfully registered!")

if __name__ == "__main__":
    training_pipeline()
