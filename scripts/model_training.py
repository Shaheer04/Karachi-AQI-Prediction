import math
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

def calculate_metrics(y_true, y_pred, model_name):
    """Calculates MAE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def prepare_data(df, target_col='calculated_aqi', split_ratio=0.8):
    """
    Splits data into train/test sets and scales features.
    """
    # Separate features and target
    feature_cols = [col for col in df.columns 
                    if col not in ['datetime_utc', 'datetime_id', target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split chronologically
    split_index = int(len(df) * split_ratio)
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Scale only non-cyclic features
    columns_to_scale = [col for col in X_train.columns 
                       if not col.endswith('_sin') and not col.endswith('_cos')]
    
    scaler_X.fit(X_train[columns_to_scale])
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columns_to_scale] = scaler_X.transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler_X.transform(X_test[columns_to_scale])
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'scaler_X': scaler_X, 'scaler_y': scaler_y
    }

def train_lightgbm(data):
    print("\nTraining LightGBM...")
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(data['X_train_scaled'], data['y_train_scaled'])
    
    y_pred_scaled = model.predict(data['X_test_scaled'])
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    metrics = calculate_metrics(data['y_test'], y_pred, 'LightGBM')
    return model, metrics

def train_catboost(data):
    print("\nTraining CatBoost...")
    model = CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
    model.fit(data['X_train_scaled'], data['y_train_scaled'])
    
    y_pred_scaled = model.predict(data['X_test_scaled'])
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    metrics = calculate_metrics(data['y_test'], y_pred, 'CatBoost')
    return model, metrics

def train_lstm(data):
    print("\nTraining LSTM...")
    X_train = data['X_train_scaled'].values
    X_test = data['X_test_scaled'].values
    
    # Reshape (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_train_lstm, data['y_train_scaled'], epochs=50, batch_size=32, verbose=0)
    
    y_pred_scaled = model.predict(X_test_lstm)
    y_pred = data['scaler_y'].inverse_transform(y_pred_scaled).ravel()
    
    metrics = calculate_metrics(data['y_test'], y_pred, 'LSTM')
    return model, metrics
