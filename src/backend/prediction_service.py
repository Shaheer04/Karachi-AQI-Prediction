import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils import feature_engineering

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
        
        # Check if PM2.5 exists, else impute
        if 'pm2_5' not in history_df.columns:
            print("PM2.5 missing in history!")
            
        # Convert to list of dicts for O(1) appending and reading (much faster than DataFrame)
        buffer = history_df.to_dict('records')
        
        future_predictions = []
        
        # Last known row for persistence
        last_row = buffer[-1]
        last_dt = last_row['datetime_utc']
        
        print(f"Starting recursion from: {last_dt}")
        print(f"--- DEBUG: Prediction Start Context ---")
        print(f"Initial PM2.5: {last_row.get('pm2_5')}")
        print(f"Initial AQI:   {last_row.get('calculated_aqi')}")
        print("---------------------------------------")

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
            # Note: The buffer contains [..., T-2, T-1]. The 'shift(1)' in pandas means 
            # we simply take the last 6 items clearly.
            rolling_window = buffer[-6:]
            feat_pm25_rolling_6h = np.mean([r['pm2_5'] for r in rolling_window])
            feat_aqi_6hr_avg = np.mean([r.get('calculated_aqi', 0) for r in rolling_window])
            
            # --- 4. Assemble Input Row ---
            # We put these into a single-row DataFrame to ensure Scaler column alignment.
            # Base 'next_row' has the "seasonal" pollutant values (pm2_5, no2, etc).
            
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
            # Blend the model's prediction with the last known value.
            # Restored to 0.3 to prevent jumps.
            
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
