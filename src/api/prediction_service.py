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
        """
        # Ensure FS history is sorted and has proper types
        history_df = history_df.sort_values("datetime_utc")
        history_df['datetime_utc'] = pd.to_datetime(history_df['datetime_utc'])
        
        # Check if PM2.5 exists, else impute
        if 'pm2_5' not in history_df.columns:
            print("PM2.5 missing in history, estimating from AQI...")
            history_df['pm2_5'] = history_df['calculated_aqi'].apply(estimate_pm25_from_aqi)

        # Initialize Buffer with history
        # We need to maintain all columns that feature_engineering expects + persistence cols
        buffer = history_df.to_dict('records')
        
        future_predictions = []
        
        # Last known row for persistence
        last_row = history_df.iloc[-1].to_dict()
        last_dt = last_row['datetime_utc']
        
        print(f"Starting recursion from: {last_dt}")

        for i in range(steps):
            next_dt = last_dt + pd.Timedelta(hours=i+1)
            
            # --- Prepare Input ---
            # Seasonal Persistence Strategy:
            # Instead of copying the last row (flat persistence), we look back 24 hours.
            # Air quality has strong daily seasonality (traffic, sun).
            # We try to find the row in the buffer that corresponds to next_dt - 24h.
            
            target_hist_dt = next_dt - timedelta(hours=24)
            hist_row = None
            
            # Search buffer for the historical row (optimized search could be done, linear is fine for small buffer)
            # Iterate backwards to find it quickly
            for item in reversed(buffer):
                if item['datetime_utc'] == target_hist_dt:
                    hist_row = item
                    break
            
            if hist_row:
                # Use seasonality for covariates
                next_row = hist_row.copy()
            else:
                # Fallback to simple persistence if 24h history missing (e.g., start of recursion)
                next_row = last_row.copy()

            next_row['datetime_utc'] = next_dt
            
            # Combine buffer + next_row
            # Optimization: Only need last ~30 hours of buffer for lags
            df_buffer = pd.DataFrame(buffer[-48:])
            df_combined = pd.concat([df_buffer, pd.DataFrame([next_row])], ignore_index=True)
            
            # --- Feature Engineering ---
            df_eng = feature_engineering(df_combined)
            features = df_eng.iloc[[-1]].copy() # Get features for the target row
            
            # --- Filter & Scale ---
            # Columns to exclude (non-features)
            cols_to_exclude = ['datetime_utc', 'datetime_id', 'calculated_aqi']
            X_cols = [c for c in features.columns if c not in cols_to_exclude]
            X = features[X_cols]
            
            # Identify scale columns (non-cyclic)
            cols_to_scale = [c for c in X.columns if not c.endswith('_sin') and not c.endswith('_cos')]
            
            X_scaled = X.copy().astype(float)
            try:
                X_scaled[cols_to_scale] = self.scaler_X.transform(X[cols_to_scale])
            except Exception as e:
                print(f"Scaling error at step {i}: {e}")
                # Fallback to last predicted value or break
                break
                
            # --- Predict ---
            if self.model_type == 'lstm':
                X_input = X_scaled.values.reshape((1, 1, X.shape[1])).astype(np.float32)
                # Note: LSTM input shape logic might need adjustment if model expected specific input dim
                # But here we reshape generically based on X.shape[1]
                pred_scaled = self.model.predict(X_input, verbose=0)
                pred_val = self.scaler_y.inverse_transform(pred_scaled).ravel()[0]
            else:
                pred_scaled = self.model.predict(X_scaled)
                pred_val = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
            
            pred_val = max(0, float(pred_val))
            
            # --- Update Buffer (Feedback) ---
            est_pm25 = estimate_pm25_from_aqi(pred_val)
            
            next_row['calculated_aqi'] = pred_val
            next_row['pm2_5'] = est_pm25
            
            buffer.append(next_row)
            future_predictions.append({
                'datetime': next_dt,
                'predicted_aqi': pred_val,
                'pm2_5': est_pm25
            })
            
            # Update last_row for next persistence
            last_row = next_row
            
        return future_predictions
