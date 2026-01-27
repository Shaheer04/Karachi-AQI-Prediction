import pandas as pd
import numpy as np

def calculate_aqi_all_pollutants(row):
    """
    Calculate AQI based on all pollutants and return the maximum (worst) AQI.
    Returns value in 0-500 range.
    Uses converted pollutant values (ppm/ppb).
    """
    
    # PM2.5 breakpoints (μg/m³)
    bp_pm25 = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    # PM10 breakpoints (μg/m³)
    bp_pm10 = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]
    
    # CO breakpoints (ppm)
    bp_co = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500)
    ]
    
    # NO2 breakpoints (ppb)
    bp_no2 = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500)
    ]
    
    # O3 breakpoints (ppb) - 8hr average
    bp_o3 = [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
        (405, 504, 301, 400),
        (505, 604, 401, 500)
    ]
    
    # SO2 breakpoints (ppb) - 1hr average
    bp_so2 = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500)
    ]
    
    def get_aqi_subindex(c, breakpoints):
        """Calculate AQI sub-index for a given concentration"""
        if pd.isna(c):
            return 0
        
        for (c_lo, c_hi, i_lo, i_hi) in breakpoints:
            if c_lo <= c <= c_hi:
                return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo
        
        # Handle values beyond max breakpoint
        if c > breakpoints[-1][1]:
            return 500
        return 0
    
    # Calculate sub-indices using converted pollutant columns
    aqi_pm25 = get_aqi_subindex(row['pm2_5'], bp_pm25)
    aqi_pm10 = get_aqi_subindex(row['pm10'], bp_pm10)
    aqi_co = get_aqi_subindex(row['co'], bp_co)
    aqi_no2 = get_aqi_subindex(row['no2'], bp_no2)
    aqi_o3 = get_aqi_subindex(row['o3'], bp_o3)
    aqi_so2 = get_aqi_subindex(row['so2'], bp_so2)
    
    # Return the maximum AQI (worst pollutant determines overall AQI)
    return round(max(aqi_pm25, aqi_pm10, aqi_co, aqi_no2, aqi_o3, aqi_so2))

def preprocess_raw(df):
    """
    Applies basic transformations (Unit Conversion, AQI Calc) to raw OWM data.
    Input: DataFrame with raw OWM columns (pm2_5, no2, etc. in ug/m3).
    Output: DataFrame with converted units (ppm/ppb) and calculated_aqi.
    """
    df = df.copy()
    
    # Ensure datetime is compatible
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Unit Conversions
    # co: μg/m³ to ppm (divide by 1145)
    # no2, o3, so2: μg/m³ to ppb
    df['co'] = df['co'] / 1145 
    df['no2'] = df['no2'] * 0.532
    df['o3'] = df['o3'] * 0.510
    df['so2'] = df['so2'] * 0.382

    # Calculate standard AQI
    df['calculated_aqi'] = df.apply(calculate_aqi_all_pollutants, axis=1)
    
    # Basic Time Features needed for Feature Store indexing
    df['datetime_id'] = df['datetime_utc'].astype('int64') // 10**6
    
    return df

def feature_engineering(df):
    """
    Generates lag and rolling features from preprocessed (unit-converted) data.
    Input: DataFrame with 'calculated_aqi' and converted pollutants.
    Output: DataFrame with 'pm2_5_lag_24', 'aqi_rolling', etc.
    """
    df = df.copy()
    
    # Ensure datetime sorted for lags
    df = df.sort_values('datetime_utc')
    
    # Time Features
    df['month'] = df['datetime_utc'].dt.month
    df['hour'] = df['datetime_utc'].dt.hour
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lags & Rolling
    # 1. 24-hour lag
    df['pm2_5_lag_24'] = df['pm2_5'].shift(24)
    df['aqi_lag_24'] = df['calculated_aqi'].shift(24)
    
    # 2. 6-hour rolling average (shifted by 1 to represent previous context)
    df['pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(window=6).mean()
    df['aqi_6hr_avg'] = df['calculated_aqi'].shift(1).rolling(window=6).mean()

    return df
