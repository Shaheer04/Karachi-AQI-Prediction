import os
import requests
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

LAT = 24.8607
LON = 67.0011
START_DATE = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
END_DATE = int(datetime.now(timezone.utc).timestamp())
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

def get_aqi_history_data(lat=LAT, lon=LON, start=START_DATE, end=END_DATE, api_key=API_KEY):
    """
    Fetches historical air pollution data from OpenWeatherMap.
    """
    if not api_key:
        print("Error: 'OPENWEATHER_API_KEY' not found in environment variables.")
        return None

    params = {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "appid": api_key
    }

    print(f"Fetching OWM Air Pollution history for Lat: {lat}, Lon: {lon}...")
    print(f"Time Range: {datetime.fromtimestamp(start, timezone.utc)} to {datetime.fromtimestamp(end, timezone.utc)}")

    try:
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            return None

        data = response.json()
        raw_list = data.get('list', [])
        
        print(f"Retrieved {len(raw_list)} data points.")

        parsed_data = []
        for item in raw_list:
            dt = item.get('dt')
            components = item.get('components', {})
            main = item.get('main', {}) # contains 'aqi'

            # OWM returns data in UNIX timestamp
            timestamp = datetime.fromtimestamp(dt, timezone.utc)
            
            record = {
                'datetime_utc': timestamp,
                'aqi_owm': main.get('aqi'),
                'co': components.get('co'),
                'no': components.get('no'),
                'no2': components.get('no2'),
                'o3': components.get('o3'),
                'so2': components.get('so2'),
                'pm2_5': components.get('pm2_5'),
                'pm10': components.get('pm10'),
                'nh3': components.get('nh3'),
                'lat': lat,
                'lon': lon
            }
            parsed_data.append(record)

        if parsed_data:
            df = pd.DataFrame(parsed_data)
            return df
        else:
            print("No data found in response.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return None

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

def preprocess_data(df):
    """
    Applies transformations and feature engineering to the raw dataframe.
    """
    if df is None or df.empty:
        return None
    
    # Copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Convert pollutants from μg/m³ to required units for AQI calculation
    df['co'] = df['co'] / 1145  # CO: μg/m³ to ppm
    df['no2'] = df['no2'] * 0.532  # NO2: μg/m³ to ppb
    df['o3'] = df['o3'] * 0.510  # O3: μg/m³ to ppb
    df['so2'] = df['so2'] * 0.382  # SO2: μg/m³ to ppb

    # Calculate standard AQI
    print("Calculating Standard AQI...")
    df['calculated_aqi'] = df.apply(calculate_aqi_all_pollutants, axis=1)

    # Convert timestamp to datetime if not already
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Feature Engineering
    print("Generating Features...")
    df['month'] = df['datetime_utc'].dt.month
    df['hour'] = df['datetime_utc'].dt.hour
    
    # Using Unix timestamp in milliseconds
    df['datetime_id'] = df['datetime_utc'].astype('int64') // 10**6
    
    # 1. Create a 24-hour lag for PM2.5 (The "Predictor")
    df['pm2_5_lag_24'] = df['pm2_5'].shift(24)
    
    # 2. Create a 24-hour lag for AQI (The "Anchor")
    df['aqi_lag_24'] = df['calculated_aqi'].shift(24)
    
    # 3. Create a 6-hour rolling average of PM2.5 (The "Trend")
    # Shift by 1 first to prevent data leakage if prediction is for next hour
    df['pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(window=6).mean()
    
    df['aqi_6hr_avg'] = df['calculated_aqi'].shift(1).rolling(window=6).mean()

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    features_to_keep = [
        'datetime_utc',
        'datetime_id',
        'pm2_5_lag_24', 
        'aqi_lag_24', 
        'pm2_5_rolling_6h', 
        'aqi_6hr_avg', 
        'hour_sin', 
        'hour_cos', 
        'month_sin', 
        'month_cos', 
        'calculated_aqi'
    ]
    
    # Filter columns
    df_processed = df[features_to_keep].copy()
    
    # Drop NaNs created by lagging/rolling
    initial_len = len(df_processed)
    df_processed.dropna(inplace=True)
    print(f"Dropped {initial_len - len(df_processed)} rows due to NaNs from feature engineering.")
    
    # Ensure datetime is compatible with Hopsworks (often requires unix epoch in ms or datetime object)
    # Hopsworks event_time expects datetime or int(ms)
    
    return df_processed

def push_to_hopsworks(df):
    """
    Pushes the processed dataframe to Hopsworks Feature Store.
    """
    if df is None or df.empty:
        print("No data to push to Hopsworks.")
        return

    if not HOPSWORKS_API_KEY:
        print("Error: 'HOPSWORKS_API_KEY' not set.")
        return

    print("Connecting to Hopsworks...")
    try:
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT_NAME,
            api_key_value=HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store()
        
        # Try to delete version 1 if it exists to ensure a clean rewrite
        try:
            print("Attempting to delete existing Feature Group version 1 (if any)...")
            fg = fs.get_feature_group(name="aqi_features_karachi", version=1)
            fg.delete()
            print("Deleted existing Feature Group version 1.")
        except:
            print("No existing Feature Group version 1 found (or delete failed). Proceeding...")

        print("Getting or Creating Feature Group...")
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_features_karachi",
            version=1,
            primary_key=["datetime_id"], 
            event_time="datetime_utc",
            description="AQI data for Karachi with engineered features",
            online_enabled=True
        )
        
        print("Inserting data into Feature Group...")
        aqi_fg.insert(df, write_options={"wait_for_job": True})
        print("Data successfully inserted into Hopsworks!")

    except Exception as e:
        print(f"Failed to push to Hopsworks: {e}")

if __name__ == "__main__":
    print("Starting Feature Pipeline...")
    
    # 1. Extract
    df_raw = get_aqi_history_data()
    
    # 2. Transform
    if df_raw is not None:
        df_processed = preprocess_data(df_raw)
        
        # 3. Load
        if df_processed is not None:
            # Temporary: print head to verify
            print("Processed Data Head:")
            print(df_processed.head())
            
            push_to_hopsworks(df_processed)
    
    print("Pipeline Complete.")
