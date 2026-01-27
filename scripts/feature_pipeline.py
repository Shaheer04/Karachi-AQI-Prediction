import argparse
import sys
import os
import pandas as pd
from datetime import datetime, timezone
import hopsworks
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.utils import preprocess_raw
from scripts.data_fetching import get_aqi_history_data

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

def get_args():
    parser = argparse.ArgumentParser(description="Feature Pipeline for Karachi AQI")
    parser.add_argument("--days", type=float, default=0.05, help="Number of past days to fetch. Default ~1.2 hours (0.05 days). Use 730 for 2 years.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(f"Starting Feature Pipeline... Fetching last {args.days} days.")
    
    # 1. Extract
    # Calculate timestamps based on args
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - pd.Timedelta(days=args.days)).timestamp())
    
    df_raw = get_aqi_history_data(start=start_ts, end=end_ts)
    
    # 2. Transform available data
    if df_raw is not None and not df_raw.empty:
        # Just Preprocess (Units + AQI), NO LAGS
        df_processed = preprocess_raw(df_raw)
        
        # Select Raw Columns Only (No lags/rolling/sin/cos)
        # We store minimal state. Lags are computed on-the-fly by training/inference using history.
        features_to_keep = [
            'datetime_utc',
            'datetime_id',
            'pm2_5', 'pm10', 'no2', 'co', 'o3', 'so2', 'nh3',
            'calculated_aqi',
            'lat', 'lon'
        ]
        
        df_final = df_processed[features_to_keep].copy()
        
        # 3. Load
        print(f"Pushing {len(df_final)} rows to Hopsworks...")
        print(df_final.head())
        
        # Push to NEW Feature Group Version 2 (Raw Schema)
        if not HOPSWORKS_API_KEY:
             print("Error: HOPSWORKS_API_KEY missing.")
        else:
            try:
                project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
                fs = project.get_feature_store()
                
                aqi_fg = fs.get_or_create_feature_group(
                    name="aqi_features_karachi",
                    version=2,  # VERSION 2: Raw Data Only
                    primary_key=["datetime_id"], 
                    event_time="datetime_utc",
                    description="Raw AQI data for Karachi (Pollutants + Calculated AQI)",
                    online_enabled=True,
                    time_travel_format="HUDI"
                )
                
                # --- Deduplication Logic ---
                # Check if we have data overlapping with this batch
                start_check = int(df_final['datetime_id'].min())
                end_check = int(df_final['datetime_id'].max())
                
                try:
                    # Select only the ID column to minimize data transfer
                    existing_data_df = aqi_fg.select(["datetime_id"]).filter(
                        (aqi_fg.datetime_id >= start_check) & (aqi_fg.datetime_id <= end_check)
                    ).read()
                    
                    if not existing_data_df.empty:
                        existing_ids = set(existing_data_df['datetime_id'].astype(int))
                        # Filter out rows that truly exist
                        original_count = len(df_final)
                        df_final = df_final[~df_final['datetime_id'].isin(existing_ids)]
                        print(f"Deduplication: Removed {original_count - len(df_final)} existing rows. {len(df_final)} new rows to insert.")
                    else:
                        print("Deduplication: No existing data found in range.")
                        
                except Exception as e:
                    print(f"Warning: Could not check for duplicates (Feature Group might be empty or new): {e}")

                if not df_final.empty:
                    aqi_fg.insert(df_final, write_options={"wait_for_job": True})
                    print("Data successfully inserted/updated in Hopsworks (v2)!")
                else:
                    print("No new data to insert.")
            except Exception as e:
                print(f"Failed to push to Hopsworks: {e}")
            
    print("Pipeline Complete.")
