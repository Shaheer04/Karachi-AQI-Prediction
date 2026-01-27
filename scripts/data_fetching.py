import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# Constants
LAT = 24.8607
LON = 67.0011
# Default start date if needed, but usually passed dynamically
START_DATE = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
END_DATE = int(datetime.now(timezone.utc).timestamp())

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
