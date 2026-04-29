#!/usr/bin/env python3
"""Debug script to test the prediction pipeline."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.core import (
    load_local_model,
    fetch_recent_history,
    get_predictions,
    generate_synthetic_history,
    AQIPredictionService,
)

print("=" * 60)
print("PREDICTION PIPELINE DEBUG")
print("=" * 60)

# Step 1: Load model
print("\n1️⃣ Loading model...")
artifacts = load_local_model()
if not artifacts:
    print("❌ Failed to load local model")
    sys.exit(1)
print(f"✅ Model loaded: {artifacts.get('type')}")
print(f"   - Model: {type(artifacts['model'])}")
print(f"   - ScalerX n_features: {artifacts['scaler_X'].n_features_in_}")
print(f"   - ScalerY n_features: {artifacts['scaler_y'].n_features_in_}")

# Step 2: Fetch history
print("\n2️⃣ Fetching history...")
history = fetch_recent_history()
if history is None or history.empty:
    print("❌ No history fetched, will use synthetic data")
    history = generate_synthetic_history(hours=72)
else:
    print(f"✅ History fetched: {history.shape} rows")

if history is None or len(history) < 24:
    print("❌ Not enough history data")
    sys.exit(1)

print(f"✅ History available: {len(history)} rows")
print(f"   - Columns: {list(history.columns)}")
print(
    f"   - Date range: {history['datetime_utc'].min()} to {history['datetime_utc'].max()}"
)

# Step 3: Try prediction
print("\n3️⃣ Running predictions...")
try:
    service = AQIPredictionService(artifacts)
    predictions = service.predict_future(history, steps=72)

    if predictions:
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"   - First prediction: {predictions[0]}")
        print(f"   - Last prediction: {predictions[-1]}")
    else:
        print("❌ Prediction service returned empty list")
        sys.exit(1)
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 4: Full end-to-end
print("\n4️⃣ Full end-to-end test...")
try:
    result = get_predictions(artifacts)
    if result:
        print(f"✅ Full pipeline successful")
        print(
            f"   - Model: {result['model_metadata']['name']} v{result['model_metadata']['version']}"
        )
        print(f"   - Type: {result['model_metadata']['type']}")
        print(f"   - Hourly predictions: {len(result['hourly_predictions'])}")
        print(f"   - Daily summaries: {len(result['daily_summary'])}")
        if result["hourly_predictions"]:
            print(
                f"   - Current AQI: {result['hourly_predictions'][0]['predicted_aqi']}"
            )
    else:
        print("❌ get_predictions returned None")
        sys.exit(1)
except Exception as e:
    print(f"❌ Full pipeline failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
