#!/usr/bin/env python3
"""Inspect model and scaler to see what features they expect."""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))

MODEL_DIR = "models"

print("=" * 70)
print("MODEL & SCALER INSPECTION")
print("=" * 70)

# Load model
if os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    print(f"\n✅ Model loaded: {type(model).__name__}")

    model_n_features = getattr(model, "n_features_in_", None)
    model_features = getattr(model, "feature_names_in_", None)

    if model_n_features:
        print(f"   n_features_in_: {model_n_features}")
    if model_features is not None:
        print(f"   feature_names_in_ ({len(model_features)}):")
        for i, feat in enumerate(model_features, 1):
            print(f"      {i:2d}. {feat}")
    else:
        print("   ⚠️ No feature_names_in_ attribute")
else:
    print("❌ model.pkl not found")

# Load scaler X
if os.path.exists(os.path.join(MODEL_DIR, "scaler_X.pkl")):
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    print(f"\n✅ Scaler X loaded: {type(scaler_X).__name__}")

    scaler_n_features = getattr(scaler_X, "n_features_in_", None)
    scaler_features = getattr(scaler_X, "feature_names_in_", None)

    if scaler_n_features:
        print(f"   n_features_in_: {scaler_n_features}")
    if scaler_features is not None:
        print(f"   feature_names_in_ ({len(scaler_features)}):")
        for i, feat in enumerate(scaler_features, 1):
            print(f"      {i:2d}. {feat}")
    else:
        print("   ⚠️ No feature_names_in_ attribute")
else:
    print("❌ scaler_X.pkl not found")

# Load scaler Y
if os.path.exists(os.path.join(MODEL_DIR, "scaler_y.pkl")):
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
    print(f"\n✅ Scaler Y loaded: {type(scaler_y).__name__}")
    print(f"   n_features_in_: {scaler_y.n_features_in_}")
else:
    print("❌ scaler_y.pkl not found")

print("\n" + "=" * 70)
