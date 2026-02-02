# ⚡ Karachi AQI Predictor

> **Air Quality Forecasting using Hybrid ML Models & Recursive Autoregression**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Hopsworks](https://img.shields.io/badge/Feature_Store-Hopsworks-green.svg)
![Models](https://img.shields.io/badge/Models-LightGBM%20|%20CatBoost%20|%20LSTM-orange.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)

## 📖 Project Overview
This project implements a robust, automated pipeline to predict the **Air Quality Index (AQI)** for Karachi, Pakistan, up to **72 hours** into the future. It leverages a serverless architecture with **Hopsworks** for feature management and model registry, ensuring reproducible and scalable ML workflows.

The system is designed to provide actionable insights into air quality trends using historical pollutant data (`PM2.5`, `PM10`, `NO2`, etc.) and meteorological factors.

---

## 🛠 Methodology

### 1. 🔄 Automated Data Pipeline (Feature Store)
We utilize **Hopsworks Feature Store** to manage our data lifecycle.
- **Raw Data Ingestion**: Historical AQI data is fetched from OpenWeatherMap API.
- **Automated Engineering**: A scheduled pipeline processes raw data into features and stores them in **Feature Groups (v2)**. This ensures that training and inference always use consistent feature definitions.

### 2. 🎯 Feature Selection via SHAP Analysis
To maximize model performance and interpretability, we employed **SHAP (SHapley Additive exPlanations)** analysis to identify the most critical drivers of AQI.
- **Key Determinants**: The analysis revealed that `pm2_5_lag_24` (24-hour lag) and `pm2_5_rolling_6h` (recent trend) are the strongest predictors.
- **Selected Features**: Based on this, we strictly filtered our feature set to:
    - **Lags**: `pm2_5_lag_24`, `aqi_lag_24` (Captures daily seasonality)
    - **Trends**: `pm2_5_rolling_6h`, `aqi_6hr_avg` (Captures immediate shifts)
    - **Cyclic Time**: `hour_sin`/`cos`, `month_sin`/`cos` (Encodes temporal patterns without discontinuities)

### 3. 🛡️ Overfitting Prevention & Robust Validation
To ensure the model generalizes well to unseen future data, we implement strict validation checks:
- **Train vs. Test Gap**: The pipeline calculates RMSE for both training and test sets. If the gap exceeds a threshold (indicating memorization), the model is flagged with an **Overfitting Warning**.
- **Time Series Cross-Validation**: We use `TimeSeriesSplit` to validate model stability across multiple historical windows, ensuring it doesn't just perform well on a single arbitrary split.

### 4. 🔮 72-Hour Recursive Prediction Strategy
Predicting multi-step time series can be challenging. We implemented a **Recursive Autoregressive Strategy** with **Seasonal Persistence** to forecast 72 hours ahead:
1.  **Step-by-Step**: The model predicts $t+1$, feeds that prediction back as history for $t+2$, and repeats for 72 steps.
2.  **Seasonal Context**: Instead of simple flat persistence, the recursion logic looks back **24 hours** in the prediction buffer to carry over daily seasonal patterns (e.g., traffic peaks), ensuring the forecast remains realistic over the 3-day horizon.

---

## 🤖 Model Training & Selection

We train multiple state-of-the-art models in parallel to ensure the best performance for current conditions.

### Hybrid Training Pipeline
The automated training script trains three distinct model architectures:
1.  **LightGBM**: Highly efficient gradient boosting decision tree.
2.  **CatBoost**: Gradient boosting with advanced handling of numerical features.
3.  **LSTM (Long Short-Term Memory)**: A recurrent neural network (built with TensorFlow/Keras) capturing complex temporal dependencies.

### ✅ Automated Selection (Why RMSE?)
The pipeline automatically evaluates all trained models and promotes the **best performer** to the Model Registry.
- **Primary Metric**: **RMSE (Root Mean Squared Error)**
- **Mathematical Rational**:
  $$ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} $$
  We chose RMSE over MAE because the squaring term $(y_i - \hat{y}_i)^2$ disproportionately penalizes **large errors**. In air quality forecasting, missing an extreme pollution spike (outlier) poses a much higher health risk than small variances in minimal pollution. Therefore, our system prioritizes models that minimize these critical large deviations.

---

## 💻 Tech Stack

- **Orchestration & MLOps**: [Hopsworks](https://www.hopsworks.ai/) (Feature Store, Model Registry)
- **Modeling**: `scikit-learn`, `LightGBM`, `CatBoost`, `TensorFlow`
- **API**: `FastAPI` (Inference Endpoint)
- **Frontend**: `Streamlit` (Interactive Dashboard)
- **Visualization**: `Plotly`, `SHAP`

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Hopsworks API Key

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
1.  **Feature Pipeline** (Backfill/Update data):
    ```bash
    python scripts/feature_pipeline.py --days 1
    ```
2.  **Training Pipeline** (Train & Register Best Model):
    ```bash
    python scripts/training_pipeline.py
    ```
3.  **Start Backend API**:
    ```bash
    uvicorn src.api.main:app --reload
    ```
4.  **Launch App**:
    ```bash
    streamlit run src/frontend/app.py
    ```
