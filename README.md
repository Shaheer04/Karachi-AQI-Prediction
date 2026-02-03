# ⚡ Karachi AQI Predictor

> **Air Quality Forecasting using Hybrid ML Models & Recursive Autoregression**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Hopsworks](https://img.shields.io/badge/Feature_Store-Hopsworks-green.svg)
![Models](https://img.shields.io/badge/Models-LightGBM%20|%20CatBoost%20|%20LSTM-orange.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)

## 📖 Project Overview
I implemented this robust, automated pipeline to predict the **Air Quality Index (AQI)** for Karachi, Pakistan, up to **72 hours** into the future. It leverages a serverless architecture with **Hopsworks** for feature management and model registry, ensuring reproducible and scalable ML workflows.

The system is designed to provide actionable insights into air quality trends using historical pollutant data (`PM2.5`, `PM10`, `NO2`, etc.) and meteorological factors.

---

## 🛠 Methodology

### 1. 🔄 Automated Data Pipeline (Feature Store)
I utilize **Hopsworks Feature Store** to manage the data lifecycle.
- **Raw Data Ingestion**: Historical AQI data is fetched from OpenWeatherMap API.
- **Automated Engineering**: A scheduled pipeline processes raw data into features and stores them in **Feature Groups (v2)**. This ensures that training and inference always use consistent feature definitions.

### 2. 🎯 Feature Selection via SHAP Analysis
To maximize model performance and interpretability, I employed **SHAP (SHapley Additive exPlanations)** analysis to identify the most critical drivers of AQI.
- **Key Determinants**: The analysis revealed that `pm2_5_lag_24` (24-hour lag) and `pm2_5_rolling_6h` (recent trend) are the strongest predictors.
- **Selected Features**: Based on this, I strictly filtered the feature set to:
    - **Lags**: `pm2_5_lag_24`, `aqi_lag_24` (Captures daily seasonality)
    - **Trends**: `pm2_5_rolling_6h`, `aqi_6hr_avg` (Captures immediate shifts)
    - **Cyclic Time**: `hour_sin`/`cos`, `month_sin`/`cos` (Encodes temporal patterns without discontinuities)

### 3. 🛡️ Overfitting Prevention & Robust Validation
To ensure the model generalizes well to unseen future data, I implement strict validation checks:
- **Train vs. Test Gap**: The pipeline calculates RMSE for both training and test sets. If the gap exceeds a threshold (indicating memorization), the model is flagged with an **Overfitting Warning**.
- **Time Series Cross-Validation**: I use `TimeSeriesSplit` to validate model stability across multiple historical windows, ensuring it doesn't just perform well on a single arbitrary split.

### 4. 🔮 72-Hour Recursive Prediction Strategy
Predicting multi-step time series can be challenging. I implemented a **Recursive Autoregressive Strategy** with **Seasonal Persistence** to forecast 72 hours ahead:
1.  **Step-by-Step**: The model predicts $t+1$, feeds that prediction back as history for $t+2$, and repeats for 72 steps.
2.  **Seasonal Context**: Instead of simple flat persistence, the recursion logic looks back **24 hours** in the prediction buffer to carry over daily seasonal patterns (e.g., traffic peaks), ensuring the forecast remains realistic over the 3-day horizon.

---

## 🤖 Model Training & Selection

I train multiple state-of-the-art models in parallel to ensure the best performance for current conditions.

### Hybrid Training Pipeline
The automated training script trains three distinct model architectures:
1.  **LightGBM**: Highly efficient gradient boosting decision tree.
2.  **CatBoost**: Gradient boosting with advanced handling of numerical features.
3.  **LSTM (Long Short-Term Memory)**: A recurrent neural network (built with TensorFlow/Keras) capturing complex temporal dependencies.

### ✅ Automated Selection (Why RMSE?)
The pipeline automatically evaluates all trained models and promotes the **best performer** to the Model Registry.
- **Primary Metric**: **RMSE (Root Mean Squared Error)**
- **Mathematical Rational**:
  I chose RMSE over MAE because the squaring term $(y_i - \hat{y}_i)^2$ disproportionately penalizes **large errors**. In air quality forecasting, missing an extreme pollution spike (outlier) poses a much higher health risk than small variances in minimal pollution. Therefore, the system prioritizes models that minimize these critical large deviations.

---

## 💻 Tech Stack

- **Orchestration & MLOps**: [Hopsworks](https://www.hopsworks.ai/) (Feature Store, Model Registry)
- **Modeling**: `scikit-learn`, `LightGBM`, `CatBoost`, `TensorFlow`
- **API**: `FastAPI` (Inference Endpoint)
- **Frontend**: `Streamlit` (Interactive Dashboard)
- **Visualization**: `Plotly`, `SHAP`

---

## 🚀 Getting Started (Local Setup)

If you want to clone and run this project locally, follow these steps.

### Prerequisites
- **Python 3.11+**
- **Git**
- **Hopsworks Account**: You need an API key from [Hopsworks](https://www.hopsworks.ai/).

### 1. Clone the Repository
```bash
git clone https://github.com/Shaheer04/Karachi-AQI-Prediction.git
cd Karachi-AQI-Prediction
```

### 2. Set Up Virtual Environment

I recommend using **[uv](https://github.com/astral-sh/uv)** for fast dependency management, but standard `pip` works too.

#### Option A: Using `uv` (Recommended)
```bash
# Install uv if you haven't
pip install uv

# Create virtual environment and sync dependencies
uv sync
```

#### Option B: Using standard `pip`
```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # on Linux/Mac
# .venv\Scripts\activate   # on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
You must set up your environment variables for the code to access the Hopsworks Feature Store.

1. Create a file named `.env` in the root directory.
2. Add your Hopsworks API key:

```ini
HOPSWORKS_API_KEY=your_api_key_here
```

### 4. Running the System

You can run different parts of the pipeline manually.

#### A. Feature Pipeline
Fetches fresh data and updates the Feature Store.
```bash
python scripts/feature_pipeline.py --days 1
```

#### B. Training Pipeline
Trains models, evaluates them, and registers the best one.
```bash
python scripts/training_pipeline.py
```

#### C. Backend API
Starts the local inference server.
```bash
uvicorn src.api.main:app --reload
```
*API will be available at `http://localhost:8000`*

#### D. Frontend Dashboard
Launches the Streamlit app.
```bash
streamlit run src/frontend/app.py
```
*App will open at `http://localhost:8501`*
