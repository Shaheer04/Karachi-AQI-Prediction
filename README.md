# ⚡ Karachi AQI Predictor

> **Air Quality Forecasting using Hybrid ML Models & Recursive Autoregression**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Hopsworks](https://img.shields.io/badge/Feature_Store-Hopsworks-green.svg)
![Models](https://img.shields.io/badge/Models-LightGBM%20|%20CatBoost%20|%20LSTM-orange.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)

## 📖 Project Overview
This project implements a robust, automated pipeline to predict the **Air Quality Index (AQI)** for Karachi, Pakistan, up to **72 hours** into the future. It utilizes a **Serverless Architecture** with **Hopsworks** for feature management and model registry, ensuring reproducible and scalable ML workflows.

The system is designed to provide actionable insights into air quality trends using historical pollutant data (`PM2.5`, `PM10`, `NO2`, etc.) and meteorological factors.

---

## 🛠 Methodology & Key Challenges

### 1. 🔄 Automated Data Pipeline (Feature Store)
I utilize **Hopsworks Feature Store** to manage the data lifecycle.
- **Raw Data Ingestion**: Historical AQI data is fetched from OpenWeatherMap API and stored in **Feature Groups (v2)**.
- **Automated Engineering**: A scheduled pipeline processes raw data into engineered features (Lags, Rolling Averages, Cyclical Time) on-the-fly, ensuring consistency between training and inference.

### 2. 🛡️ Overfitting Prevention Strategy
To ensure the model generalizes well to unseen future data, I implement a strict **Gap-Based Overfitting Check**:
- **Train vs. Test RMSE Gap**: The pipeline calculates the absolute difference between Train RMSE and Test RMSE.
- **Strict Threshold**: If the gap exceeds **10 units**, the model is flagged with an **Overfitting Warning** and deprioritized during selection. This prevents "memorization" of historical data.

### 3. 📉 Challenge: Error Accumulation in Forecasting
**Problem**: A standard 72-step recursive prediction often degrades into a "flat line" (converging to the mean) because errors compound at each step.
**Solution**:
- **Seasonal Persistence (Anchoring)**: I implemented a mechanism that looks back **24 hours** in the prediction buffer during recursion.
- **Context Injection**: By feeding the `pm2_5` value from the previous day (t-24h) as a lag feature into the model at each step, the forecast maintains realistic **diurnal seasonality** (morning peaks/evening lows) throughout the 3-day horizon.

### 4. ⚡ Challenge: Serverless Execution Limits
**Problem**: GitHub Actions and other serverless runners have strict execution time limits. Deep Learning models (LSTM) can easily time out.
**Solution**:
- **Efficient Architectures**: The pipeline prioritizes fast, gradient-boosting models (**LightGBM**, **CatBoost**) which train in seconds on this dataset.
- **Sequential Execution**: Models are trained sequentially in a single job to avoid complex orchestration overheads, ensuring the entire pipeline (Fetch -> Train -> Register) completes well within standard timeout limits.

---

## 🤖 Model Training & Selection

I train multiple state-of-the-art models to ensure the best performance for current conditions.

### Hybrid Training Pipeline
The automated training script trains three distinct model architectures:
1.  **LightGBM**: Highly efficient gradient boosting decision tree.
2.  **CatBoost**: Gradient boosting with advanced handling of numerical features.
3.  **LSTM**: A recurrent neural network capturing complex temporal dependencies (TensorFlow/Keras).

### ✅ Automated Selection (Why RMSE?)
The pipeline automatically evaluates all trained models and promotes the **best performer** to the Model Registry.
- **Primary Metric**: **RMSE (Root Mean Squared Error)**
- **Mathematical Rational**:
  I chose RMSE over MAE because the squaring term $(y_i - \hat{y}_i)^2$ disproportionately penalizes **large errors**. In air quality forecasting, missing an extreme pollution spike (outlier) poses a much higher health risk than small variances in minimal pollution. Therefore, the system prioritizes models that minimize these critical large deviations.

---

## 💻 Tech Stack

- **Orchestration & MLOps**: [Hopsworks](https://www.hopsworks.ai/) (Feature Store, Model Registry)
- **Modeling**: `scikit-learn`, `LightGBM`, `CatBoost`, `TensorFlow`
- **Frontend**: `Streamlit` (Interactive Dashboard & Inference)
- **Visualization**: `Plotly`, `SHAP` (Offline Analysis)

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
HOPSWORKS_PROJECT_NAME=your_project_name
```

### 4. Running the System

You can run different parts of the pipeline manually.

#### A. Feature Pipeline
Fetches fresh data and updates the Feature Store.
```bash
python scripts/feature_pipeline.py --days 1
```

#### B. Training Pipeline
Trains models, selects the best one based on RMSE, and registers it.
```bash
python scripts/training_pipeline.py
```

#### C. Frontend Dashboard (Main App)
Launches the Streamlit app which handles accurate real-time inference.
```bash
streamlit run src/frontend/app.py
```
*App will open at `http://localhost:8501`*
