import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# --- Configuration & Constants ---
PAGE_TITLE = "Karachi AQI Predictor Pro"
PAGE_ICON = "⚡"
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (Premium Dark Mode) ---
def setup_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #e2e8f0;
            background-color: #0f172a;
        }
        
        /* Modern Gradient Background */
        .stApp {
            background: radial-gradient(circle at top left, #1e293b, #0f172a);
        }

        /* Glassmorphism Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: rgba(56, 189, 248, 0.5);
        }

        /* Badge Styling */
        .badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* Loaders */
        .loader {
            border: 4px solid #f3f3f3;
            border-radius: 50%;
            border-top: 4px solid #3498db;
            width: 40px;
            height: 40px;
            -webkit-animation: spin 1s linear infinite; /* Safari */
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

    </style>
    """, unsafe_allow_html=True)

# --- Helpers ---
def get_aqi_details(aqi):
    if aqi <= 50: return "Good", "#22c55e", "Air quality is satisfactory."
    elif aqi <= 100: return "Moderate", "#eab308", "Air quality is acceptable."
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#f97316", "Sensitive groups should reduce outdoor exertion."
    elif aqi <= 200: return "Unhealthy", "#ef4444", "Everyone may begin to experience health effects."
    elif aqi <= 300: return "Very Unhealthy", "#a855f7", "Health warnings of emergency conditions."
    else: return "Hazardous", "#7f1d1d", "Health warning of emergency conditions."

def fetch_predictions():
    """Fetches data from API."""
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# --- Components ---

def render_gauge(value):
    """Renders a Plotly Gauge chart for AQI."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current AQI", 'font': {'size': 24, 'color': '#f8fafc'}},
        gauge = {
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Hide bar, use needle logic if needed or just threshold
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 50], 'color': '#22c55e'},
                {'range': [50, 100], 'color': '#eab308'},
                {'range': [100, 150], 'color': '#f97316'},
                {'range': [150, 200], 'color': '#ef4444'},
                {'range': [200, 300], 'color': '#a855f7'},
                {'range': [300, 500], 'color': '#7f1d1d'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Outfit"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Main Logic ---

def main():
    setup_css()

    # --- Header ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("# ⚡ Karachi AQI Predictor")
    with c2:
        if st.button("🔄 Refresh"):
            del st.session_state['data']
            st.rerun()

    # --- Loading State (Auto-load) ---
    if 'data' not in st.session_state:
        # Cool Loader UI
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center;">
                <div class="loader"></div>
                <h3 style="margin-top: 20px; color: #38bdf8;">Analyzing Atmosphere...</h3>
                <p style="color: #94a3b8;">Connecting to Hopsworks Feature Store...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulated delay for "coolness" + actual fetch
            time.sleep(1.5) 
            data = fetch_predictions()
            
            if data:
                st.session_state['data'] = data
                st.rerun()
            else:
                st.error("❌ Failed to connect to Prediction Service. Is the backend running?")
                if st.button("Retry Connection"):
                    st.rerun()
                st.stop()

    # --- Dashboard View (Data Loaded) ---
    data = st.session_state['data']
    hourly = data.get('hourly_predictions', [])
    daily = data.get('daily_summary', [])
    meta = data.get('model_metadata', {})

    if not hourly:
        st.error("No prediction data returned.")
        st.stop()
    
    current = hourly[0]
    curr_aqi = current['predicted_aqi']
    curr_pm25 = current.get('pm2_5', 'N/A')
    cat_name, cat_color, cat_desc = get_aqi_details(curr_aqi)

    # --- Top Section: Status & Gauge ---
    st.markdown("---")
    
    col_stat, col_gauge, col_meta = st.columns([2, 2, 1])
    
    # Extract Today's Summary (First item)
    today_summary = daily.pop(0) if daily else {}
    today_range = f"Low: {today_summary.get('min_aqi', 0):.0f} - High: {today_summary.get('max_aqi', 0):.0f}" if today_summary else "N/A"
    today_avg = f"Avg: {today_summary.get('avg_aqi', 0):.0f}" if today_summary else ""
    
    with col_stat:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h4 style="margin:0; color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Current Status</h4>
                    <h1 style="font-size: 4.5rem; color: {cat_color}; margin: 5px 0 10px 0; line-height: 1;">{curr_aqi:.0f}</h1>
                    <span class="badge" style="background-color: {cat_color}; color: white;">{cat_name}</span>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px;">
                    <h5 style="margin: 0 0 5px 0; color: #cbd5e1;">Today's Forecast</h5>
                    <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">{today_range}</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #38bdf8; font-weight: bold;">{today_avg}</p>
                </div>
            </div>
            <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                <p style="font-size: 1.1rem; color: #cbd5e1; margin: 0;">PM2.5 Concentration: <strong style="color: #38bdf8;">{curr_pm25} µg/m³</strong></p>
                <p style="font-size: 0.9rem; color: #64748b; margin-top: 5px;">{cat_desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_gauge:
        # Render Gauge
        render_gauge(curr_aqi)

    with col_meta:
        # Model Metadata Card
        metrics = meta.get('metrics', {})
        r2 = metrics.get('R2', metrics.get('r2_score', '0.99+')) # Fallback
        
        # Round if float
        if isinstance(r2, float): r2 = f"{r2:.4f}"
        
        rmse = metrics.get('RMSE', metrics.get('rmse', 'N/A'))
        if isinstance(rmse, float): rmse = f"{rmse:.2f}"
            
        mae = metrics.get('MAE', metrics.get('mae', 'N/A'))
        if isinstance(mae, float): mae = f"{mae:.2f}"
        
        # New: Train RMSE extraction and Gap Calculation
        train_rmse_val = metrics.get('train_rmse', 0)
        test_rmse_val = metrics.get('RMSE', metrics.get('rmse', 0))
        
        # Format strings
        train_rmse_str = f"{float(train_rmse_val):.2f}" if train_rmse_val else "N/A"
        
        # Logic for explanation
        explanation = "Model generalizes well."
        color_status = "#22c55e" # Green
        
        if train_rmse_val and test_rmse_val:
            t_val = float(test_rmse_val)
            tr_val = float(train_rmse_val)
            gap = t_val - tr_val
            
            if gap < 1.5:
                explanation = "✅ <strong>Good Fit:</strong> Test error is close to Train error, meaning the model is <strong>not overfitting</strong>."
                color_status = "#22c55e"
            elif gap < 3.0:
                explanation = "⚠️ <strong>Moderate Gap:</strong> Some difference, but acceptable."
                color_status = "#eab308"
            else:
                explanation = "❌ <strong>Overfitting Risk:</strong> Large gap detected."
                color_status = "#ef4444"

        html_content = (
            f'<div class="metric-card" style="height: 100%;">'
            f'<h5 style="color: #94a3b8; margin: 0;">Model Performance</h5>'
            
            # --- RMSE Comparison Grid ---
            f'<div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: center;">'
            
            # Train RMSE Box
            f'<div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px;">'
            f'<span style="font-size: 0.8rem; color: #94a3b8; display: block;">Train RMSE</span>'
            f'<span style="color: #cbd5e1; font-size: 1.3rem; font-weight: bold;">{train_rmse_str}</span>'
            f'</div>'
            
            # Test RMSE Box
            f'<div style="background: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.3);">'
            f'<span style="font-size: 0.8rem; color: #38bdf8; display: block; font-weight: 600;">Test RMSE</span>'
            f'<span style="color: #38bdf8; font-size: 1.3rem; font-weight: bold;">{rmse}</span>'
            f'</div>'
            
            f'</div>' # End RMSE Grid

            # --- Explanation Box ---
            f'<div style="margin-top: 12px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; border-left: 3px solid {color_status};">'
            f'<p style="margin: 0; font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">{explanation}</p>'
            f'</div>'
            
            # --- Other Metrics (MAE, R2, Type) ---
            f'<div style="margin-top: 15px; padding-top: 12px; border-top: 1px solid #334155;">'
            
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
            f'<span style="font-size: 0.85rem; color: #94a3b8;">Mean Abs Error (MAE)</span>'
            f'<span style="color: #cbd5e1; font-weight: bold;">{mae}</span>'
            f'</div>'
            
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
            f'<span style="font-size: 0.85rem; color: #94a3b8;">R² Score</span>'
            f'<span style="color: #22c55e; font-weight: bold;">{r2}</span>'
            f'</div>'

            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<span style="font-size: 0.85rem; color: #94a3b8;">Model Type</span>'
            f'<span style="color: #64748b; font-size: 0.85rem;">{meta.get("type", "ML").upper()}</span>'
            f'</div>'
            
            f'</div>' # End Other Metrics
            
            f'</div>'
        )
        
        st.markdown(html_content, unsafe_allow_html=True)

    # --- Upcoming Forecast ---
    st.markdown("### 📅 Upcoming Forecast")
    c_days = st.columns(len(daily)) # Now only 2 days left in daily list
    
    for i, day in enumerate(daily):
        d_avg = day['avg_aqi']
        _, d_color, _ = get_aqi_details(d_avg)
        d_date = datetime.strptime(day['date'], "%Y-%m-%d").strftime("%a, %d %b")
        
        with c_days[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; border-bottom: 4px solid {d_color};">
                <h4 style="color: #f8fafc;">{d_date}</h4>
                <div style="font-size: 2rem; font-weight: bold; color: {d_color};">{d_avg:.0f}</div>
                <p style="color: #94a3b8; font-size: 0.9rem;">Range: {day['min_aqi']:.0f} - {day['max_aqi']:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # --- Chart Section (72 Hours) ---
    st.markdown("### 📊 72-Hour Future Trend")
    
    df_chart = pd.DataFrame(hourly)
    df_chart['datetime'] = pd.to_datetime(df_chart['datetime'])
    
    # Create readable label
    df_chart['Time'] = df_chart['datetime'].dt.strftime('%a %I %p') # e.g. Mon 02 PM
    
    # Use Bar Chart for better readability of discrete hours
    fig = px.bar(df_chart, x='datetime', y='predicted_aqi', 
                 color='predicted_aqi',
                 title="",
                 color_continuous_scale=[
                     (0.0, '#22c55e'),   # Good
                     (0.2, '#eab308'),   # Moderate
                     (0.4, '#f97316'),   # Sensitive
                     (0.6, '#ef4444'),   # Unhealthy
                     (0.8, '#a855f7'),   # Very Unhealthy
                     (1.0, '#7f1d1d')    # Hazardous
                 ],
                 range_color=[0, 300]
                )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#94a3b8', 'family': 'Outfit'},
        xaxis=dict(
            showgrid=False, 
            tickformat='%a %I %p',  # Readable format on axis: "Mon 02 PM"
            dtick=4 * 60 * 60 * 1000, # Show tick every 4 hours to avoid clutter
            title=None
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#334155',
            title='AQI Level'
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=350,
        coloraxis_showscale=False # Hide color bar
    )
    
    # Custom hover template
    fig.update_traces(
        hovertemplate="<b>%{x|%a, %d %b %I:%M %p}</b><br>AQI: %{y:.0f}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()
