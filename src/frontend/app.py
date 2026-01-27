import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

# --- Configuration & Constants ---
PAGE_TITLE = "Karachi AQI Predictor"
PAGE_ICON = "🍃"
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS (The "Premium" Dark Look) ---
def setup_css():
    st.markdown("""
    <style>
        /* Import a nice Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #e2e8f0; /* Light Slate */
            background-color: #0f172a; /* Slate 900 */
        }

        /* Gradient Background for the main app */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }

        /* Custom Card Styling */
        .metric-card {
            background-color: #1e293b; /* Slate 800 */
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
            border: 1px solid #334155; /* Slate 700 */
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
            border-color: #475569;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-weight: 700;
        }
        
        /* Custom Button Styling */
        div.stButton > button {
            background: linear-gradient(to right, #0ea5e9, #0284c7);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.2s;
            width: 100%;
        }
        div.stButton > button:hover {
            background: linear-gradient(to right, #0284c7, #0369a1);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
            transform: translateY(-1px);
        }
        div.stButton > button:active {
            color: white; /* Keep text white on click */
        }

        /* Divider */
        hr {
            margin: 2rem 0;
            border-color: #334155;
        }

        /* Tooltip help icons */
        .tooltip-icon {
            color: #94a3b8;
            font-size: 0.9em;
            cursor: help;
        }
        
        /* AQI Label badges */
        .aqi-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }

        /* Streamlit Text Elements */
        .stMarkdown p {
            color: #cbd5e1; /* Slate 300 */
        }
    </style>
    """, unsafe_allow_html=True)

# --- Helpers ---
def get_aqi_category_details(aqi):
    if aqi <= 50: return "Good", "#22c55e", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100: return "Moderate", "#eab308", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#f97316", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi <= 200: return "Unhealthy", "#ef4444", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300: return "Very Unhealthy", "#a855f7", "Health alert: The risk of health effects is increased for everyone."
    else: return "Hazardous", "#7f1d1d", "Health warning of emergency conditions: everyone is more likely to be affected."

# --- Views ---

def render_landing():
    """Renders the initial welcoming state."""
    
    # Hero Section
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown(f"# 🍃 Karachi AQI Predictor")
        st.markdown("""
        ### Breathe smarter with AI-powered forecasts.
        
        Karachi's air quality changes rapidly. This tool uses advanced machine learning to predict 
        Air Quality Index (AQI) levels for the next **72 hours** using real-time weather patterns 
        and historical data.
        
        **Why check AQI?**
        - Plan your outdoor activities.
        - Protect sensitive family members.
        - mitigate health risks.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Analyze Current Air Quality", type="primary"):
            st.session_state['page'] = 'loading'
            st.rerun()

    with col2:
        # A nice placeholder graphic or informative card
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌫️ ➡️ 🌤️</div>
            <h4 style="margin:0;">Real-time Analysis</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">Connecting to AQI Sensors & Weather APIs</p>
        </div>
        <br>
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖 + 📊</div>
            <h4 style="margin:0;">ML Powered</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">XGBoost/LightGBM Model Architecture</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # Educational Footer
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Data Collection**\nWe aggregate historical pollution data (PM2.5, NO2) and real-time weather metrics.")
    with c2:
        st.info("**2. ML Inference**\nOur model processes these features to forecast AQI trends for the upcoming days.")
    with c3:
        st.info("**3. Actionable Insights**\nWe translate complex numbers into simple categories and health advice.")


def render_dashboard():
    """Renders the main dashboard with predictions."""
    
    # Top Bar with Back Button
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            del st.session_state['data']
            st.session_state['page'] = 'landing'
            st.rerun()
    with c2:
        st.markdown("## 📊 Forecast Results")

    data = st.session_state.get('data')
    if not data:
        st.error("No data found. Please try again.")
        return

    hourly_data = data.get("hourly_predictions", [])
    daily_data = data.get("daily_summary", [])

    if not hourly_data:
        st.warning("Prediction service returned empty data.")
        return

    # --- Current Status Highlight ---
    current = hourly_data[0]
    curr_aqi = current['predicted_aqi']
    cat, color, desc = get_aqi_category_details(curr_aqi)
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 8px solid {color}">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0;">Right Now ({current['datetime']})</p>
                <h1 style="font-size: 3.5rem; margin: 0; color: {color};">{curr_aqi:.0f}</h1>
                <span class="aqi-badge" style="background-color: {color}">{cat}</span>
                <p style="margin-top: 8px; font-size: 0.9rem; color: #cbd5e1;">PM2.5: <strong>{current.get('pm2_5', 'N/A')} µg/m³</strong></p>
            </div>
            <div style="max-width: 600px;">
                <h3 style="margin-bottom: 0.5rem; color: #f8fafc;">Health Implication</h3>
                <p style="font-size: 1.05rem; color: #cbd5e1; line-height: 1.5;">{desc}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 3 Day Forecast Cards ---
    st.subheader("📅 3-Day Outlook")
    cols = st.columns(len(daily_data))
    
    for i, day in enumerate(daily_data):
        d_avg = day['avg_aqi']
        d_cat, d_color, _ = get_aqi_category_details(d_avg)
        d_date = datetime.strptime(day['date'], "%Y-%m-%d").strftime("%A, %b %d")
        
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; height: 100%;">
                <h4 style="margin-bottom: 0.5rem; color: #f8fafc;">{d_date}</h4>
                <div style="font-size: 2rem; font-weight: bold; color: {d_color};">{d_avg:.0f}</div>
                <div style="margin: 0.5rem 0;">
                    <span class="aqi-badge" style="background-color: {d_color}; font-size: 0.75rem;">{d_cat}</span>
                </div>
                <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">
                    Range: {day['min_aqi']:.0f} - {day['max_aqi']:.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Interactive Chart ---
    st.subheader("📈 72-Hour Trend Analysis")
    
    df_hourly = pd.DataFrame(hourly_data)
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
    
    # Improved Plotly Chart (Dark Mode)
    fig = px.line(df_hourly, x='datetime', y='predicted_aqi', 
                  markers=True, line_shape='spline')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Time",
        yaxis_title="Predicted AQI",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        font=dict(color="#e2e8f0")
    )
    fig.update_traces(line_color='#38bdf8', line_width=3, marker_size=6) # Light Blue line
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
    fig.update_xaxes(showgrid=False, gridcolor='#334155')
    
    # Add colored background zones
    shapes = [
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=50, fillcolor="green", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=50, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=100, y1=150, fillcolor="orange", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=150, y1=200, fillcolor="red", opacity=0.1, layer="below", line_width=0),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=200, y1=500, fillcolor="purple", opacity=0.1, layer="below", line_width=0),
    ]
    fig.update_layout(shapes=shapes)

    st.plotly_chart(fig, use_container_width=True)


# --- Main Controller ---
def main():
    setup_css()
    
    # Initialize Session State
    if 'page' not in st.session_state:
        st.session_state['page'] = 'landing'
    
    # Logic
    if st.session_state['page'] == 'landing':
        render_landing()
        
    elif st.session_state['page'] == 'loading':
        # Simulated loading screen with messages
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                with st.spinner("Connecting to Model Registry..."):
                    time.sleep(0.5)
                with st.spinner("Fetching Real-time Weather Data..."):
                    time.sleep(0.5)
                with st.spinner("Running Inference Models..."):
                    # Actual API Call
                    try:
                        response = requests.get(API_URL)
                        if response.status_code == 200:
                            st.session_state['data'] = response.json()
                            st.session_state['page'] = 'dashboard'
                            # Rerun to switch views
                        else:
                            st.error(f"API Error: {response.text}")
                            if st.button("Return"):
                                st.session_state['page'] = 'landing'
                                st.rerun()
                            return
                    except Exception as e:
                        st.error(f"Connection Failed: {e}")
                        if st.button("Return"):
                            st.session_state['page'] = 'landing'
                            st.rerun()
                        return
            
            st.rerun()

    elif st.session_state['page'] == 'dashboard':
        render_dashboard()

if __name__ == "__main__":
    main()
