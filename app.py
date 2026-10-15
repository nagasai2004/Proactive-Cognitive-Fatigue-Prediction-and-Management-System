import streamlit as st
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import logging
try:
    from plyer import notification
except Exception:
    class notification:  # no-op stand-in when running on a server
        @staticmethod
        def notify(*args, **kwargs):
            pass
import io

from activity_monitor import ActivityMonitor
from feature_processor import FeatureProcessor
from fatigue_predictor import FatiguePredictor
from decision_engine import DecisionEngine
from database_manager import DatabaseManager

# --- Page Configuration ---
st.set_page_config(
    page_title="Fatigue Guard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced CSS for Data-Reactive Premium UI ---
def apply_custom_styles(level="Low"):
    colors = {
        "Low": ("#00b09b", "#96c93d", "rgba(0, 176, 155, 0.4)"),
        "Medium": ("#f8b500", "#fceabb", "rgba(248, 181, 0, 0.4)"),
        "High": ("#ff5f6d", "#ffc371", "rgba(255, 95, 109, 0.4)")
    }
    primary, secondary, glow = colors.get(level, colors["Low"])

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Outfit', sans-serif;
            background-color: #f0f4f8;
            background-image: 
                radial-gradient(at 0% 0%, hsla(190, 100%, 90%, 1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(260, 100%, 95%, 1) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(320, 100%, 90%, 1) 0, transparent 50%), 
                radial-gradient(at 0% 100%, hsla(150, 100%, 90%, 1) 0, transparent 50%), 
                radial-gradient(at 50% 100%, hsla(200, 100%, 95%, 1) 0, transparent 50%), 
                radial-gradient(at 100% 100%, hsla(280, 100%, 90%, 1) 0, transparent 50%);
            background-attachment: fixed;
        }}

        [data-testid="stMetricLabel"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50 !important;
            font-weight: 600 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #3a7bd5 !important;
            font-weight: 800 !important;
            font-size: 2.0rem !important;
        }}

        /* Data-Reactive Containers */
        div[data-testid="stHorizontalBlock"]:nth-of-type(4) div[data-testid="column"]:nth-of-type(1) .stPlotlyChart {{
            background: linear-gradient(135deg, white 0%, {glow} 100%) !important;
            border: 3px solid {primary} !important;
            box-shadow: 0 10px 40px {glow} !important;
            border-radius: 24px !important;
            transition: all 0.6s ease;
            animation: breathe {level == 'High' and '1.5s' or '4s'} ease-in-out infinite;
        }}

        div[data-testid="stHorizontalBlock"]:nth-of-type(4) div[data-testid="column"]:nth-of-type(2) .stPlotlyChart {{
            background: white !important;
            border: 3px solid #3a7bd5 !important;
            border-radius: 24px !important;
        }}

        @keyframes breathe {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.005); }}
            100% {{ transform: scale(1); }}
        }}

        [data-testid="stMetric"] {{
            background: white !important;
            border-radius: 20px !important;
            padding: 12px 10px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        }}

        .insight-card {{
            background: white;
            padding: 15px;
            border-radius: 12px;
            margin: 5px 0;
            border-left: 5px solid {primary};
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .main-header {{
            background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 4.2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def trigger_alert(level):
    if level == "High":
        try:
            notification.notify(
                title="⚠️ Fatigue Alert!",
                message="Your fatigue level is critically high. Please take a 15-minute break now.",
                app_name="Fatigue Guard",
                timeout=10
            )
        except: pass

@st.cache_resource
def get_monitor():
    processor, predictor, engine, db = get_ml_system()
    
    # Store the last notification time in a localized attribute
    if 'last_notify_time' not in st.session_state:
        st.session_state.last_notify_time = time.time()

    def background_logging_worker(stats):
        # 1. Process and Log Data
        norm = processor.normalize(stats, interval=60)
        scr = predictor.predict(norm)
        stat = engine.get_status(scr)
        inf = predictor.get_insights(norm)
        driver = inf[0] if inf else ("None", 0.0)
        db.log_record(stats, scr, stat, driver)

        # 2. Periodic Notification Logic (Every 5 Minutes)
        # We check if 300 seconds (5 mins) have passed since start or last notify
        now = time.time()
        
        # Using a file-based or global-variable approach since this runs in a thread
        # that might not have access to Streamlit's session_state comfortably
        if not hasattr(background_logging_worker, "last_popup"):
            # Set to 0 to trigger the FIRST notification immediately
            background_logging_worker.last_popup = 0 

        if now - background_logging_worker.last_popup >= 120 or (stat['level'] == "High" and not getattr(background_logging_worker, "high_alert_tripped", False)):
            try:
                # Determine title based on severity
                msg_title = "⚠️ CRITICAL: High Fatigue" if stat['level'] == "High" else "🕒 Fatigue Status Update"
                
                notification.notify(
                    title=msg_title,
                    message=f"Level: {stat['level']}\nScore: {int(scr*100)}%\nRecommendation: {stat['recommendation']}",
                    app_name="Fatigue Guard",
                    timeout=10
                )
                background_logging_worker.last_popup = now
                
                # Flag to prevent spamming High alert until it changes
                background_logging_worker.high_alert_tripped = (stat['level'] == "High")
            except Exception as e:
                logging.error(f"Notification error: {e}")
        
        # Reset high alert flag if state drops below High
        if stat['level'] != "High":
            background_logging_worker.high_alert_tripped = False

    monitor = ActivityMonitor(interval=60)
    try:
        monitor.start(callback=background_logging_worker)
    except Exception as e:
        logging.error(f"ActivityMonitor.start() raised an exception: {e}")
        try:
            monitor.running = False
        except:
            pass
    return monitor

@st.cache_resource
def get_ml_system(version="2026_02_11_v4"):
    return FeatureProcessor(), FatiguePredictor(), DecisionEngine(), DatabaseManager()

# --- Visualization Utilities ---
def create_fatigue_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fatigue Level (%)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [40, 70], 'color': 'rgba(255, 165, 0, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
    return fig

# --- App Logic ---
def main():
    monitor = get_monitor()
    processor, predictor, engine, db = get_ml_system()
    
    # Initialize session state for tracking
    if 'last_log_id' not in st.session_state:
        st.session_state.last_log_id = None
    if 'last_alert' not in st.session_state:
        st.session_state.last_alert = ""

    # 1. Process Latest Data for UI
    agg_stats = monitor.get_latest_stats()
    normalized = processor.normalize(agg_stats, interval=60)
    score = predictor.predict(normalized)
    status = engine.get_status(score)
    insights = predictor.get_insights(normalized)

    # 2. Load Persistent History for Charts
    history_df = db.get_history(limit=50)

    apply_custom_styles(level=status['level'])

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2040/2040504.png", width=80)
        st.title("Fatigue Guard Pro")
        st.markdown(f"**Storage:** Local SQLite (`data/fatigue_history.db`)")
        st.markdown("---")
        
        st.subheader("📤 Export Data")
        if not history_df.empty:
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full History (CSV)", data=csv, file_name="fatigue_database_export.csv", mime="text/csv", use_container_width=True)
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            db.clear_history()
            st.rerun()

        if st.button("🔔 Test Notification", use_container_width=True):
            try:
                notification.notify(
                    title="🔔 Test Successful",
                    message="If you see this, your notifications are working correctly!",
                    app_name="Fatigue Guard",
                    timeout=5
                )
                st.toast("Test notification sent!")
            except Exception as e:
                st.error(f"Notification Error: {e}")

        st.markdown("---")
        st.subheader("💡 Wellness Tips")
        st.caption("• Follow the 20-20-20 rule for eyes.")
        st.caption("• Keep your room well-ventilated.")
        st.caption("• Take a short walk every hour.")

    # --- Header ---
    st.markdown('<div class="main-header">🛡️ Fatigue Guard Pro</div>', unsafe_allow_html=True)
    
    live_stats = monitor.get_live_incremental_stats()
    
    # --- Layout ---
    tcol1, tcol2 = st.columns([1.5, 1])
    with tcol1:
        st.plotly_chart(create_fatigue_gauge(score, status['color']), use_container_width=True)

    with tcol2:
        st.subheader("🧠 Status Analysis")
        st.markdown(f"<div class='insight-card'><b>State:</b> {status['level']} Fatigue<br><b>Recommendation:</b> {status['recommendation']}</div>", unsafe_allow_html=True)
        
        # 3. Future Fatigue Forecast (Next 5, 10, 15 Mins)
        if not history_df.empty and len(history_df) >= 2:
            st.markdown("<b>🔮 Future Trajectory:</b>", unsafe_allow_html=True)
            fcol1, fcol2, fcol3 = st.columns(3)
            
            history_list = history_df['fatigue_score'].tolist()
            
            for col, horizon in zip([fcol1, fcol2, fcol3], [5, 10, 15]):
                f_score = predictor.predict_future(history_list, minutes_ahead=horizon)
                f_level = engine.get_status(f_score)['level']
                col.metric(f"{horizon}m", f"{int(f_score*100)}%", delta=f_level, delta_color="inverse")
        
        # Performance Metrics
        st.markdown("---")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Keystrokes", f"{live_stats['keystrokes']}")
        mcol2.metric("Mouse Moves", f"{live_stats['mouse_moves']}")
        mcol3, mcol4 = st.columns(2)
        mcol3.metric("Clicks", f"{live_stats['mouse_clicks']}")
        mcol4.metric("Idle Time", f"{live_stats['idle_time']}s")

    st.markdown("---")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.subheader("📈 Fatigue History")
        if len(history_df) > 1:
            fig_line = px.line(history_df, x='timestamp', y='fatigue_score', color_discrete_sequence=['#3a7bd5'])
            fig_line.update_traces(line=dict(width=4, shape='spline'), mode='lines+markers')
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, xaxis_title="Time", yaxis_title="Score")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Gathering historical data... (Updates every 60s)")

    with bcol2:
        st.subheader("📊 Feature Influence")
        if insights:
            df_feat = pd.DataFrame({'Feature': [x[0] for x in insights], 'Influence': [x[1] for x in insights]})
            fig_bar = px.bar(df_feat, x='Influence', y='Feature', orientation='h', color='Influence', color_continuous_scale='Blues')
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Analyzing feature contributions...")

    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()
