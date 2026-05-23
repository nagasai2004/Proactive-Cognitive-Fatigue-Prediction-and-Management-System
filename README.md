# Automatic Cognitive Fatigue Monitoring System

A real-time desktop application designed to monitor user computer activity and predict cognitive fatigue levels using machine learning.

## 🚀 Overview

The system operates fully automatically in the background, capturing keyboard and mouse events to estimate the user's workload and fatigue levels. It uses a Random Forest regression model to provide real-time scores and break recommendations via a modern Streamlit dashboard.

---

## 🏗️ System Architecture

1.  **Activity Monitoring (`activity_monitor.py`)**: Uses non-blocking listeners (`pynput`) to track raw interaction events.
2.  **Feature Normalization (`feature_processor.py`)**: Scales raw counts to a normalized [0, 1] range.
3.  **Prediction Logic (`fatigue_predictor.py`)**: Loads the pre-trained Random Forest model and predicts the fatigue score.
4.  **Decision Engine (`decision_engine.py`)**: Categorizes scores into Low, Medium, or High fatigue levels.
5.  **Dashboard UI (`app.py`)**: Visualizes real-time metrics and trends using Streamlit.

---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.8+
- Windows/Linux/MacOS

### 2. Setup
Clone or download this repository to the desired directory.

```bash
# Navigate to project directory
cd cognitive_fatigue_app

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application

To start the system, run the following command:

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

---

## 📊 Module Responsibilities

| Module | Responsibility |
| :--- | :--- |
| `activity_monitor.py` | Background event capture (Mouse, Keyboard, Inactivity). |
| `feature_processor.py` | Data scaling and feature engineering. |
| `fatigue_predictor.py` | Model inference using `joblib` and `scikit-learn`. |
| `decision_engine.py` | Rule-based classification for break recommendations. |
| `app.py` | Main orchestrator and interactive UI dashboard. |

---

## 📝 Example Console Logs

Upon startup, you will see logs similar to:
```text
2026-02-04 08:45:00 - INFO - Model loaded successfully from models/rf_fatigue_model.pkl
2026-02-04 08:45:00 - INFO - Activity Monitoring Started.
2026-02-04 08:46:00 - INFO - Aggregated Stats: {'keystrokes': 142, 'mouse_moves': 420, 'mouse_clicks': 12, 'idle_time': 5, 'screen_time': 60.0}
2026-02-04 08:47:00 - INFO - Aggregated Stats: {'keystrokes': 85, 'mouse_moves': 310, 'mouse_clicks': 8, 'idle_time': 15, 'screen_time': 120.0}
```

---

## 🛡️ Important Notes
- **Privacy**: No text is logged; only keystroke frequencies are tracked.
- **Model**: The `rf_fatigue_model.pkl` must be present in the `models/` directory.
- **Background Run**: Keep the terminal window open to maintain background monitoring.
