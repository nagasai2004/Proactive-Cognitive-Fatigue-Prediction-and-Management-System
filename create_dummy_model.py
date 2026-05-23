import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def generate_model():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate dummy data
    # Features: screen_time, keystrokes, mouse_moves, mouse_clicks, idle_time, workload_level
    n_samples = 1000
    X = np.random.rand(n_samples, 6)
    
    # Target: Fatigue score (simple weighted logic for dummy model)
    # Higher screen time, more keystrokes, and high workload increase fatigue
    # High idle time decreases fatigue
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 + X[:, 2] * 0.1 + X[:, 3] * 0.1 - X[:, 4] * 0.2 + X[:, 5] * 0.3)
    y = (y - y.min()) / (y.max() - y.min())  # Scale to 0-1
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Ensure directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/rf_fatigue_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    generate_model()
