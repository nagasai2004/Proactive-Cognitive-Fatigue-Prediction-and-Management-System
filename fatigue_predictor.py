import joblib
import os
import numpy as np
import logging

class FatiguePredictor:
    """
    Loads the Random Forest model and predicts fatigue score.
    """
    def __init__(self, model_path='models/rf_fatigue_model.pkl'):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logging.info(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        else:
            logging.error(f"Model file not found at {self.model_path}")

    def predict(self, normalized_features):
        if self.model is None:
            return sum(normalized_features) / len(normalized_features)
        
        try:
            features_array = np.array(normalized_features).reshape(1, -1)
            prediction = self.model.predict(features_array)
            return float(prediction[0])
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return 0.5

    def get_insights(self, normalized_features):
        """Returns the top contributing features for the current prediction."""
        feature_names = ['Screen Time', 'Keystrokes', 'Mouse Moving', 'Mouse Clicks', 'Idle', 'Workload']
        
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            # Multiply current feature values by importance to get weighted contribution
            contributions = normalized_features * self.model.feature_importances_
            # Sort by contribution
            sorted_indices = np.argsort(contributions)[::-1]
            return [(feature_names[i], round(float(contributions[i]), 3)) for i in sorted_indices]
        
        return []

    def predict_future(self, history_scores, minutes_ahead=5, lookback=5):
        """
        Forecasting fatigue for the next X minutes.
        Uses a simple trend projection based on the last 'lookback' scores.
        """
        if len(history_scores) < 2:
            return None
        
        # Take the last N readings
        recent = history_scores[-lookback:]
        
        x = np.arange(len(recent))
        y = np.array(recent)
        
        m, c = np.polyfit(x, y, 1)
        
        # Forecast steps ahead from the LAST point
        future_index = (len(recent) - 1) + minutes_ahead
        forecast = m * future_index + c
        
        return min(max(float(forecast), 0.0), 1.0)
