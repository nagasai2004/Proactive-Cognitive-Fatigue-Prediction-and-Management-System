class DecisionEngine:
    """
    Categorizes fatigue scores and provides recommendations.
    """
    def get_status(self, score):
        """
        Returns level and recommendation.
        
        • 0.0 – 0.4 → Low fatigue
        • 0.4 – 0.7 → Medium fatigue
        • 0.7 – 1.0 → High fatigue
        """
        if score < 0.4:
            level = "Low"
            recommendation = "Continue working. Focus level is optimal."
            color = "green"
        elif score < 0.7:
            level = "Medium"
            recommendation = "Take a short break (5 minutes). Consider some stretching."
            color = "orange"
        else:
            level = "High"
            recommendation = "Take a long break (15 minutes). Step away from the screen."
            color = "red"
            
        return {
            "level": level,
            "recommendation": recommendation,
            "color": color,
            "score": round(score, 2)
        }
