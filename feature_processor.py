class FeatureProcessor:
    """
    Normalizes raw activity data into the 0-1 range expected by the ML model.
    """
    def __init__(self):
        # Define reasonable max limits for normalization (per 60 seconds)
        # These would ideally be based on user baseline or training data distribution
        self.MAX_VALS = {
            'screen_time': 3600,   # Max 1 hour (as total cumulative)
            'keystrokes': 300,    # Max 300 keys/min
            'mouse_moves': 100,  # Max 100 discrete strokes/min
            'mouse_clicks': 60,    # Max 60 clicks/min
            'idle_time': 60,      # Max 60 seconds/min
            'workload_level': 10   # Scale 1-10
        }

    def normalize(self, raw_stats, workload_level=5, interval=60):
        """
        Converts raw counts to normalized 0-1 values.
        Default interval is 60s.
        """
        scale_factor = 60 / interval
        
        def scale(val, max_val):
            scaled_val = val * scale_factor
            return min(max(scaled_val / max_val, 0.0), 1.0)

        normalized = [
            # Screen time is cumulative, no need for interval scaling
            min(max(raw_stats.get('screen_time', 0) / self.MAX_VALS['screen_time'], 0.0), 1.0),
            scale(raw_stats.get('keystrokes', 0), self.MAX_VALS['keystrokes']),
            scale(raw_stats.get('mouse_moves', 0), self.MAX_VALS['mouse_moves']),
            scale(raw_stats.get('mouse_clicks', 0), self.MAX_VALS['mouse_clicks']),
            scale(raw_stats.get('idle_time', 0), self.MAX_VALS['idle_time']),
            # Workload is a static level, no scaling needed
            raw_stats.get('workload_level', workload_level) / self.MAX_VALS['workload_level']
        ]
        
        return normalized
