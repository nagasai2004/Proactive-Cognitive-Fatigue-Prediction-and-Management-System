import time
import threading
from pynput import mouse, keyboard
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ActivityMonitor:
    """
    Background worker that monitors keyboard and mouse activity using pynput.
    Aggregates statistics every 60 seconds.
    """
    def __init__(self, interval=60):
        self.interval = interval
        self.keystrokes = 0
        self.mouse_moves = 0
        self.mouse_clicks = 0
        self.last_activity_time = time.time()
        self.last_move_time = 0
        self.idle_time = 0
        self.start_time = time.time()
        
        self.running = False
        self.current_stats = {
            'keystrokes': 0,
            'mouse_moves': 0,
            'mouse_clicks': 0,
            'idle_time': 0,
            'screen_time': 0
        }
        
        # Cumulative stats for session verification
        self.session_keystrokes = 0
        self.session_mouse_moves = 0
        self.session_mouse_clicks = 0
        
        # Lock for thread-safe data access
        self.lock = threading.Lock()

    def _on_press(self, key):
        with self.lock:
            self.keystrokes += 1
            self.session_keystrokes += 1
            self.last_activity_time = time.time()
            # logging.debug("Key pressed")

    def _on_move(self, x, y):
        now = time.time()
        with self.lock:
            # If the interval between move events is > 0.5s, it's a NEW discrete move
            # This counts one continuous 'stroke' as 1 move count
            if now - self.last_move_time > 0.5:
                self.mouse_moves += 1
                self.session_mouse_moves += 1
            
            self.last_move_time = now
            self.last_activity_time = now

    def _on_click(self, x, y, button, pressed):
        if pressed:
            with self.lock:
                self.mouse_clicks += 1
                self.session_mouse_clicks += 1
                self.last_activity_time = time.time()

    def _monitor_idle(self):
        """Separate thread to track inactivity."""
        while self.running:
            time.sleep(1)
            now = time.time()
            # If no activity for more than 1 second, increment idle time
            if now - self.last_activity_time > 1: 
                with self.lock:
                    self.idle_time += 1

    def _aggregation_loop(self):
        """Aggregates data every interval (60s)."""
        while self.running:
            time.sleep(self.interval)
            with self.lock:
                # Store snapshot for ML prediction
                self.current_stats = {
                    'keystrokes': self.keystrokes,
                    'mouse_moves': self.mouse_moves,
                    'mouse_clicks': self.mouse_clicks,
                    'idle_time': self.idle_time,
                    'screen_time': time.time() - self.start_time
                }
                
                logging.info(f"60s Interval Aggregated: {self.current_stats}")
                
                # If a callback is provided (e.g. for DB logging), trigger it
                if self.callback:
                    try:
                        self.callback(self.current_stats.copy())
                    except Exception as e:
                        logging.error(f"Error in background callback: {e}")
                
                # Reset counts for next interval
                self.keystrokes = 0
                self.mouse_moves = 0
                self.mouse_clicks = 0
                self.idle_time = 0

    def start(self, callback=None):
        self.running = True
        self.callback = callback
        
        # Start input listeners
        self.mouse_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self._on_press)
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        # Start background threads
        threading.Thread(target=self._monitor_idle, daemon=True).start()
        threading.Thread(target=self._aggregation_loop, daemon=True).start()
        
        logging.info("Activity Monitoring Started (Interval: 60s).")

    def stop(self):
        self.running = False
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        logging.info("Activity Monitoring Stopped.")

    def get_latest_stats(self):
        """Returns the stats from the LAST completed 60s interval."""
        with self.lock:
            return self.current_stats.copy()

    def get_live_incremental_stats(self):
        """Returns counts accumulated in the CURRENT incomplete interval."""
        with self.lock:
            return {
                'keystrokes': self.keystrokes,
                'mouse_moves': self.mouse_moves,
                'mouse_clicks': self.mouse_clicks,
                'idle_time': self.idle_time,
                'screen_time': time.time() - self.start_time,
                'session_keystrokes': self.session_keystrokes,
                'session_mouse_moves': self.session_mouse_moves,
                'session_mouse_clicks': self.session_mouse_clicks
            }

if __name__ == "__main__":
    # Test run
    monitor = ActivityMonitor(interval=5)
    monitor.start()
    try:
        while True:
            time.sleep(2)
            print(f"Live Stats: {monitor.get_latest_stats()}")
    except KeyboardInterrupt:
        monitor.stop()
