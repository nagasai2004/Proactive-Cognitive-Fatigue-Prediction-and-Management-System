import sqlite3
import os
from datetime import datetime
import pandas as pd
import logging

class DatabaseManager:
    """
    Handles persistence for activity metrics and fatigue predictions using SQLite.
    """
    def __init__(self, db_path='data/fatigue_history.db'):
        self.db_path = db_path
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        """Creates the necessary tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fatigue_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    keystrokes INTEGER,
                    mouse_moves INTEGER,
                    mouse_clicks INTEGER,
                    idle_time INTEGER,
                    fatigue_score REAL,
                    fatigue_level TEXT,
                    recommendation TEXT,
                    primary_driver TEXT,
                    driver_impact REAL
                )
            ''')
            conn.commit()
            logging.info("Database initialized successfully.")

    def log_record(self, stats, score, status, driver_info):
        """
        Saves a single snapshot of activity and prediction to the database.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO fatigue_history (
                    keystrokes, mouse_moves, mouse_clicks, idle_time, 
                    fatigue_score, fatigue_level, recommendation, 
                    primary_driver, driver_impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats.get('keystrokes', 0),
                stats.get('mouse_moves', 0),
                stats.get('mouse_clicks', 0),
                stats.get('idle_time', 0),
                round(float(score), 4),
                status.get('level', 'Unknown'),
                status.get('recommendation', ''),
                driver_info[0] if driver_info else "None",
                driver_info[1] if driver_info else 0.0
            ))
            conn.commit()

    def get_history(self, limit=100):
        """Returns the latest history as a Pandas DataFrame."""
        with self._get_connection() as conn:
            query = f"SELECT * FROM fatigue_history ORDER BY timestamp DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            return df.sort_values('timestamp')

    def clear_history(self):
        """Wipes all data from the history table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM fatigue_history')
            conn.commit()
            logging.warning("All fatigue history has been cleared.")
