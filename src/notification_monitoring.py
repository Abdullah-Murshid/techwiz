import os
import json
import sqlite3
import datetime
import traceback
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import get_connection

def init_notification_and_monitoring_tables():
    """Initializes notifications and anomaly_alerts tables in database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Notifications Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            claim_id TEXT,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    # 2. Anomaly Alerts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')) NOT NULL,
            details TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Ensure claims table has model version columns
    cursor.execute("PRAGMA table_info(claims)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if 'python_model_version' not in existing_cols:
        cursor.execute("ALTER TABLE claims ADD COLUMN python_model_version TEXT DEFAULT '1.0.0'")
    if 'teachable_machine_version' not in existing_cols:
        cursor.execute("ALTER TABLE claims ADD COLUMN teachable_machine_version TEXT DEFAULT 'TM-2.4.16'")

    conn.commit()
    conn.close()

# --- NOTIFICATIONS SYSTEM ---
def create_notification(user_id: int, claim_id: str, event_type: str, message: str) -> bool:
    """Creates a user notification for claim lifecycle events."""
    init_notification_and_monitoring_tables()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO notifications (user_id, claim_id, event_type, message)
            VALUES (?, ?, ?, ?)
        """, (int(user_id), claim_id, event_type, message))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        return False

def get_user_notifications(user_id: int) -> list:
    """Retrieves notifications for a specific user."""
    init_notification_and_monitoring_tables()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 50", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- ANOMALY MONITORING SYSTEM ---
def log_anomaly_alert(alert_type: str, severity: str, details: str) -> bool:
    """Logs system anomaly alert for admin visibility."""
    init_notification_and_monitoring_tables()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO anomaly_alerts (alert_type, severity, details)
            VALUES (?, ?, ?)
        """, (alert_type, severity, details))
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False

def get_admin_anomaly_alerts(limit=50) -> list:
    """Retrieves system anomaly alerts for admin dashboard."""
    init_notification_and_monitoring_tables()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM anomaly_alerts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def check_and_log_model_anomalies(pipeline_result: dict, claim_id: str):
    """Scans pipeline result for model disagreement or low confidence spikes."""
    py_conf = pipeline_result.get('python_model', {}).get('confidence', 1.0)
    comp_status = pipeline_result.get('model_comparison', {}).get('comparison_status', 'Strong Match')
    
    if py_conf < 0.60:
        log_anomaly_alert(
            alert_type='low_confidence_spike',
            severity='MEDIUM',
            details=f"Low confidence prediction spike detected for Claim '{claim_id}' (Confidence: {py_conf*100:.1f}%)."
        )
        
    if comp_status == 'Model Disagreement':
        log_anomaly_alert(
            alert_type='model_disagreement_spike',
            severity='HIGH',
            details=f"Model disagreement spike detected for Claim '{claim_id}'. Python vs Teachable Machine mismatch."
        )

# --- CENTRALIZED USER-SAFE ERROR HANDLER ---
def safe_execute(func, *args, **kwargs):
    """
    Executes a function wrapped in centralized error handling.
    Catches exceptions, logs anomaly alert, and returns user-safe error message.
    """
    try:
        return True, func(*args, **kwargs)
    except FileNotFoundError as e:
        log_anomaly_alert('failed_upload', 'MEDIUM', f"File Not Found Error: {str(e)}")
        return False, "The requested file or document could not be found."
    except sqlite3.Error as e:
        log_anomaly_alert('db_error', 'HIGH', f"Database Error: {str(e)}")
        return False, "A database operation encountered an error. Please try again."
    except Exception as e:
        err_msg = str(e)
        log_anomaly_alert('db_error', 'HIGH', f"Unhandled Error: {err_msg}\n{traceback.format_exc()}")
        return False, "An unexpected error occurred while processing your request. Please try again."

if __name__ == '__main__':
    init_notification_and_monitoring_tables()
    print("Notification & Monitoring system initialized.")
