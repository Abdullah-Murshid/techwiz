import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import init_db
from src.notification_monitoring import (
    init_notification_and_monitoring_tables,
    create_notification,
    get_user_notifications,
    log_anomaly_alert,
    get_admin_anomaly_alerts,
    check_and_log_model_anomalies,
    safe_execute
)

def test_module_4_notification_and_monitoring():
    init_db()
    init_notification_and_monitoring_tables()

    user_id = 999
    claim_id = 'CLM-NOTIF-TEST-01'

    # 1. Test Notification Triggers
    assert create_notification(user_id, claim_id, 'claim_submitted', 'Your warranty claim CLM-NOTIF-TEST-01 has been submitted.') is True
    assert create_notification(user_id, claim_id, 'status_changed', 'Your claim status updated to Approved.') is True

    notifs = get_user_notifications(user_id)
    assert len(notifs) >= 2
    assert any(n['event_type'] == 'claim_submitted' for n in notifs)

    # 2. Test Anomaly Monitoring Alerts
    assert log_anomaly_alert('failed_login', 'LOW', 'Repeated failed login attempt for user test_user') is True
    assert log_anomaly_alert('model_disagreement_spike', 'HIGH', 'Spike in model disagreement for Claim CLM-NOTIF-TEST-01') is True

    alerts = get_admin_anomaly_alerts(limit=10)
    assert len(alerts) >= 2
    assert any(a['alert_type'] == 'model_disagreement_spike' for a in alerts)

    # 3. Test Model Disagreement & Low Confidence Anomaly Detector
    mock_pipeline_res = {
        'python_model': {'confidence': 0.45},
        'model_comparison': {'comparison_status': 'Model Disagreement'}
    }
    check_and_log_model_anomalies(mock_pipeline_res, claim_id)

    updated_alerts = get_admin_anomaly_alerts(limit=20)
    assert any('Low confidence' in a['details'] for a in updated_alerts)

    # 4. Test Centralized User-Safe Error Handling
    def buggy_func():
        raise FileNotFoundError("Missing receipt file path")

    success, res = safe_execute(buggy_func)
    assert success is False
    assert res == "The requested file or document could not be found."
    assert "FileNotFoundError" not in res # No stack trace leak

    print("Module 4 (Notifications & Anomaly Monitoring) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_4_notification_and_monitoring()
