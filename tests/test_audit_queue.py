import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import init_db, create_user, get_manual_review_queue, submit_reviewer_decision, get_audit_trail_logs, get_connection

def test_module_8_manual_review_and_audit():
    init_db()

    # Seed test claim
    conn = get_connection()
    cursor = conn.cursor()
    
    # Ensure test user and product exist
    cursor.execute("SELECT user_id FROM users WHERE username = 'admin'")
    u_row = cursor.fetchone()
    user_id = u_row['user_id'] if u_row else 1

    cursor.execute("SELECT product_id FROM products WHERE serial_number = 'SN-TEST-AUDIT'")
    p_row = cursor.fetchone()
    if p_row:
        product_id = p_row['product_id']
    else:
        cursor.execute("""
            INSERT INTO products (user_id, product_name, category, brand, model_number, serial_number, purchase_date, purchase_price, retailer, warranty_duration_months)
            VALUES (?, 'Audit Laptop', 'Laptop', 'Dell', 'DEL-01', 'SN-TEST-AUDIT', '2023-01-01', 1000.0, 'Store', 12)
        """, (user_id,))
        product_id = cursor.lastrowid

    test_claim_id = 'CLM-AUDIT-TEST-01'
    cursor.execute("DELETE FROM claims WHERE claim_id = ?", (test_claim_id,))
    cursor.execute("""
        INSERT INTO claims (claim_id, user_id, product_id, fault_type, claim_status, ml_prediction, ml_confidence, original_ai_verdict, final_verdict)
        VALUES (?, ?, ?, 'Power Failure', 'Manual Review', 'Valid Claim', 0.65, 'Manual Review Required', 'Manual Review Required')
    """, (test_claim_id, user_id, product_id))
    conn.commit()
    conn.close()

    # 1. Fetch Manual Review Queue
    queue = get_manual_review_queue()
    found = any(c['claim_id'] == test_claim_id for c in queue)
    assert found, f"Claim {test_claim_id} not found in Manual Review Queue"

    # 2. Submit Reviewer Override Decision
    success, msg = submit_reviewer_decision(
        claim_id=test_claim_id,
        reviewer_user_id=user_id,
        reviewer_name='System Admin',
        new_verdict='Approved',
        override_reason='Manually verified original purchase receipt via retailer portal call.'
    )
    assert success is True, f"Submission failed: {msg}"

    # 3. Verify Audit Trail Entry
    logs = get_audit_trail_logs(limit=10)
    audit_found = any(test_claim_id in str(l.get('details')) and 'Approved' in str(l.get('details')) for l in logs)
    assert audit_found, f"Audit log entry preserving original AI result & override reason missing. Logs: {logs}"


    print("Module 8 (Manual Review Queue & Audit Trail) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_8_manual_review_and_audit()
