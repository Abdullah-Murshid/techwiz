import secrets
import hashlib
import time
import sqlite3
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import get_connection, hash_password, create_user, authenticate_user

# In-memory session store for tokens (token -> user dict)
SESSION_TOKENS = {}

VALID_ROLES = ['Customer', 'Service Center', 'Reviewer', 'Admin']

ROLE_PERMISSIONS = {
    'Customer': ['create_claim', 'view_own_claims', 'view_own_products', 'upload_documents'],
    'Service Center': ['view_assigned_claims', 'update_repair_status', 'view_products', 'upload_documents'],
    'Reviewer': ['view_manual_queue', 'submit_verdict_override', 'view_claims', 'view_audit_logs'],
    'Admin': ['create_claim', 'view_own_claims', 'view_own_products', 'upload_documents',
              'view_assigned_claims', 'update_repair_status', 'view_products',
              'view_manual_queue', 'submit_verdict_override', 'view_claims', 'view_audit_logs',
              'view_analytics', 'view_anomalies', 'manage_settings', 'export_data']
}

def generate_session_token(user_dict: dict) -> str:
    """Generates a secure random session token mapped to the user session."""
    token = secrets.token_hex(32)
    session_data = dict(user_dict)
    session_data['created_at_ts'] = time.time()
    SESSION_TOKENS[token] = session_data
    return token

def verify_session_token(token: str):
    """Verifies session token validity and returns user dictionary if valid."""
    if not token or token not in SESSION_TOKENS:
        return None
    session_data = SESSION_TOKENS[token]
    # Expire tokens older than 24 hours (86400 seconds)
    if time.time() - session_data.get('created_at_ts', 0) > 86400:
        del SESSION_TOKENS[token]
        return None
    return session_data

def invalidate_session_token(token: str):
    """Logs out and removes session token."""
    if token in SESSION_TOKENS:
        del SESSION_TOKENS[token]

def has_permission(role: str, action: str) -> bool:
    """Checks if a given user role has permission to perform an action."""
    allowed_actions = ROLE_PERMISSIONS.get(role, [])
    return action in allowed_actions

def get_role_accessible_claims(user_id: int, role: str):
    """
    Returns filtered claims query based on RBAC rules:
    - Customer: only their own claims (user_id)
    - Service Center: claims with service status or assigned
    - Reviewer: manual review queue & submitted claims
    - Admin: all claims
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if role == 'Customer':
        query = """
            SELECT c.*, p.product_name, p.category, p.brand, p.serial_number
            FROM claims c
            JOIN products p ON c.product_id = p.product_id
            WHERE c.user_id = ?
            ORDER BY c.created_at DESC
        """
        cursor.execute(query, (user_id,))
    elif role == 'Service Center':
        query = """
            SELECT c.*, p.product_name, p.category, p.brand, p.serial_number, u.full_name as customer_name
            FROM claims c
            JOIN products p ON c.product_id = p.product_id
            JOIN users u ON c.user_id = u.user_id
            WHERE c.claim_status IN ('Submitted', 'Under Evaluation', 'Additional Info Required', 'Approved')
            ORDER BY c.created_at DESC
        """
        cursor.execute(query)
    elif role == 'Reviewer':
        query = """
            SELECT c.*, p.product_name, p.category, p.brand, p.serial_number, u.full_name as customer_name
            FROM claims c
            JOIN products p ON c.product_id = p.product_id
            JOIN users u ON c.user_id = u.user_id
            WHERE c.final_verdict = 'Manual Review Required' OR c.claim_status IN ('Submitted', 'Manual Review', 'Under Evaluation')
            ORDER BY c.created_at DESC
        """
        cursor.execute(query)
    else:  # Admin
        query = """
            SELECT c.*, p.product_name, p.category, p.brand, p.serial_number, u.full_name as customer_name
            FROM claims c
            JOIN products p ON c.product_id = p.product_id
            JOIN users u ON c.user_id = u.user_id
            ORDER BY c.created_at DESC
        """
        cursor.execute(query)

    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == '__main__':
    print("Auth & RBAC module initialized.")
