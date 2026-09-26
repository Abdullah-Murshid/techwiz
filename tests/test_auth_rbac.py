import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import init_db, create_user, authenticate_user, get_connection
from src.auth import generate_session_token, verify_session_token, invalidate_session_token, has_permission, get_role_accessible_claims

def test_module_1_auth_and_roles():
    init_db()

    # 1. Register test users for 4 roles
    create_user('cust_test1', 'cust1@test.com', 'pass123', 'Customer One', 'Customer')
    create_user('sc_test1', 'sc1@test.com', 'pass123', 'Service Center Tech', 'Service Center')
    create_user('rev_test1', 'rev1@test.com', 'pass123', 'Reviewer Auditor', 'Reviewer')
    create_user('admin_test1', 'admin1@test.com', 'pass123', 'System Administrator', 'Admin')

    # 2. Test authentication
    user = authenticate_user('cust_test1', 'pass123')
    assert user is not None, "Customer authentication failed"
    assert user['role'] == 'Customer'

    # 3. Test Session Token Generation & Verification
    token = generate_session_token(dict(user))
    assert token is not None, "Token generation failed"
    
    session_user = verify_session_token(token)
    assert session_user is not None, "Session token verification failed"
    assert session_user['username'] == 'cust_test1'

    invalidate_session_token(token)
    assert verify_session_token(token) is None, "Token invalidation failed"

    # 4. Test RBAC Permissions
    assert has_permission('Customer', 'create_claim') is True
    assert has_permission('Customer', 'view_anomalies') is False
    assert has_permission('Admin', 'view_anomalies') is True
    assert has_permission('Reviewer', 'submit_verdict_override') is True
    assert has_permission('Service Center', 'update_repair_status') is True

    # 5. Test Role-Based Claim Access Filtering
    cust_claims = get_role_accessible_claims(user['user_id'], 'Customer')
    admin_claims = get_role_accessible_claims(user['user_id'], 'Admin')
    assert isinstance(cust_claims, list)
    assert isinstance(admin_claims, list)

    print("Module 1 (Authentication & Roles) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_1_auth_and_roles()
