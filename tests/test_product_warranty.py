import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import init_db, create_user, authenticate_user, get_connection

from src.product_warranty_manager import (
    register_product_and_warranty,
    get_registered_user_products,
    calculate_warranty_status,
    set_expiry_alert_threshold,
    get_expiry_alert_threshold
)

def test_module_2_product_and_warranty():
    init_db()

    # Clean test product record if exists
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM products WHERE serial_number = 'SN-APPLE-MAC-99'")
    conn.commit()
    conn.close()

    # 1. Test Admin-Settable Expiry Alert Threshold

    assert set_expiry_alert_threshold(45) is True
    assert get_expiry_alert_threshold() == 45
    set_expiry_alert_threshold(30) # reset to standard 30

    # 2. Test Warranty Status Calculation
    status_active = calculate_warranty_status('2026-01-01', 24)
    assert status_active['status'] in ['Active', 'Nearing Expiry']

    status_expired = calculate_warranty_status('2020-01-01', 12)
    assert status_expired['status'] == 'Expired'
    assert status_expired['remaining_days'] <= 0

    # 3. Test Product & Warranty Registration
    create_user('prod_owner1', 'owner1@test.com', 'pass123', 'Device Owner', 'Customer')
    user = authenticate_user('prod_owner1', 'pass123')
    user_id = user['user_id']

    p_data = {
        'product_name': 'MacBook Pro 16',
        'category': 'Laptop',
        'brand': 'Apple',
        'model_number': 'MBP-16-2023',
        'serial_number': 'SN-APPLE-MAC-99',
        'purchase_date': '2025-10-01',
        'purchase_price': 2499.0,
        'retailer': 'Apple Store',
        'warranty_duration_months': 12,
        'provider': 'AppleCare+'
    }

    success, p_id, w_id, msg = register_product_and_warranty(user_id, p_data)
    assert success is True, f"Registration failed: {msg}"
    assert p_id is not None
    assert w_id is not None

    # 4. Test User Product Retrieval with Live Warranty Status
    user_prods = get_registered_user_products(user_id, 'Customer')
    assert len(user_prods) >= 1
    p_retrieved = [p for p in user_prods if p['serial_number'] == 'SN-APPLE-MAC-99'][0]

    assert p_retrieved['brand'] == 'Apple'
    assert p_retrieved['warranty_status'] in ['Active', 'Nearing Expiry', 'Expired', 'Extended']
    assert 'calculated_expiry_date' in p_retrieved

    print("Module 2 (Product & Warranty Management) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_2_product_and_warranty()
