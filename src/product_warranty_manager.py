import os
import json
import sqlite3
import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import get_connection, DB_PATH
from policies.rules import RuleEngine

# System Settings file for admin-configurable thresholds
SETTINGS_FILE = os.path.join('policies', 'system_settings.json')

def get_expiry_alert_threshold() -> int:
    """Gets admin-settable days-before-expiry threshold (default: 30 days)."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('expiry_alert_threshold_days', 30)
        except Exception:
            pass
    return 30

def set_expiry_alert_threshold(days: int) -> bool:
    """Sets admin-configurable days-before-expiry threshold."""
    os.makedirs('policies', exist_ok=True)
    try:
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
        settings['expiry_alert_threshold_days'] = int(days)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception:
        return False

def calculate_warranty_status(purchase_date_str: str, duration_months: int, is_extended: bool = False) -> dict:
    """
    Auto-calculates warranty status: Active, Expired, Nearing Expiry, or Extended.
    Exposes remaining days, expiry date, and alert flags.
    """
    alert_days = get_expiry_alert_threshold()
    try:
        p_dt = datetime.datetime.strptime(str(purchase_date_str), '%Y-%m-%d').date()
    except Exception:
        p_dt = datetime.date.today()

    expiry_dt = p_dt + datetime.timedelta(days=int(duration_months * 30.4375))
    today = datetime.date.today()
    remaining_days = (expiry_dt - today).days

    if remaining_days <= 0:
        status = "Expired"
    elif remaining_days <= alert_days:
        status = "Nearing Expiry"
    elif is_extended:
        status = "Extended"
    else:
        status = "Active"

    return {
        'status': status,
        'purchase_date': str(p_dt),
        'expiry_date': str(expiry_dt),
        'remaining_days': remaining_days,
        'alert_flag': remaining_days <= alert_days and remaining_days > 0,
        'alert_threshold_days': alert_days
    }

def init_warranties_table():
    """Initializes warranties table in assurex.db if missing."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS warranties (
            warranty_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER UNIQUE NOT NULL,
            provider TEXT NOT NULL,
            start_date TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            coverage_conditions TEXT,
            exclusions TEXT,
            service_center_details TEXT,
            is_extended INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    ''')
    conn.commit()
    conn.close()

def register_product_and_warranty(user_id: int, product_data: dict) -> tuple:
    """
    Registers a new product and creates its associated warranty record 
    pulling category policies from warranty_policies.json.
    """
    init_warranties_table()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        category = product_data.get('category', 'Laptop')
        rule_engine = RuleEngine()
        cat_policy = rule_engine.policies.get(category, rule_engine.policies.get('Default', {}))

        duration_months = int(product_data.get('warranty_duration_months', cat_policy.get('standard_warranty_months', 12)))
        purchase_date_str = str(product_data.get('purchase_date', datetime.date.today().strftime('%Y-%m-%d')))
        
        calc_res = calculate_warranty_status(purchase_date_str, duration_months)

        # 1. Insert product into products table
        product_name = product_data.get('product_name') or f"{product_data.get('brand', '')} {product_data.get('model_number', '')}".strip()
        cursor.execute("""
            INSERT INTO products (
                user_id, product_name, category, brand, model_number, serial_number,
                purchase_date, purchase_price, retailer, warranty_duration_months
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(user_id),
            product_name,
            category,
            product_data.get('brand', 'Unknown'),
            product_data.get('model_number', 'N/A'),
            product_data.get('serial_number', f"SN-{int(datetime.datetime.now().timestamp())}"),
            purchase_date_str,
            float(product_data.get('purchase_price', 500.0)),
            product_data.get('retailer', 'Authorized Store'),
            duration_months
        ))
        product_id = cursor.lastrowid

        # 2. Insert warranty into warranties table
        provider = product_data.get('provider', f"{product_data.get('brand', 'Manufacturer')} AssureX Warranty")
        coverage_json = json.dumps(cat_policy.get('covered_faults', []))
        exclusions_json = json.dumps(cat_policy.get('excluded_faults', []))
        service_center = product_data.get('service_center_details', f"{product_data.get('brand', 'Authorized')} Service Network")

        cursor.execute("""
            INSERT INTO warranties (
                product_id, provider, start_date, expiry_date, coverage_conditions,
                exclusions, service_center_details, is_extended
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(product_id),
            provider,
            calc_res['purchase_date'],
            calc_res['expiry_date'],
            coverage_json,
            exclusions_json,
            service_center,
            1 if product_data.get('is_extended') else 0
        ))
        warranty_id = cursor.lastrowid

        conn.commit()
        conn.close()
        return True, product_id, warranty_id, "Product & Warranty registered successfully!"
    except Exception as e:
        conn.close()
        return False, None, None, f"Registration failed: {str(e)}"

def get_registered_user_products(user_id: int, role: str = 'Customer') -> list:
    """Retrieves list of registered products with live calculated warranty status."""
    init_warranties_table()
    conn = get_connection()
    cursor = conn.cursor()

    if role == 'Customer':
        query = """
            SELECT p.*, w.provider, w.start_date, w.expiry_date, w.coverage_conditions, w.exclusions, w.is_extended
            FROM products p
            LEFT JOIN warranties w ON p.product_id = w.product_id
            WHERE p.user_id = ?
            ORDER BY p.product_id DESC
        """
        cursor.execute(query, (user_id,))
    else: # Admin / Reviewer / Service Center sees all
        query = """
            SELECT p.*, w.provider, w.start_date, w.expiry_date, w.coverage_conditions, w.exclusions, w.is_extended, u.full_name as owner_name
            FROM products p
            LEFT JOIN warranties w ON p.product_id = w.product_id
            LEFT JOIN users u ON p.user_id = u.user_id
            ORDER BY p.product_id DESC
        """
        cursor.execute(query)

    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        status_info = calculate_warranty_status(d.get('purchase_date'), d.get('warranty_duration_months', 12), bool(d.get('is_extended')))
        d['warranty_status'] = status_info['status']
        d['remaining_days'] = status_info['remaining_days']
        d['calculated_expiry_date'] = status_info['expiry_date']
        d['alert_flag'] = status_info['alert_flag']
        results.append(d)

    return results

if __name__ == '__main__':
    init_warranties_table()
    print("Product & Warranty Manager initialized.")
