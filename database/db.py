import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join('database', 'assurex.db')

def get_connection():
    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT CHECK(role IN ('Customer', 'Service Center', 'Reviewer', 'Admin')) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Products Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            brand TEXT NOT NULL,
            model_number TEXT NOT NULL,
            serial_number TEXT UNIQUE NOT NULL,
            purchase_date TEXT NOT NULL,
            purchase_price REAL NOT NULL,
            retailer TEXT NOT NULL,
            warranty_duration_months INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    # 3. Claims Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            fault_type TEXT NOT NULL,
            fault_description TEXT,
            claim_status TEXT DEFAULT 'Submitted',
            ml_prediction TEXT,
            ml_confidence REAL,
            teachable_prediction TEXT,
            teachable_confidence REAL,
            original_ai_verdict TEXT,
            final_verdict TEXT,
            override_reason TEXT,
            reviewer_comments TEXT,
            reviewed_by TEXT,
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
    ''')

    # 4. Audit Trail Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Seed Default Admin & Reviewer Accounts if missing
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@assurex.com', hash_password('admin123'), 'System Admin', 'Admin'))

    cursor.execute("SELECT * FROM users WHERE username = 'reviewer'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('reviewer', 'reviewer@assurex.com', hash_password('reviewer123'), 'Claims Auditor', 'Reviewer'))

    # Run migrations for legacy claims table if missing new columns
    cursor.execute("PRAGMA table_info(claims)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    new_cols = {
        'original_ai_verdict': 'TEXT',
        'override_reason': 'TEXT',
        'reviewed_by': 'TEXT',
        'reviewed_at': 'TIMESTAMP'
    }
    for col_name, col_type in new_cols.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE claims ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def authenticate_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, pwd_hash))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(username, email, password, full_name, role='Customer'):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, hash_password(password), full_name, role))
        conn.commit()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Username or Email already exists."
    finally:
        conn.close()

def get_manual_review_queue():
    """Fetches all claims awaiting manual review or under evaluation."""
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        SELECT c.claim_id, c.fault_type, c.claim_status, c.ml_prediction, c.ml_confidence,
               c.teachable_prediction, c.teachable_confidence, c.original_ai_verdict,
               c.final_verdict, c.created_at, u.full_name as customer_name,
               p.product_name, p.category, p.brand, p.serial_number
        FROM claims c
        JOIN users u ON c.user_id = u.user_id
        JOIN products p ON c.product_id = p.product_id
        WHERE c.final_verdict = 'Manual Review Required' OR c.claim_status IN ('Submitted', 'Manual Review', 'Under Evaluation')
        ORDER BY c.created_at DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def submit_reviewer_decision(claim_id: str, reviewer_user_id: int, reviewer_name: str, new_verdict: str, override_reason: str):
    """
    Updates a claim verdict (Approved / Rejected / Overridden) and logs 
    audit entry preserving original AI result alongside override reason.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Fetch existing AI verdict
        cursor.execute("SELECT final_verdict, original_ai_verdict, ml_prediction FROM claims WHERE claim_id = ?", (claim_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False, f"Claim {claim_id} not found."
            
        current_verdict = row['final_verdict']
        orig_ai = row['original_ai_verdict'] or current_verdict or row['ml_prediction'] or 'N/A'

        new_status = 'Approved' if new_verdict in ['Approved', 'Likely Valid'] else ('Rejected' if new_verdict in ['Rejected', 'Likely Invalid'] else 'Closed')

        cursor.execute("""
            UPDATE claims
            SET final_verdict = ?,
                claim_status = ?,
                override_reason = ?,
                reviewer_comments = ?,
                reviewed_by = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE claim_id = ?
        """, (new_verdict, new_status, override_reason, override_reason, reviewer_name, claim_id))

        # Insert into Audit Trail Log (Req 10)
        audit_details = (
            f"Claim '{claim_id}': Reviewer '{reviewer_name}' (ID: {reviewer_user_id}) set verdict to '{new_verdict}'. "
            f"Original AI Verdict: '{orig_ai}'. Reason: {override_reason}"
        )
        cursor.execute("""
            INSERT INTO audit_logs (user_id, action, details)
            VALUES (?, ?, ?)
        """, (reviewer_user_id, f"CLAIM_VERDICT_OVERRIDE_{new_status.upper()}", audit_details))


        conn.commit()
        conn.close()
        return True, f"Claim {claim_id} updated successfully to '{new_verdict}'!"
    except Exception as e:
        conn.close()
        return False, f"Failed to submit decision: {str(e)}"

def get_audit_trail_logs(limit=50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT log_id, user_id, action, details, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == '__main__':
    init_db()
    print("Database initialized with Reviewer Queue & Audit Trail support!")