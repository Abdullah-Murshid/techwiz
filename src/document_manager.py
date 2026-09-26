import os
import hashlib
import sqlite3
import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import get_connection
from src.auth import has_permission

def init_documents_table():
    """Initializes documents table in database if missing."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT NOT NULL,
            product_id INTEGER,
            user_id INTEGER NOT NULL,
            doc_type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
        )
    ''')
    conn.commit()
    conn.close()

def save_claim_document(user_id: int, role: str, claim_id: str, product_id: int, doc_type: str, file_name: str, file_bytes: bytes) -> tuple:
    """Stores uploaded document file under claim directory with MD5 hash tracking."""
    if not has_permission(role, 'upload_documents'):
        return False, "Permission denied: your role cannot upload documents."

    init_documents_table()
    
    upload_dir = os.path.join('data', 'uploads', claim_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{doc_type}_{file_name}")
    
    # Save file to disk
    with open(file_path, "wb") as f:
        f.write(file_bytes)
        
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_size = len(file_bytes)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO documents (
                claim_id, product_id, user_id, doc_type, file_name, file_path, file_hash, file_size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (claim_id, product_id, int(user_id), doc_type, file_name, file_path, file_hash, file_size))
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, doc_id, "Document uploaded and stored successfully!"
    except Exception as e:
        conn.close()
        return False, None, f"Failed to store document: {str(e)}"

def get_claim_documents(user_id: int, role: str, claim_id: str) -> list:
    """Retrieves list of documents for a claim gated by RBAC."""
    init_documents_table()
    conn = get_connection()
    cursor = conn.cursor()
    
    if role in ['Admin', 'Reviewer', 'Service Center']:
        cursor.execute("SELECT * FROM documents WHERE claim_id = ? ORDER BY uploaded_at DESC", (claim_id,))
    else:
        cursor.execute("SELECT * FROM documents WHERE claim_id = ? AND user_id = ? ORDER BY uploaded_at DESC", (claim_id, user_id))

    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_claim_document(user_id: int, role: str, doc_id: int) -> tuple:
    """Deletes a document file and DB record gated by role permissions."""
    init_documents_table()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        return False, "Document not found."

    # RBAC check: Customer can delete only their own uploaded documents
    if role == 'Customer' and doc['user_id'] != user_id:
        conn.close()
        return False, "Permission denied: cannot delete documents owned by another user."

    # Remove file from disk
    file_path = doc['file_path']
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return True, "Document deleted successfully."

def replace_claim_document(user_id: int, role: str, doc_id: int, new_file_name: str, new_file_bytes: bytes) -> tuple:
    """Replaces an existing document with a new file binary."""
    init_documents_table()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        return False, "Document not found."

    if role == 'Customer' and doc['user_id'] != user_id:
        conn.close()
        return False, "Permission denied."

    file_path = doc['file_path']
    with open(file_path, "wb") as f:
        f.write(new_file_bytes)

    new_hash = hashlib.md5(new_file_bytes).hexdigest()
    new_size = len(new_file_bytes)

    cursor.execute("""
        UPDATE documents
        SET file_name = ?, file_hash = ?, file_size_bytes = ?, uploaded_at = CURRENT_TIMESTAMP
        WHERE doc_id = ?
    """, (new_file_name, new_hash, new_size, doc_id))

    conn.commit()
    conn.close()
    return True, "Document replaced successfully."

if __name__ == '__main__':
    init_documents_table()
    print("Document Manager initialized.")
