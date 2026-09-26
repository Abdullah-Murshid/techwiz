import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import init_db, create_user, authenticate_user, get_connection

from src.document_manager import (
    init_documents_table,
    save_claim_document,
    get_claim_documents,
    replace_claim_document,
    delete_claim_document
)

def test_module_3_document_organization():
    init_db()
    init_documents_table()

    # Clean existing test records
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE claim_id = 'CLM-DOC-TEST-01'")
    conn.commit()
    conn.close()

    # 1. Register test user

    create_user('doc_user1', 'doc1@test.com', 'pass123', 'Document Owner', 'Customer')
    user = authenticate_user('doc_user1', 'pass123')
    user_id = user['user_id']
    claim_id = 'CLM-DOC-TEST-01'

    # 2. Upload Document
    dummy_bytes = b"SAMPLE RECEIPT DATA INVOICE #12345"
    success, doc_id, msg = save_claim_document(
        user_id=user_id,
        role='Customer',
        claim_id=claim_id,
        product_id=1,
        doc_type='receipt',
        file_name='receipt_12345.pdf',
        file_bytes=dummy_bytes
    )
    assert success is True, f"Save failed: {msg}"
    assert doc_id is not None

    # 3. Retrieve Documents
    docs = get_claim_documents(user_id, 'Customer', claim_id)
    assert len(docs) == 1, f"Expected 1 doc for user {user_id}, got {docs}"

    assert docs[0]['file_name'] == 'receipt_12345.pdf'
    assert docs[0]['doc_type'] == 'receipt'

    # 4. RBAC Check: Another Customer Cannot Access This Claim's Documents
    create_user('doc_user2', 'doc2@test.com', 'pass123', 'Other Owner', 'Customer')
    other_user = authenticate_user('doc_user2', 'pass123')
    other_docs = get_claim_documents(other_user['user_id'], 'Customer', claim_id)
    assert len(other_docs) == 0, "RBAC violation: Customer accessed unowned claim docs"

    # 5. Replace Document
    new_bytes = b"UPDATED INVOICE CONTENT DATA V2"
    rep_success, rep_msg = replace_claim_document(user_id, 'Customer', doc_id, 'receipt_v2.pdf', new_bytes)
    assert rep_success is True, f"Replace failed: {rep_msg}"

    # 6. Delete Document
    del_success, del_msg = delete_claim_document(user_id, 'Customer', doc_id)
    assert del_success is True, f"Delete failed: {del_msg}"
    assert len(get_claim_documents(user_id, 'Customer', claim_id)) == 0

    print("Module 3 (Document Organization & Storage) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_3_document_organization()
