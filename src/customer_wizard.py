import streamlit as st
import pandas as pd
import os
import sqlite3
import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_engine import ReceiptOCRProcessor
from src.pipeline import ClaimVerificationPipeline
from src.document_manager import replace_claim_document, save_claim_document

DB_PATH = os.path.join('database', 'assurex.db')

def save_uploaded_file(uploaded_file, claim_id, file_type):
    upload_dir = os.path.join('data', 'uploads', claim_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{file_type}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def record_claim_to_db(claim_data, user, pipeline_result=None):
    """Persists product details and customer claim records to SQLite DB."""
    if not os.path.exists(DB_PATH):
        return False, "Database file not found."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        user_id = user.get('user_id') or user.get('id')
        if user_id is None:
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (user.get('username'),))
            user_row = cursor.fetchone()
            if user_row:
                user_id = user_row[0]
            else:
                user_id = 1

        # 1. Product Insertion / Lookup
        cursor.execute("SELECT product_id FROM products WHERE serial_number = ?", (claim_data['serial_number'],))
        row = cursor.fetchone()
        
        if row:
            product_id = row[0]
        else:
            product_name = f"{claim_data.get('brand', '')} {claim_data.get('model_number', '')}".strip()
            cursor.execute("""
                INSERT INTO products (
                    user_id, product_name, category, brand, model_number, serial_number, 
                    purchase_price, warranty_duration_months, purchase_date, retailer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(user_id),
                product_name if product_name else "Unknown Device",
                claim_data.get('category', 'General'),
                claim_data.get('brand', 'Unknown'),
                claim_data.get('model_number', 'N/A'),
                claim_data.get('serial_number', 'N/A'),
                float(claim_data.get('purchase_price', 0.0)),
                int(claim_data.get('warranty_duration', 12)),
                str(claim_data.get('purchase_date', datetime.date.today())),
                claim_data.get('retailer', 'Authorized Retailer')
            ))
            product_id = cursor.lastrowid

        # Extract predictions from pipeline result if available
        if pipeline_result and 'python_model' in pipeline_result and 'teachable_machine' in pipeline_result:
            py_pred = pipeline_result['python_model'].get('prediction', 'Unknown')
            py_conf = float(pipeline_result['python_model'].get('confidence', 0.0) or 0.0)
            tm_pred = pipeline_result['teachable_machine'].get('prediction', 'Unknown')
            tm_conf = float(pipeline_result['teachable_machine'].get('confidence', 0.0) or 0.0)
            final_verdict = pipeline_result.get('final_verdict', 'Manual Review Required')
            status_map = {
                'Likely Valid': 'Submitted',
                'Likely Invalid': 'Closed',
                'Manual Review Required': 'Manual Review'
            }
            claim_status = status_map.get(final_verdict, 'Manual Review')
        else:
            # Explicit failure handling: Flag claim distinctly as 'Processing Failed' rather than silent fake verdict
            py_pred = None
            py_conf = None
            tm_pred = None
            tm_conf = None
            final_verdict = 'Processing Failed'
            claim_status = 'Processing Failed'

        fault_desc = claim_data.get('fault_description')
        if fault_desc is None:
            fault_desc = ""
        else:
            fault_desc = str(fault_desc)

        # 2. Insert claim record into database
        cursor.execute("""
            INSERT OR REPLACE INTO claims (
                claim_id, user_id, product_id, fault_type, fault_description, claim_status,
                ml_prediction, ml_confidence, teachable_prediction, teachable_confidence,
                original_ai_verdict, final_verdict, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(claim_data['claim_id']),
            int(user_id),
            int(product_id),
            str(claim_data.get('fault_type', 'General Defect')),
            fault_desc,
            claim_status,
            py_pred,
            py_conf,
            tm_pred,
            tm_conf,
            final_verdict,
            final_verdict,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        conn.commit()
        conn.close()
        return True, "Claim submitted successfully!"
    except Exception as e:
        conn.close()
        return False, f"Failed to record claim: {str(e)}"

def render_customer_wizard(user):
    st.title("📝 Submit & Track Warranty Claim")
    
    if 'wizard_step' not in st.session_state:
        st.session_state['wizard_step'] = 1
    if 'ocr_data' not in st.session_state:
        st.session_state['ocr_data'] = {}
    if 'claim_draft' not in st.session_state:
        st.session_state['claim_draft'] = {
            'claim_id': f"CLM-{int(datetime.datetime.now().timestamp()) % 100000}",
            'receipt_uploaded': False,
            'warranty_uploaded': False,
            'photo_uploaded': False
        }

    steps = ["1. Product Details", "2. Fault & Damage", "3. Document Uploads & OCR", "4. OCR Verification & Multi-AI Verdict"]
    current_step = st.session_state['wizard_step']
    
    progress_val = int((current_step / 4) * 100)
    st.progress(progress_val)
    st.caption(f"**Step {current_step} of 4:** {steps[current_step - 1]}")
    st.markdown("---")

    draft = st.session_state['claim_draft']

    # --- STEP 1: PRODUCT DETAILS ---
    if current_step == 1:
        st.subheader("📦 Step 1: Product Information")
        col1, col2 = st.columns(2)
        with col1:
            draft['category'] = st.selectbox("Product Category", ["Washing Machine", "Laptop", "Refrigerator", "Smartphone"], index=0)
            draft['brand'] = st.text_input("Brand / Manufacturer", value=draft.get('brand', 'Samsung'))
            draft['model_number'] = st.text_input("Model Number", value=draft.get('model_number', 'SAM-801'))
            draft['retailer'] = st.text_input("Retailer / Store Name", value=draft.get('retailer', 'Official Store'))
        with col2:
            draft['serial_number'] = st.text_input("Serial Number", value=draft.get('serial_number', 'SA-991823'))
            draft['purchase_price'] = st.number_input("Purchase Price ($)", min_value=10, max_value=10000, value=draft.get('purchase_price', 850))
            draft['purchase_date'] = st.date_input("Purchase Date", value=datetime.date.today() - datetime.timedelta(days=300))

        st.markdown("---")
        if st.button("Next: Fault Information ➡️", type="primary"):
            if not draft['brand'] or not draft['serial_number']:
                st.error("Please fill in all required product fields.")
            else:
                st.session_state['wizard_step'] = 2
                st.rerun()

    # --- STEP 2: FAULT DETAILS ---
    elif current_step == 2:
        st.subheader("🛠️ Step 2: Fault & Damage Details")
        col1, col2 = st.columns(2)
        with col1:
            draft['fault_type'] = st.selectbox(
                "Primary Issue Category",
                ['Motor Failure', 'Screen Damage', 'Power Failure', 'Water Leakage', 'Overheating', 'Board Defect'],
                index=0
            )
            draft['product_age'] = st.number_input("Product Age (Months)", min_value=0, max_value=120, value=draft.get('product_age', 10))
        with col2:
            draft['warranty_duration'] = st.number_input("Standard Warranty Duration (Months)", min_value=1, max_value=60, value=draft.get('warranty_duration', 24))
            draft['remaining_warranty'] = max(0, draft['warranty_duration'] - draft['product_age'])
            st.metric("Estimated Remaining Warranty", f"{draft['remaining_warranty']} Months")

        draft['fault_description'] = st.text_area("Detailed Problem Description", value=draft.get('fault_description', 'Device stopped operating during regular use.'))

        st.markdown("---")
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("⬅️ Back"):
                st.session_state['wizard_step'] = 1
                st.rerun()
        with c2:
            if st.button("Next: Document Uploads & OCR ➡️", type="primary"):
                st.session_state['wizard_step'] = 3
                st.rerun()

    # --- STEP 3: DOCUMENT UPLOADS & OCR EXTRACTION ---
    elif current_step == 3:
        st.subheader("📎 Step 3: Supporting Document Uploads & OCR Extraction")
        st.info("Upload receipt images for automatic OCR field extraction.")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🧾 Purchase Receipt")
            receipt_file = st.file_uploader("Upload Receipt (PDF/JPG/PNG)", type=['png', 'jpg', 'jpeg', 'pdf'], key='receipt')
            if receipt_file:
                path = save_uploaded_file(receipt_file, draft['claim_id'], "receipt")
                draft['receipt_uploaded'] = True
                draft['receipt_path'] = path
                
                # Run OCR Extraction
                with st.spinner("Running OCR text extraction on receipt..."):
                    ocr_engine = ReceiptOCRProcessor()
                    raw_text = ocr_engine.extract_text_from_image(path)
                    st.session_state['ocr_data'] = ocr_engine.parse_receipt_data(raw_text)
                st.success("OCR Extraction Completed!")

        with col2:
            st.markdown("#### 📇 Warranty Card")
            warranty_file = st.file_uploader("Upload Warranty Card", type=['png', 'jpg', 'jpeg', 'pdf'], key='warranty')
            if warranty_file:
                save_uploaded_file(warranty_file, draft['claim_id'], "warranty")
                draft['warranty_uploaded'] = True
                st.success("Warranty Card attached")

        with col3:
            st.markdown("#### 📷 Damage Evidence")
            photo_file = st.file_uploader("Upload Damage Photo", type=['png', 'jpg', 'jpeg'], key='damage')
            if photo_file:
                save_uploaded_file(photo_file, draft['claim_id'], "damage_photo")
                draft['photo_uploaded'] = True
                st.success("Damage Photo attached")

        # Document Replacement Widget (Req 39)
        with st.expander("🔄 Document Replacement & Revision (Req 39)", expanded=False):
            st.caption("Replace an existing uploaded document with an updated file.")
            doc_id_val = st.number_input("Existing Document ID to Replace", min_value=1, value=1, step=1)
            replacement_file = st.file_uploader("Upload Replacement Document File", type=['png', 'jpg', 'jpeg', 'pdf'], key='replacement_file')
            if st.button("Submit Document Replacement"):
                if replacement_file:
                    u_id = user.get('user_id') or user.get('id') or 1
                    u_role = user.get('role', 'Customer')
                    success, msg = replace_claim_document(u_id, u_role, int(doc_id_val), replacement_file.name, replacement_file.getbuffer())
                    if success:
                        st.success(f"Document #{doc_id_val} replaced successfully!")
                    else:
                        st.error(msg)
                else:
                    st.warning("Please select a replacement file first.")

        # Show OCR preview if extracted
        ocr_data = st.session_state.get('ocr_data', {})
        if ocr_data.get('extracted_serial') or ocr_data.get('extracted_date'):
            st.markdown("---")
            st.markdown("### 🔍 OCR Extracted Field Preview")
            st.json({
                "OCR Serial Number": ocr_data.get('extracted_serial') or "Not Detected",
                "OCR Purchase Date": ocr_data.get('extracted_date') or "Not Detected",
                "OCR Amount": ocr_data.get('extracted_amount') or "Not Detected",
                "OCR Retailer": ocr_data.get('extracted_retailer') or "Not Detected"
            })

        st.markdown("---")
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("⬅️ Back"):
                st.session_state['wizard_step'] = 2
                st.rerun()
        with c2:
            if st.button("Next: OCR Verification & Verdict ➡️", type="primary"):
                st.session_state['wizard_step'] = 4
                st.rerun()

    # --- STEP 4: OCR VERIFICATION & MULTI-AI VERDICT ---
    elif current_step == 4:
        st.subheader("🔍 Step 4: OCR Verification & Multi-AI Verdict Consensus")
        
        ocr_data = st.session_state.get('ocr_data', {})

        # Verification & Correction Screen (Req 1)
        st.markdown("### 🛠️ OCR Field Verification & Correction Screen")
        st.caption("Compare customer form inputs against OCR extracted fields. Edit any discrepancy before finalizing.")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### Customer Form Inputs")
            confirmed_sn = st.text_input("Serial Number", value=draft.get('serial_number', 'SA-991823'))
            confirmed_price = st.number_input("Purchase Price ($)", value=float(draft.get('purchase_price', 850)))
            confirmed_retailer = st.text_input("Retailer Name", value=draft.get('retailer', 'Official Store'))

        with col_v2:
            st.markdown("#### OCR Extracted Document Values")
            st.text_input("OCR Extracted Serial", value=str(ocr_data.get('extracted_serial') or "Not Found"), disabled=True)
            st.text_input("OCR Extracted Amount", value=str(ocr_data.get('extracted_amount') or "Not Found"), disabled=True)
            st.text_input("OCR Extracted Retailer", value=str(ocr_data.get('extracted_retailer') or "Not Found"), disabled=True)

        serial_matched = (confirmed_sn.replace(" ", "").upper() == str(ocr_data.get('extracted_serial', '')).replace(" ", "").upper()) if ocr_data.get('extracted_serial') else True
        if not serial_matched:
            st.warning("⚠️ Serial Number Mismatch Alert: Form Serial Number differs from OCR extracted receipt serial.")

        # Update draft with confirmed values
        draft['serial_number'] = confirmed_sn
        draft['purchase_price'] = confirmed_price
        draft['retailer'] = confirmed_retailer
        draft['serial_match'] = serial_matched

        # Run Unified Multi-Modal Pipeline Engine
        st.markdown("---")
        st.markdown("### 🤖 Multi-AI & Rule Engine Consensus Evaluation")

        claim_dict = {
            'Claim_ID': draft['claim_id'],
            'Customer_Name': user['full_name'],
            'Product_Category': draft.get('category', 'Laptop'),
            'Brand': draft.get('brand', 'Samsung'),
            'Model_Number': draft.get('model_number', 'SAM-801'),
            'Serial_Number': draft.get('serial_number', 'SA-991823'),
            'Purchase_Price': draft.get('purchase_price', 850),
            'Purchase_Date': str(draft.get('purchase_date', datetime.date.today())),
            'Claim_Date': datetime.date.today().strftime('%Y-%m-%d'),
            'Product_Age_Months': draft.get('product_age', 10),
            'Warranty_Duration_Months': draft.get('warranty_duration', 24),
            'Remaining_Warranty_Months': draft.get('remaining_warranty', 14),
            'Fault_Type': draft.get('fault_type', 'Motor Failure'),
            'Repair_History': 'None',
            'Has_Receipt': draft.get('receipt_uploaded', True),
            'Has_Warranty_Card': draft.get('warranty_uploaded', True),
            'Has_Damage_Photo': draft.get('photo_uploaded', True),
            'Serial_Number_Match': serial_matched,
            'Previous_Unauthorized_Repairs': False,
            'Duplicate_Claim_Flag': False,
            'Date_Contradiction_Flag': False,
            'Missing_Documents_Count': (0 if draft.get('receipt_uploaded') else 1) + (0 if draft.get('warranty_uploaded') else 1) + (0 if draft.get('photo_uploaded') else 1)
        }

        pipeline_res = None
        try:
            pipeline = ClaimVerificationPipeline()
            pipeline_res = pipeline.process_claim(claim_dict)
            print(f"[LOG] ClaimVerificationPipeline processed claim '{draft.get('claim_id')}' successfully.")
        except Exception as e:
            import traceback
            err_tb = traceback.format_exc()
            print(f"[ERROR] ClaimVerificationPipeline failed for claim '{draft.get('claim_id')}': {str(e)}\n{err_tb}")
            st.error(f"⚠️ Multi-AI Pipeline Evaluation Failed: {str(e)}. Please review input details and retry.")

        if pipeline_res is None:
            st.warning("AI Pipeline evaluation did not complete. Fix errors above before submitting.")
            return

        # Render Verification Dashboard Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Python ML Prediction", f"{pipeline_res['python_model']['prediction']}", f"{pipeline_res['python_model']['confidence']*100:.1f}% Conf")
        with c2:
            st.metric("Teachable Machine Visual AI", f"{pipeline_res['teachable_machine']['prediction']}", f"{pipeline_res['teachable_machine']['confidence']*100:.1f}% Conf")
        with c3:
            st.metric("Model Agreement Tier", pipeline_res['model_comparison']['comparison_status'], f"|Δconf|: {pipeline_res['model_comparison']['confidence_delta']*100:.1f}%")

        verdict = pipeline_res['final_verdict']
        if verdict == "Likely Valid":
            st.success(f"### 🎉 Final Verdict: {verdict}")
        elif verdict == "Likely Invalid":
            st.error(f"### 🛑 Final Verdict: {verdict}")
        else:
            st.warning(f"### ⚠️ Final Verdict: {verdict}")

        st.markdown(f"**Required System Action:** `{pipeline_res['action_required']}`")

        # Render Decision Explanation (Supporting & Opposing Factors, Rules Passed/Failed)
        exp = pipeline_res['decision_explanation']
        
        with st.expander("📊 View Detailed Decision Explanation & Factor Breakdown", expanded=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown("#### ✅ Supporting Factors")
                for sf in exp['supporting_factors']:
                    st.write(f"- {sf}")
            with f_col2:
                st.markdown("#### ⚠️ Opposing / Risk Factors")
                if exp['opposing_factors']:
                    for of in exp['opposing_factors']:
                        st.write(f"- {of}")
                else:
                    st.write("None identified.")

            st.markdown("---")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("#### 📜 Policy Rules Passed")
                for rp in exp['rules_passed']:
                    st.write(f"✔️ `{rp}`")
            with r_col2:
                st.markdown("#### ❌ Policy Rules Failed / Flagged")
                if exp['rules_failed']:
                    for rf in exp['rules_failed']:
                        st.write(f"❌ `{rf}`")
                else:
                    st.write("None.")

            if exp['required_additional_evidence']:
                st.markdown("#### 📁 Required Additional Evidence Checklist")
                for ev in exp['required_additional_evidence']:
                    st.write(f"📌 `{ev}`")

        st.markdown("---")
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("⬅️ Back"):
                st.session_state['wizard_step'] = 3
                st.rerun()
        with c2:
            if st.button("🚀 Finalize & Submit Claim Record", type="primary"):
                success, msg = record_claim_to_db(draft, user, pipeline_result=pipeline_res)
                if success:
                    st.balloons()
                    st.success(f"Claim **{draft['claim_id']}** successfully recorded to system database!")
                    st.session_state['wizard_step'] = 1
                    st.session_state['claim_draft'] = {
                        'claim_id': f"CLM-{int(datetime.datetime.now().timestamp()) % 100000}",
                        'receipt_uploaded': False,
                        'warranty_uploaded': False,
                        'photo_uploaded': False
                    }
                else:
                    st.error(msg)