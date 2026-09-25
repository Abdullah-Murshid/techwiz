import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Fix path to load src/ and policies/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import ClaimPreprocessor
from policies.rules import RuleEngine

# 1. Page Configuration
st.set_page_config(
    page_title="AssureX Claim Engine",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AssureX Automated Claim Verification Engine")
st.markdown("Automated claim validation, policy risk flagging, and machine learning triage system.")

# 2. Load Models & Dependencies
@st.cache_resource
def load_assets():
    model_path = os.path.join('model', 'best_tabular_model.joblib')
    prep_path = os.path.join('model', 'preprocessor.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        st.error("Model or preprocessor not found! Run 'uv run python src/train_models.py' first.")
        st.stop()
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    rule_engine = RuleEngine()
    return model, preprocessor, rule_engine

model, preprocessor, rule_engine = load_assets()

# 3. Sidebar Input Options
st.sidebar.header("📋 Claim Input Options")
input_method = st.sidebar.radio("Select Input Method:", ["Interactive Claim Form", "Batch CSV Validation"])

if input_method == "Interactive Claim Form":
    st.subheader("Manual Claim Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        claim_id = st.text_input("Claim ID", "CLM-8801")
        customer_name = st.text_input("Customer Name", "Jane Doe")
        category = st.selectbox("Product Category", ["Washing Machine", "Laptop", "Refrigerator", "Smartphone"])
        brand = st.text_input("Brand", "Samsung")
        model_num = st.text_input("Model Number", "SAM-801")
        serial_num = st.text_input("Serial Number", "SA-991823")

    with col2:
        price = st.number_input("Purchase Price ($)", min_value=10, max_value=5000, value=850)
        fault_type = st.selectbox("Fault Type", ['Motor Failure', 'Screen Damage', 'Power Failure', 'Water Leakage', 'Overheating', 'Board Defect'])
        warranty_duration = st.number_input("Standard Warranty (Months)", value=24)
        product_age = st.number_input("Product Age (Months)", value=10)
        remaining_warranty = st.number_input("Remaining Warranty (Months)", value=14)

    with col3:
        st.markdown("**Compliance & Policy Verification**")
        has_receipt = st.checkbox("Receipt Uploaded", value=True)
        has_warranty_card = st.checkbox("Warranty Card Uploaded", value=True)
        has_damage_photo = st.checkbox("Damage Photo Uploaded", value=True)
        serial_match = st.checkbox("Serial Number Matched", value=True)
        unauth_repairs = st.checkbox("Previous Unauthorized Repairs", value=False)
        duplicate_flag = st.checkbox("Duplicate Claim Detected", value=False)
        date_contradiction = st.checkbox("Date Contradiction Flag", value=False)

    if st.button("Evaluate Claim", type="primary"):
        # Construct DataFrame record
        raw_record = {
            'Claim_ID': claim_id,
            'Customer_Name': customer_name,
            'Product_Category': category,
            'Brand': brand,
            'Model_Number': model_num,
            'Serial_Number': serial_num,
            'Purchase_Price': price,
            'Product_Age_Months': product_age,
            'Warranty_Duration_Months': warranty_duration,
            'Remaining_Warranty_Months': remaining_warranty,
            'Fault_Type': fault_type,
            'Has_Receipt': has_receipt,
            'Has_Warranty_Card': has_warranty_card,
            'Has_Damage_Photo': has_damage_photo,
            'Serial_Number_Match': serial_match,
            'Previous_Unauthorized_Repairs': unauth_repairs,
            'Duplicate_Claim_Flag': duplicate_flag,
            'Date_Contradiction_Flag': date_contradiction,
            'Missing_Documents_Count': (0 if has_receipt else 1) + (0 if has_warranty_card else 1) + (0 if has_damage_photo else 1),
            'Claim_Class': 'Valid Claim'  # Placeholder for preprocessing transformation
        }
        
        df_single = pd.DataFrame([raw_record])
        
        # ML Inference
        X_vec = preprocessor.transform(df_single)
        ml_pred_idx = model.predict(X_vec)[0]
        ml_probs = model.predict_proba(X_vec)[0]
        ml_class = preprocessor.label_encoder.classes_[ml_pred_idx]
        ml_confidence = float(np.max(ml_probs))
        
        # Rule Engine Evaluation
        policy_result = rule_engine.evaluate_claim(raw_record, ml_class, ml_confidence)
        
        st.markdown("---")
        st.header("🔍 Verification Results")
        
        r_col1, r_col2, r_col3 = st.columns(3)
        
        with r_col1:
            st.metric("ML Initial Prediction", ml_class)
            st.metric("ML Confidence Score", f"{ml_confidence * 100:.1f}%")
            
        with r_col2:
            final_status = policy_result['final_status']
            if final_status == "Valid Claim":
                st.success(f"### Final Verdict: {final_status}")
            elif final_status == "Invalid Claim":
                st.error(f"### Final Verdict: {final_status}")
            else:
                st.warning(f"### Final Verdict: {final_status}")
                
        with r_col3:
            st.markdown("**Action Required:**")
            st.info(policy_result['action_required'])
            
        if policy_result['rule_flags']:
            st.subheader("🚩 Active Policy Flags")
            for flag in policy_result['rule_flags']:
                st.warning(flag)

else:
    st.subheader("Batch CSV Claim Processing")
    uploaded_file = st.file_uploader("Upload CSV file containing claims data", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} claims.")
        
        if st.button("Process Batch Claims"):
            results = []
            
            for _, row in batch_df.iterrows():
                row_dict = row.to_dict()
                df_single = pd.DataFrame([row_dict])
                
                try:
                    X_vec = preprocessor.transform(df_single)
                    ml_pred_idx = model.predict(X_vec)[0]
                    ml_probs = model.predict_proba(X_vec)[0]
                    ml_class = preprocessor.label_encoder.classes_[ml_pred_idx]
                    ml_confidence = float(np.max(ml_probs))
                    
                    eval_res = rule_engine.evaluate_claim(row_dict, ml_class, ml_confidence)
                    
                    results.append({
                        'Claim_ID': row_dict.get('Claim_ID', 'N/A'),
                        'ML_Prediction': ml_class,
                        'Confidence': f"{ml_confidence*100:.1f}%",
                        'Final_Verdict': eval_res['final_status'],
                        'Action_Required': eval_res['action_required'],
                        'Flags': " | ".join(eval_res['rule_flags']) if eval_res['rule_flags'] else "None"
                    })
                except Exception as e:
                    results.append({
                        'Claim_ID': row_dict.get('Claim_ID', 'N/A'),
                        'ML_Prediction': 'ERROR',
                        'Confidence': 'N/A',
                        'Final_Verdict': 'ERROR',
                        'Action_Required': str(e),
                        'Flags': 'Processing Error'
                    })
                    
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            csv_download = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Verification Results CSV",
                data=csv_download,
                file_name="assurex_batch_results.csv",
                mime="text/csv"
            )