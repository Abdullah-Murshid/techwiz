import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import ClaimPreprocessor
from policies.rules import RuleEngine
from database.db import init_db, authenticate_user, create_user, get_manual_review_queue, submit_reviewer_decision
from src.admin_analytics import render_admin_analytics
from src.customer_wizard import render_customer_wizard
from src.pipeline import ClaimVerificationPipeline
from src.auth import generate_session_token, verify_session_token, invalidate_session_token
from src.notification_monitoring import get_user_notifications

init_db()

st.set_page_config(page_title="AssureX Claim Engine", page_icon="🛡️", layout="wide")

# Session State for User Authentication & Token Lifecycle (Req 33)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'session_token' not in st.session_state:
    st.session_state['session_token'] = None

# Verify Active Session Token
if st.session_state['logged_in'] and st.session_state['session_token']:
    session_user = verify_session_token(st.session_state['session_token'])
    if not session_user:
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
        st.session_state['session_token'] = None

# --- AUTHENTICATION SIDEBAR ---
st.sidebar.title("🔐 Account Portal")

if not st.session_state['logged_in']:
    auth_mode = st.sidebar.radio("Choose Action", ["Login", "Register"])
    
    if auth_mode == "Login":
        username = st.sidebar.text_input("Username", value="admin")
        password = st.sidebar.text_input("Password", type="password", value="admin123")
        if st.sidebar.button("Login"):
            user = authenticate_user(username, password)
            if user:
                token = generate_session_token(user)
                st.session_state['session_token'] = token
                st.session_state['logged_in'] = True
                st.session_state['user'] = dict(user)
                st.sidebar.success(f"Welcome, {user['full_name']}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid Username or Password.")
    else:
        new_user = st.sidebar.text_input("New Username")
        new_email = st.sidebar.text_input("Email")
        new_name = st.sidebar.text_input("Full Name")
        new_pass = st.sidebar.text_input("New Password", type="password")
        role = st.sidebar.selectbox("Role", ["Customer", "Service Center", "Reviewer", "Admin"])
        
        if st.sidebar.button("Register Account"):
            success, msg = create_user(new_user, new_email, new_pass, new_name, role)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
    st.stop()

# --- LOGGED IN USER VIEW ---
user = st.session_state['user']
st.sidebar.markdown(f"**Logged in as:** {user['full_name']} (`{user['role']}`)")
st.sidebar.caption(f"Token: `{st.session_state['session_token'][:12]}...`")

if st.sidebar.button("Logout"):
    if st.session_state.get('session_token'):
        invalidate_session_token(st.session_state['session_token'])
    st.session_state['logged_in'] = False
    st.session_state['user'] = None
    st.session_state['session_token'] = None
    st.rerun()

# User Notification Tray Widget (Req 41)
u_id = user.get('user_id') or user.get('id') or 1
notifications = get_user_notifications(u_id)
with st.sidebar.expander(f"🔔 Notification Tray ({len(notifications)})", expanded=False):
    if notifications:
        for n in notifications:
            st.markdown(f"**[{n.get('event_type')}]** {n.get('message')}")
            st.caption(f"Time: {n.get('created_at')}")
            st.markdown("---")
    else:
        st.info("No notifications.")

st.title("🛡️ AssureX Automated Claim Verification Engine")

# Helper function to render manual reviewer queue (Req 10)
def render_reviewer_queue():
    st.subheader("👥 Claims Manual Review & Audit Queue")
    queue_claims = get_manual_review_queue()
    
    if not queue_claims:
        st.success("🎉 No claims currently awaiting manual review!")
        return

    st.info(f"There are **{len(queue_claims)}** claims requiring reviewer evaluation.")

    # Select claim to review
    claim_options = {f"{c['claim_id']} - {c['customer_name']} ({c['product_name']})": c for c in queue_claims}
    selected_label = st.selectbox("Select Claim from Queue", list(claim_options.keys()))
    
    if selected_label:
        claim_data = claim_options[selected_label]
        
        st.markdown("---")
        st.markdown(f"### Claim Details: `{claim_data['claim_id']}`")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Customer:** {claim_data['customer_name']}")
            st.write(f"**Category:** {claim_data['category']}")
            st.write(f"**Brand / Device:** {claim_data['brand']} ({claim_data['product_name']})")
        with c2:
            st.write(f"**Serial Number:** `{claim_data['serial_number']}`")
            st.write(f"**Reported Fault:** {claim_data['fault_type']}")
            st.write(f"**Submission Date:** {claim_data['created_at']}")
        with c3:
            ml_pred = claim_data.get('ml_prediction') or 'N/A'
            ml_conf = float(claim_data.get('ml_confidence') or 0.0)
            tm_pred = claim_data.get('teachable_prediction') or 'N/A'
            tm_conf = float(claim_data.get('teachable_confidence') or 0.0)
            st.metric("Python ML Prediction", ml_pred, f"{ml_conf * 100:.1f}%")
            st.metric("Teachable Machine AI", tm_pred, f"{tm_conf * 100:.1f}%")

        st.markdown("---")
        st.markdown("### ⚖️ Reviewer Override & Decision Submission")
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            new_verdict = st.selectbox(
                "Reviewer Verdict Decision",
                ["Approved", "Rejected", "Override - Approved", "Override - Rejected", "Additional Info Required"],
                index=0
            )
        with col_act2:
            override_reason = st.text_area("Mandatory Reviewer Override Reason / Comments", value="")

        if st.button("Submit Reviewer Decision & Log Audit Record", type="primary"):
            if not override_reason:
                st.error("Please provide a mandatory override reason / comment before submitting.")
            else:
                clean_verdict = new_verdict.replace("Override - ", "")
                u_id = user.get('user_id') or user.get('id') or 1
                success, msg = submit_reviewer_decision(
                    claim_id=claim_data['claim_id'],
                    reviewer_user_id=u_id,
                    reviewer_name=user['full_name'],
                    new_verdict=clean_verdict,
                    override_reason=override_reason
                )
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

# Helper function to render claim evaluation tool
def render_claim_evaluation():
    st.subheader(f"Instant Claim Evaluation Portal ({user['role']})")

    col1, col2, col3 = st.columns(3)
    with col1:
        claim_id = st.text_input("Claim ID", "CLM-8801")
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
        st.markdown("**Compliance Checks**")
        has_receipt = st.checkbox("Receipt Uploaded", value=True)
        has_warranty_card = st.checkbox("Warranty Card Uploaded", value=True)
        has_damage_photo = st.checkbox("Damage Photo Uploaded", value=True)
        serial_match = st.checkbox("Serial Number Matched", value=True)
        unauth_repairs = st.checkbox("Previous Unauthorized Repairs", value=False)
        duplicate_flag = st.checkbox("Duplicate Claim Detected", value=False)
        date_contradiction = st.checkbox("Date Contradiction Flag", value=False)

    if st.button("Evaluate Claim End-to-End", type="primary"):
        raw_record = {
            'Claim_ID': claim_id,
            'Customer_Name': user['full_name'],
            'Product_Category': category,
            'Brand': brand,
            'Model_Number': model_num,
            'Serial_Number': serial_num,
            'Purchase_Price': price,
            'Purchase_Date': '2023-01-15',
            'Claim_Date': '2023-06-15',
            'Product_Age_Months': product_age,
            'Warranty_Duration_Months': warranty_duration,
            'Remaining_Warranty_Months': remaining_warranty,
            'Fault_Type': fault_type,
            'Repair_History': 'Unauthorized Repair' if unauth_repairs else 'None',
            'Has_Receipt': has_receipt,
            'Has_Warranty_Card': has_warranty_card,
            'Has_Damage_Photo': has_damage_photo,
            'Serial_Number_Match': serial_match,
            'Previous_Unauthorized_Repairs': unauth_repairs,
            'Duplicate_Claim_Flag': duplicate_flag,
            'Date_Contradiction_Flag': date_contradiction,
            'Missing_Documents_Count': (0 if has_receipt else 1) + (0 if has_warranty_card else 1) + (0 if has_damage_photo else 1),
            'Claim_Class': 'Valid Claim'
        }
        
        pipeline = ClaimVerificationPipeline()
        res = pipeline.process_claim(raw_record)
        
        st.markdown("---")
        st.header("🔍 Verification Results")
        
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.metric("Python ML Prediction", res['python_model']['prediction'], f"{res['python_model']['confidence']*100:.1f}%")
        with r_col2:
            st.metric("Teachable Machine Visual AI", res['teachable_machine']['prediction'], f"{res['teachable_machine']['confidence']*100:.1f}%")
        with r_col3:
            st.metric("Model Consensus", res['model_comparison']['comparison_status'])

        final_status = res['final_verdict']
        if final_status == "Likely Valid":
            st.success(f"### Final Verdict: {final_status}")
        elif final_status == "Likely Invalid":
            st.error(f"### Final Verdict: {final_status}")
        else:
            st.warning(f"### Final Verdict: {final_status}")

# --- ROLE-BASED NAVIGATION ---
if user['role'] in ['Admin', 'Reviewer']:
    tab_eval, tab_queue, tab_analytics = st.tabs([
        "⚡ Claim Evaluation Engine", 
        "👥 Reviewer Queue & Audit Workflow", 
        "📈 Analytics & Executive Dashboard"
    ])
    with tab_eval:
        render_claim_evaluation()
    with tab_queue:
        render_reviewer_queue()
    with tab_analytics:
        render_admin_analytics()
else:
    render_customer_wizard(user)