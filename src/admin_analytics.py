import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db import get_audit_trail_logs, submit_reviewer_decision, get_manual_review_queue
from src.product_warranty_manager import get_expiry_alert_threshold, set_expiry_alert_threshold

DB_PATH = os.path.join('database', 'assurex.db')

def fetch_claims_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT c.claim_id, c.fault_type, c.claim_status, c.ml_prediction, 
               c.ml_confidence, c.teachable_prediction, c.teachable_confidence,
               c.original_ai_verdict, c.final_verdict, c.override_reason,
               c.reviewer_comments, c.reviewed_by, c.reviewed_at, c.created_at,
               p.product_name, p.category, p.brand, p.serial_number, p.purchase_price,
               u.full_name as customer_name, u.email as customer_email
        FROM claims c
        LEFT JOIN products p ON c.product_id = p.product_id
        LEFT JOIN users u ON c.user_id = u.user_id
        ORDER BY c.created_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def generate_claim_report_markdown(claim_row):
    """Generates a downloadable structured evaluation report for a single claim."""
    md = f"# ASSUREX CLAIM ENGINE - INDIVIDUAL CLAIM REPORT\n"
    md += f"**Generated At:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"--- \n\n"
    md += f"### 📄 1. Claim Identification & Customer Information\n"
    md += f"- **Claim ID:** `{claim_row.get('claim_id')}`\n"
    md += f"- **Customer Name:** {claim_row.get('customer_name', 'N/A')}\n"
    md += f"- **Customer Email:** {claim_row.get('customer_email', 'N/A')}\n"
    md += f"- **Submission Date:** {claim_row.get('created_at')}\n"
    md += f"- **Current Lifecycle Status:** `{claim_row.get('claim_status')}`\n\n"
    
    md += f"### 📦 2. Product & Fault Information\n"
    md += f"- **Product Name:** {claim_row.get('product_name')}\n"
    md += f"- **Category:** {claim_row.get('category')}\n"
    md += f"- **Brand:** {claim_row.get('brand')}\n"
    md += f"- **Serial Number:** `{claim_row.get('serial_number')}`\n"
    md += f"- **Purchase Price:** ${claim_row.get('purchase_price')}\n"
    md += f"- **Reported Fault:** {claim_row.get('fault_type')}\n\n"

    ml_pred = claim_row.get('ml_prediction') or 'N/A'
    ml_conf = float(claim_row.get('ml_confidence') or 0.0)
    tm_pred = claim_row.get('teachable_prediction') or 'N/A'
    tm_conf = float(claim_row.get('teachable_confidence') or 0.0)
    md += f"- **Python Tabular Model Prediction:** {ml_pred} ({ml_conf * 100:.1f}% confidence)\n"
    md += f"- **Teachable Machine Visual Prediction:** {tm_pred} ({tm_conf * 100:.1f}% confidence)\n"
    md += f"- **Original AI System Verdict:** `{claim_row.get('original_ai_verdict', 'N/A')}`\n"
    md += f"- **Final System / Reviewer Verdict:** `{claim_row.get('final_verdict')}`\n\n"

    if claim_row.get('reviewed_by'):
        md += f"### 👤 4. Manual Reviewer Audit Log\n"
        md += f"- **Reviewed By:** {claim_row.get('reviewed_by')}\n"
        md += f"- **Reviewed At:** {claim_row.get('reviewed_at')}\n"
        md += f"- **Override Reason / Comments:** {claim_row.get('override_reason') or claim_row.get('reviewer_comments')}\n\n"
        
    md += f"--- \n"
    md += f"*AssureX Automated Claim Verification Engine - Confidential Evaluation Artifact*\n"
    return md

def render_admin_analytics():
    st.title("📈 Executive Analytics, Search & Audit Intelligence")
    
    df = fetch_claims_data()
    
    # --- MULTI-CRITERION SEARCH & FILTERING (Req 11) ---
    with st.expander("🔍 Search & Multi-Criterion Claim Filtering", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            search_query = st.text_input("Search Claim ID / Customer / Serial", "")
        with col_s2:
            status_filter = st.multiselect("Filter by Claim Status", df['claim_status'].unique() if not df.empty else [], default=[])
        with col_s3:
            verdict_filter = st.multiselect("Filter by Final Verdict", df['final_verdict'].unique() if not df.empty else [], default=[])

    if not df.empty:
        if search_query:
            q = search_query.lower()
            df = df[
                df['claim_id'].astype(str).str.lower().str.contains(q) |
                df['customer_name'].astype(str).str.lower().str.contains(q) |
                df['serial_number'].astype(str).str.lower().str.contains(q)
            ]
        if status_filter:
            df = df[df['claim_status'].isin(status_filter)]
        if verdict_filter:
            df = df[df['final_verdict'].isin(verdict_filter)]

    if df.empty:
        st.info("No claim records found matching criteria.")
        return

    # Key Performance Indicators
    st.subheader("📌 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    total_claims = len(df)
    valid_claims = len(df[df['final_verdict'].isin(['Likely Valid', 'Approved'])])
    invalid_claims = len(df[df['final_verdict'].isin(['Likely Invalid', 'Rejected'])])
    manual_reviews = len(df[df['final_verdict'].isin(['Manual Review Required', 'Manual Review'])])
    avg_conf = df['ml_confidence'].mean() * 100 if 'ml_confidence' in df and not df['ml_confidence'].isna().all() else 0.0

    kpi1.metric("Total Claims", total_claims)
    kpi2.metric("Valid / Approved", valid_claims, f"{valid_claims/total_claims*100:.1f}%" if total_claims else "0%")
    kpi3.metric("Invalid / Rejected", invalid_claims, f"{invalid_claims/total_claims*100:.1f}%" if total_claims else "0%", delta_color="inverse")
    kpi4.metric("Manual Review", manual_reviews)
    kpi5.metric("Avg ML Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # Visual Analytics Grid
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Final Verdict Distribution")
        fig_verdict = px.pie(
            df, 
            names='final_verdict', 
            color='final_verdict',
            color_discrete_map={
                'Likely Valid': '#2ecc71',
                'Approved': '#27ae60',
                'Likely Invalid': '#e74c3c',
                'Rejected': '#c0392b',
                'Manual Review Required': '#f1c40f'
            },
            hole=0.4
        )
        st.plotly_chart(fig_verdict, use_container_width=True)

    with col_right:
        st.subheader("🔧 Frequently Reported Faults")
        fault_counts = df['fault_type'].value_counts().reset_index()
        fault_counts.columns = ['Fault Type', 'Count']
        fig_faults = px.bar(
            fault_counts, 
            x='Fault Type', 
            y='Count', 
            color='Count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_faults, use_container_width=True)

    # Search & Claim Report Generator
    st.markdown("---")
    st.subheader("📄 Claim Records & Downloadable Reports")
    
    st.dataframe(
        df[['claim_id', 'customer_name', 'category', 'serial_number', 'fault_type', 'ml_prediction', 'teachable_prediction', 'final_verdict', 'claim_status', 'created_at']],
        use_container_width=True
    )

    selected_claim_id = st.selectbox("Select a Claim to Download Individual Report", df['claim_id'].tolist())
    if selected_claim_id:
        selected_row = df[df['claim_id'] == selected_claim_id].iloc[0].to_dict()
        report_md = generate_claim_report_markdown(selected_row)
        
        st.download_button(
            label=f"📥 Download Claim Report for {selected_claim_id} (.md)",
            data=report_md.encode('utf-8'),
            file_name=f"AssureX_Report_{selected_claim_id}.md",
            mime="text/markdown"
        )

    st.markdown("---")

    # Data Export Tools (Req 11 - CSV / Excel Export)
    st.subheader("📥 Export Complete Dataset")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=csv_data,
            file_name="assurex_claims_export.csv",
            mime="text/csv",
            type="primary"
        )

    with col_exp2:
        # Excel Export Buffer
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Claims_Analytics')
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="Download Full Dataset (Excel .xlsx)",
                data=excel_data,
                file_name="assurex_claims_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            pass

    # Audit Trail Logs Tab (Req 10)
    st.markdown("---")
    st.subheader("📜 System Audit Trail Logs")
    logs = get_audit_trail_logs(limit=50)
    if logs:
        st.table(pd.DataFrame(logs))
    else:
        st.info("No audit trail logs recorded yet.")

    # System Administration & Model Retraining Controls (Req 20 & Req 36)
    st.markdown("---")
    st.subheader("⚙️ System Administration & AI Model Controls")
    col_adm1, col_adm2 = st.columns(2)
    
    with col_adm1:
        st.markdown("**Tabular Model Retraining Trigger (Req 20)**")
        if st.button("🔄 Retrain Tabular ML Models", type="primary"):
            from src.train_models import train_and_evaluate
            with st.spinner("Retraining all tabular ML algorithms..."):
                train_and_evaluate()
            st.success("🎉 Tabular ML Models retrained and serialized successfully!")

    with col_adm2:
        st.markdown("**Warranty Expiry Alert Threshold (Req 36)**")
        curr_thresh = get_expiry_alert_threshold()
        new_thresh = st.number_input("Days Before Expiry Alert Threshold", min_value=1, max_value=180, value=curr_thresh)
        if st.button("Save Threshold Setting"):
            set_expiry_alert_threshold(new_thresh)
            st.success(f"Expiry alert threshold updated to {new_thresh} days!")