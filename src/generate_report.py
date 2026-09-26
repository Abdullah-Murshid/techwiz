import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.pipeline import ClaimVerificationPipeline

def generate_30_claim_report():
    os.makedirs('reports', exist_ok=True)
    pipeline = ClaimVerificationPipeline()
    
    # Load 30 test claims
    test_df = pd.read_csv('data/test/test_claims.csv').head(30)
    
    results = []
    
    for _, row in test_df.iterrows():
        claim_dict = row.to_dict()
        res = pipeline.process_claim(claim_dict)
        
        results.append({
            'Claim_ID': claim_dict.get('Claim_ID'),
            'Ground_Truth_Class': claim_dict.get('Claim_Class'),
            'ML_Tabular_Prediction': res['python_model']['prediction'],
            'ML_Confidence': f"{res['python_model']['confidence']*100:.1f}%",
            'TM_Visual_Prediction': res['teachable_machine']['prediction'],
            'TM_Confidence': f"{res['teachable_machine']['confidence']*100:.1f}%",
            'Model_Consensus': res['model_comparison']['comparison_status'],
            'Final_Verdict': res['final_verdict'],
            'Action_Required': res['action_required'],
            'Policy_Flags': " | ".join(res['rule_engine']['rule_flags']) if res['rule_engine']['rule_flags'] else "None"
        })

    report_df = pd.DataFrame(results)
    
    # Save CSV Report
    report_df.to_csv('reports/30_claim_model_comparison.csv', index=False)
    
    # Generate Markdown Summary
    md_content = "# AssureX Claim Verification Engine - 30-Claim Model Comparison Report\n\n"
    md_content += "## System Performance Summary\n\n"
    
    verdict_map = {
        'Valid Claim': 'Likely Valid',
        'Invalid Claim': 'Likely Invalid',
        'Manual Review': 'Manual Review Required'
    }
    report_df['Mapped_Ground_Truth'] = report_df['Ground_Truth_Class'].map(verdict_map)
    correct_preds = (report_df['Mapped_Ground_Truth'] == report_df['Final_Verdict']).sum()
    accuracy = (correct_preds / len(report_df)) * 100

    
    md_content += f"- **Total Claims Evaluated:** 30\n"
    md_content += f"- **System Output Accuracy:** {accuracy:.2f}%\n\n"
    md_content += "## Detailed Comparison Table\n\n"
    md_content += report_df.to_markdown(index=False)
    
    with open('reports/30_claim_model_comparison.md', 'w') as f:
        f.write(md_content)
        
    print(f"30-Claim Report generated successfully! Accuracy: {accuracy:.2f}%")
    print("Files saved to 'reports/30_claim_model_comparison.csv' and 'reports/30_claim_model_comparison.md'")

if __name__ == '__main__':
    generate_30_claim_report()