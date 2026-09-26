import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import ClaimVerificationPipeline

def test_module_6_unified_pipeline():
    pipeline = ClaimVerificationPipeline()

    test_claim = {
        'Claim_ID': 'TEST-UNI-01',
        'Customer_Name': 'John Doe',
        'Product_Category': 'Laptop',
        'Brand': 'Dell',
        'Model_Number': 'DEL-101',
        'Serial_Number': 'DE-123456',
        'Purchase_Price': 1000,
        'Purchase_Date': '2023-01-10',
        'Claim_Date': '2023-05-10',
        'Product_Age_Months': 4,
        'Warranty_Duration_Months': 12,
        'Remaining_Warranty_Months': 8,
        'Fault_Type': 'Power Failure',
        'Repair_History': 'None',
        'Has_Receipt': True,
        'Has_Warranty_Card': True,
        'Has_Damage_Photo': True,
        'Serial_Number_Match': True,
        'Previous_Unauthorized_Repairs': False,
        'Duplicate_Claim_Flag': False,
        'Date_Contradiction_Flag': False,
        'Missing_Documents_Count': 0,
        'Claim_Class': 'Valid Claim'
    }

    result = pipeline.process_claim(test_claim)

    assert 'final_verdict' in result, "Missing final_verdict"
    assert result['final_verdict'] in ['Likely Valid', 'Likely Invalid', 'Manual Review Required'], f"Invalid verdict: {result['final_verdict']}"

    assert 'python_model' in result, "Missing python_model details"
    assert len(result['python_model']['class_probabilities']) == 3

    assert 'teachable_machine' in result, "Missing teachable_machine details"
    assert len(result['teachable_machine']['class_probabilities']) == 3

    assert 'model_comparison' in result, "Missing model_comparison details"
    assert 'decision_explanation' in result, "Missing decision_explanation"

    exp = result['decision_explanation']
    assert 'supporting_factors' in exp, "Missing supporting_factors"
    assert 'opposing_factors' in exp, "Missing opposing_factors"
    assert 'rules_passed' in exp, "Missing rules_passed"
    assert 'rules_failed' in exp, "Missing rules_failed"

    # Test Hard Rejection Path
    invalid_claim = test_claim.copy()
    invalid_claim['Remaining_Warranty_Months'] = 0.0
    res_inv = pipeline.process_claim(invalid_claim)
    assert res_inv['final_verdict'] == 'Likely Invalid', f"Expected Likely Invalid, got {res_inv['final_verdict']}"

    print("Module 6 (Unified Final Decision Logic & Explanation Engine) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_6_unified_pipeline()
