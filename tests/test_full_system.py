import sys
import os

# Append project root directory to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import pytest  <-- REMOVE OR COMMENT OUT THIS LINE

from src.pipeline import ClaimVerificationPipeline

def test_pipeline_execution():
    pipeline = ClaimVerificationPipeline()
    
    # 1. Test Valid Claim
    valid_claim = {
        'Claim_ID': 'TEST-VALID-01',
        'Customer_Name': 'John Doe',
        'Product_Category': 'Laptop',
        'Brand': 'Dell',
        'Model_Number': 'DEL-101',
        'Serial_Number': 'DE-123456',
        'Purchase_Price': 1000,
        'Product_Age_Months': 5,
        'Warranty_Duration_Months': 12,
        'Remaining_Warranty_Months': 7,
        'Fault_Type': 'Power Failure',
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
    res_valid = pipeline.process_claim(valid_claim)
    assert res_valid['final_verdict'] == 'Valid Claim'

    # 2. Test Expired Warranty Hard Rejection
    invalid_claim = valid_claim.copy()
    invalid_claim['Claim_ID'] = 'TEST-INVALID-01'
    invalid_claim['Remaining_Warranty_Months'] = 0.0
    res_invalid = pipeline.process_claim(invalid_claim)
    assert res_invalid['final_verdict'] == 'Invalid Claim'

    # 3. Test Serial Mismatch Manual Review
    mismatch_claim = valid_claim.copy()
    mismatch_claim['Claim_ID'] = 'TEST-REVIEW-01'
    mismatch_claim['Serial_Number_Match'] = False
    res_mismatch = pipeline.process_claim(mismatch_claim)
    assert res_mismatch['final_verdict'] == 'Manual Review'

if __name__ == '__main__':
    test_pipeline_execution()
    print("All System Pipeline Unit Tests Passed Successfully!")