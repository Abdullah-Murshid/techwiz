import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies.rules import RuleEngine

def test_module_5_rule_engine():
    engine = RuleEngine()
    assert os.path.exists('policies/warranty_policies.json'), "warranty_policies.json missing"
    assert "Washing Machine" in engine.policies, "Washing Machine policy missing"

    # 1. Test Valid Claim Pass
    valid_claim = {
        'Claim_ID': 'CLM-V01',
        'Product_Category': 'Washing Machine',
        'Fault_Type': 'Motor Failure',
        'Remaining_Warranty_Months': 10.0,
        'Product_Age_Months': 10.0,
        'Has_Receipt': True,
        'Has_Warranty_Card': True,
        'Has_Damage_Photo': True,
        'Serial_Number_Match': True,
        'Previous_Unauthorized_Repairs': False,
        'Duplicate_Claim_Flag': False,
        'Date_Contradiction_Flag': False
    }
    res_valid = engine.evaluate_claim(valid_claim, "Valid Claim", 0.95)
    assert res_valid['final_status'] == "Valid Claim"
    assert len(res_valid['rules_failed']) == 0

    # 2. Test Expired Warranty Hard Rejection
    expired_claim = valid_claim.copy()
    expired_claim['Remaining_Warranty_Months'] = 0.0
    res_exp = engine.evaluate_claim(expired_claim, "Valid Claim", 0.95)
    assert res_exp['final_status'] == "Invalid Claim"
    assert "Warranty Duration Coverage" in res_exp['rules_failed']

    # 3. Test Date Contradiction Hard Rejection
    date_claim = valid_claim.copy()
    date_claim['Purchase_Date'] = "2024-05-10"
    date_claim['Claim_Date'] = "2024-01-10"
    res_date = engine.evaluate_claim(date_claim, "Valid Claim", 0.95)
    assert res_date['final_status'] == "Invalid Claim"
    assert "Chronological Date Logic" in res_date['rules_failed']

    # 4. Test Excluded Fault Rejection
    fault_claim = valid_claim.copy()
    fault_claim['Fault_Type'] = "Cosmetic Scratches"
    res_fault = engine.evaluate_claim(fault_claim, "Valid Claim", 0.95)
    assert res_fault['final_status'] == "Invalid Claim"
    assert "Fault Exclusions Check" in res_fault['rules_failed']

    # 5. Test Duplicate Hash Flag
    dup_claim = valid_claim.copy()
    dup_claim['Receipt_Hash'] = "DUP_HASH_REF_001"
    res_dup = engine.evaluate_claim(dup_claim, "Valid Claim", 0.95)
    assert res_dup['final_status'] == "Manual Review Required"
    assert "Duplicate Document & Claim Hash Check" in res_dup['rules_failed']

    # 6. Test Document Hashing Helper
    doc_hash = engine.compute_document_hash("Sample Receipt Content 12345")
    assert len(doc_hash) == 32, f"Expected 32-char MD5 hash, got len {len(doc_hash)}"

    print("Module 5 (Externalized Policy Rule Engine) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_5_rule_engine()
