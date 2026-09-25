import sys
import os

# Append project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policies.rules import RuleEngine

def run_tests():
    engine = RuleEngine()
    
    # Test Case 1: Expired Warranty
    sample_claim = {
        'Claim_ID': 'CLM-0001',
        'Remaining_Warranty_Months': 0.0,
        'Serial_Number_Match': True,
        'Duplicate_Claim_Flag': False,
        'Date_Contradiction_Flag': False
    }
    
    result = engine.evaluate_claim(sample_claim, ml_pred_class='Valid Claim', ml_confidence=0.92)
    print("Test 1 (Expired Warranty Override):", result['final_status'] == 'Invalid Claim')

    # Test Case 2: Serial Mismatch
    sample_claim_2 = {
        'Claim_ID': 'CLM-0002',
        'Remaining_Warranty_Months': 6.0,
        'Serial_Number_Match': False,
        'Duplicate_Claim_Flag': False,
        'Date_Contradiction_Flag': False
    }
    result_2 = engine.evaluate_claim(sample_claim_2, ml_pred_class='Valid Claim', ml_confidence=0.88)
    print("Test 2 (Serial Mismatch Override):", result_2['final_status'] == 'Manual Review')

if __name__ == '__main__':
    run_tests()