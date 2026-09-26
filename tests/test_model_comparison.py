import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_comparison import ModelComparisonEngine

def test_module_4_model_comparison():
    engine = ModelComparisonEngine(strong_threshold=0.10, acceptable_threshold=0.20, min_certainty_threshold=0.60)

    # 1. Test Strong Match
    res_strong = engine.compare_models("Valid Claim", 0.95, "Valid Claim", 0.92)
    assert res_strong['comparison_status'] == "Strong Match"
    assert res_strong['is_agreement'] is True
    assert res_strong['confidence_delta'] == 0.03

    # 2. Test Acceptable Match
    res_acceptable = engine.compare_models("Valid Claim", 0.95, "Valid Claim", 0.80)
    assert res_acceptable['comparison_status'] == "Acceptable Match"

    # 3. Test Weak Match
    res_weak = engine.compare_models("Valid Claim", 0.95, "Valid Claim", 0.70)
    assert res_weak['comparison_status'] == "Weak Match"

    # 4. Test Model Disagreement
    res_disagree = engine.compare_models("Valid Claim", 0.90, "Invalid Claim", 0.85)
    assert res_disagree['comparison_status'] == "Model Disagreement"
    assert res_disagree['is_agreement'] is False

    # 5. Test Uncertain Result
    res_uncertain = engine.compare_models("Valid Claim", 0.55, "Valid Claim", 0.52)
    assert res_uncertain['comparison_status'] == "Uncertain Result"

    print("Module 4 (Model Comparison & Verification) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_4_model_comparison()
