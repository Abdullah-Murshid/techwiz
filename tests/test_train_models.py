import os
import json
import joblib
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import ClaimPreprocessor

def test_module_2_tabular_models():
    assert os.path.exists('model/best_tabular_model.joblib'), "best_tabular_model.joblib missing"
    assert os.path.exists('model/preprocessor.joblib'), "preprocessor.joblib missing"
    assert os.path.exists('model/model_metadata.json'), "model_metadata.json missing"
    assert os.path.exists('reports/tabular_model_report.txt'), "tabular_model_report.txt missing"

    # Load artifacts
    model = joblib.load('model/best_tabular_model.joblib')
    preprocessor = joblib.load('model/preprocessor.joblib')
    
    with open('model/model_metadata.json', 'r') as f:
        meta = json.load(f)

    assert meta['model_version'] == '1.0.0', "Metadata version mismatch"
    assert len(meta['classes']) == 3, f"Expected 3 classes, got {len(meta['classes'])}"

    # Test single record prediction & 3-class probability distribution
    sample_df = pd.read_csv('data/test/test_claims.csv').head(1)
    X_vec = preprocessor.transform(sample_df)
    
    probs = model.predict_proba(X_vec)[0]
    preds = model.predict(X_vec)[0]

    assert len(probs) == 3, f"Expected 3-class probability output, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 1e-4, "Probabilities must sum to 1.0"
    
    pred_class_name = preprocessor.label_encoder.classes_[preds]
    assert pred_class_name in ['Valid Claim', 'Invalid Claim', 'Manual Review'], f"Invalid class predicted: {pred_class_name}"

    print("Module 2 (Tabular Classification & Preprocessing) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_2_tabular_models()
