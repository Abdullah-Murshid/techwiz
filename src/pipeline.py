import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import numpy as np
from src.preprocessing import ClaimPreprocessor
from policies.rules import RuleEngine
from src.ocr_engine import ReceiptOCRProcessor


class ClaimVerificationPipeline:
    def __init__(self):
        self.model = joblib.load('model/best_tabular_model.joblib')
        self.preprocessor = joblib.load('model/preprocessor.joblib')
        self.rule_engine = RuleEngine()
        self.ocr_processor = ReceiptOCRProcessor()

    def process_claim(self, claim_data: dict, receipt_image_path: str = None) -> dict:
        ocr_details = {}

        # 1. Run OCR if image is supplied
        if receipt_image_path and os.path.exists(receipt_image_path):
            raw_text = self.ocr_processor.extract_text_from_image(receipt_image_path)
            ocr_details = self.ocr_processor.parse_receipt_data(raw_text)

            # Auto-flag serial mismatch if OCR extracted serial differs from form input
            if ocr_details.get('extracted_serial'):
                input_sn = claim_data.get('Serial_Number', '').replace(" ", "").upper()
                ocr_sn = ocr_details['extracted_serial'].replace(" ", "").upper()
                if input_sn and ocr_sn and input_sn != ocr_sn:
                    claim_data['Serial_Number_Match'] = False

        # 2. Preprocess Form Data
        df_single = pd.DataFrame([claim_data])
        X_vec = self.preprocessor.transform(df_single)

        # 3. Model Inference
        ml_pred_idx = self.model.predict(X_vec)[0]
        ml_probs = self.model.predict_proba(X_vec)[0]
        ml_class = self.preprocessor.label_encoder.classes_[ml_pred_idx]
        ml_confidence = float(np.max(ml_probs))

        # 4. Evaluate Business Rules
        rule_results = self.rule_engine.evaluate_claim(claim_data, ml_class, ml_confidence)

        return {
            'claim_id': claim_data.get('Claim_ID', 'N/A'),
            'ml_prediction': ml_class,
            'ml_confidence': ml_confidence,
            'final_verdict': rule_results['final_status'],
            'action_required': rule_results['action_required'],
            'flags': rule_results['rule_flags'],
            'ocr_extracted_data': ocr_details
        }


if __name__ == '__main__':
    pipeline = ClaimVerificationPipeline()

    test_data = {
        'Claim_ID': 'CLM-TEST-01',
        'Customer_Name': 'Alice Smith',
        'Product_Category': 'Laptop',
        'Brand': 'Dell',
        'Model_Number': 'DEL-101',
        'Serial_Number': 'DE-123456',
        'Purchase_Price': 1200,
        'Product_Age_Months': 6,
        'Warranty_Duration_Months': 12,
        'Remaining_Warranty_Months': 6,
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

    result = pipeline.process_claim(test_data)
    print("Full Pipeline Execution Result:\n", result)