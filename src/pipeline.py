import os
import sys
import tempfile
import joblib
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import ClaimPreprocessor
from policies.rules import RuleEngine
from src.ocr_engine import ReceiptOCRProcessor
from src.teachable_machine_engine import TeachableMachineClassifier
from src.model_comparison import ModelComparisonEngine
from dataset_generator.generate_summary_cards import create_card_image

class ClaimVerificationPipeline:
    def __init__(self):
        self.model = joblib.load('model/best_tabular_model.joblib')
        self.preprocessor = joblib.load('model/preprocessor.joblib')
        self.rule_engine = RuleEngine()
        self.ocr_processor = ReceiptOCRProcessor()
        self.tm_classifier = TeachableMachineClassifier()
        self.comparison_engine = ModelComparisonEngine()

    def process_claim(self, claim_data: dict, receipt_image_path: str = None, summary_card_image_path: str = None) -> dict:
        ocr_details = {}

        # 1. Run OCR if receipt image is supplied
        if receipt_image_path and os.path.exists(receipt_image_path):
            raw_text = self.ocr_processor.extract_text_from_image(receipt_image_path)
            ocr_details = self.ocr_processor.parse_receipt_data(raw_text)

            # Auto-flag serial mismatch if OCR extracted serial differs from form input
            if ocr_details.get('extracted_serial'):
                input_sn = str(claim_data.get('Serial_Number', '')).replace(" ", "").upper()
                ocr_sn = str(ocr_details['extracted_serial']).replace(" ", "").upper()
                if input_sn and ocr_sn and input_sn != ocr_sn:
                    claim_data['Serial_Number_Match'] = False

        # 2. Python Tabular Model Inference
        df_single = pd.DataFrame([claim_data])
        X_vec = self.preprocessor.transform(df_single)

        py_pred_idx = self.model.predict(X_vec)[0]
        py_probs = self.model.predict_proba(X_vec)[0]
        py_class = self.preprocessor.label_encoder.classes_[py_pred_idx]
        py_confidence = float(np.max(py_probs))
        
        py_class_probs = {
            cls_name: float(prob) 
            for cls_name, prob in zip(self.preprocessor.label_encoder.classes_, py_probs)
        }

        # 3. Teachable Machine Card Inference
        temp_card_created = False
        if not summary_card_image_path or not os.path.exists(summary_card_image_path):
            # Generate summary card on the fly to avoid shortcutting
            temp_card_path = os.path.join(tempfile.gettempdir(), f"temp_card_{claim_data.get('Claim_ID', 'N/A')}.png")
            card_row = claim_data.copy()
            if 'Claim_ID' not in card_row:
                card_row['Claim_ID'] = 'CLM-TEMP'
            create_card_image(card_row, temp_card_path, variation=1)
            summary_card_image_path = temp_card_path
            temp_card_created = True

        tm_res = self.tm_classifier.predict(summary_card_image_path)
        tm_class = tm_res['predicted_class']
        tm_confidence = tm_res['confidence']
        tm_class_probs = tm_res['class_probabilities']

        # Clean up temporary card artifact
        if temp_card_created and os.path.exists(summary_card_image_path):
            try:
                os.remove(summary_card_image_path)
            except Exception:
                pass

        # 4. Model Comparison Evaluation
        comparison_res = self.comparison_engine.compare_models(
            py_class, py_confidence, tm_class, tm_confidence
        )

        # 5. Business Policy Rules Evaluation
        rule_res = self.rule_engine.evaluate_claim(claim_data, py_class, py_confidence)

        # 6. Final Verdict Synthesis (Likely Valid / Likely Invalid / Manual Review Required)
        rule_status = rule_res['final_status']
        comp_status = comparison_res['comparison_status']

        if rule_status == "Invalid Claim":
            final_verdict = "Likely Invalid"
        elif rule_status == "Manual Review Required" or comp_status in ["Model Disagreement", "Uncertain Result"] or py_class != tm_class:
            final_verdict = "Manual Review Required"
        elif py_class == "Valid Claim" and rule_status == "Valid Claim" and comp_status in ["Strong Match", "Acceptable Match"]:
            final_verdict = "Likely Valid"
        else:
            final_verdict = "Manual Review Required"

        # 7. Construct Supporting & Opposing Factors Explanation
        supporting_factors = []
        opposing_factors = []

        if py_class == ( "Valid Claim" if final_verdict == "Likely Valid" else "Invalid Claim" ):
            supporting_factors.append(f"Python Tabular Classifier predicted '{py_class}' with {py_confidence*100:.1f}% confidence.")
        else:
            opposing_factors.append(f"Python Tabular Classifier predicted '{py_class}' ({py_confidence*100:.1f}% confidence).")

        if tm_class == ( "Valid Claim" if final_verdict == "Likely Valid" else "Invalid Claim" ):
            supporting_factors.append(f"Teachable Machine Image Classifier predicted '{tm_class}' with {tm_confidence*100:.1f}% confidence.")
        else:
            opposing_factors.append(f"Teachable Machine Image Classifier predicted '{tm_class}' ({tm_confidence*100:.1f}% confidence).")

        if comp_status in ["Strong Match", "Acceptable Match"]:
            supporting_factors.append(f"Model Consensus: {comp_status} (|delta conf| = {comparison_res['confidence_delta']*100:.1f}%).")
        else:
            opposing_factors.append(f"Model Consensus Risk: {comp_status} (|delta conf| = {comparison_res['confidence_delta']*100:.1f}%).")


        for rule in rule_res['rules_passed']:
            supporting_factors.append(f"Passed Policy Rule: {rule}")

        for flag in rule_res['rule_flags']:
            opposing_factors.append(f"Policy Risk Flag: {flag}")

        decision_explanation = {
            'summary': f"Final verdict '{final_verdict}' derived from multi-modal AI models (Python + Teachable Machine) and policy rule engine.",
            'supporting_factors': supporting_factors,
            'opposing_factors': opposing_factors,
            'rules_passed': rule_res['rules_passed'],
            'rules_failed': rule_res['rules_failed'],
            'required_additional_evidence': rule_res['required_evidence']
        }

        return {
            'claim_id': claim_data.get('Claim_ID', 'N/A'),
            'final_verdict': final_verdict,
            'action_required': rule_res['action_required'],
            'python_model': {
                'prediction': py_class,
                'confidence': py_confidence,
                'class_probabilities': py_class_probs
            },
            'teachable_machine': {
                'prediction': tm_class,
                'confidence': tm_confidence,
                'class_probabilities': tm_class_probs
            },
            'model_comparison': comparison_res,
            'rule_engine': rule_res,
            'decision_explanation': decision_explanation,
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
        'Purchase_Date': '2023-01-15',
        'Claim_Date': '2023-06-15',
        'Product_Age_Months': 5,
        'Warranty_Duration_Months': 12,
        'Remaining_Warranty_Months': 7,
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

    result = pipeline.process_claim(test_data)
    print("Full Multi-Modal Verification Pipeline Output:\n")
    print(f"Claim ID: {result['claim_id']}")
    print(f"Final Verdict: {result['final_verdict']}")
    print(f"Action Required: {result['action_required']}")
    print(f"Python Model: {result['python_model']['prediction']} ({result['python_model']['confidence']*100:.1f}%)")
    print(f"Teachable Machine: {result['teachable_machine']['prediction']} ({result['teachable_machine']['confidence']*100:.1f}%)")
    print(f"Comparison: {result['model_comparison']['comparison_status']}")
    print("Supporting Factors:", result['decision_explanation']['supporting_factors'])
    print("Opposing Factors:", result['decision_explanation']['opposing_factors'])