import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ocr_engine import ReceiptOCRProcessor

def test_module_7_ocr_engine():
    processor = ReceiptOCRProcessor()

    sample_raw_text = "OFFICIAL INVOICE Brand: Samsung Serial: SA-991823 Date: 2023-05-12 Total Amount: $850.00 Store: Official Retail Store"
    parsed = processor.parse_receipt_data(sample_raw_text)

    assert parsed['extracted_serial'] == 'SA-991823', f"Expected SA-991823, got {parsed['extracted_serial']}"
    assert parsed['extracted_date'] == '2023-05-12', f"Expected 2023-05-12, got {parsed['extracted_date']}"
    assert parsed['extracted_amount'] == '850.00', f"Expected 850.00, got {parsed['extracted_amount']}"
    assert parsed['extracted_retailer'] == 'Official Retail Store', f"Expected Official Retail Store, got {parsed['extracted_retailer']}"
    assert parsed['confidence'] > 0.0

    print("Module 7 (OCR Intake & Verification Parsing) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_7_ocr_engine()
