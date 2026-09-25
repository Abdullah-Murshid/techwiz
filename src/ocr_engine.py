import re
from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class ReceiptOCRProcessor:
    def __init__(self, languages=['en']):
        if EASYOCR_AVAILABLE:
            # Initialize EasyOCR reader (runs on CPU by default)
            self.reader = easyocr.Reader(languages, gpu=False)
        else:
            self.reader = None

    def extract_text_from_image(self, image_input):
        """
        Extracts raw text blocks from image path or PIL Image object.
        """
        if not EASYOCR_AVAILABLE or self.reader is None:
            return "EasyOCR is not installed. Fallback mode active."

        if isinstance(image_input, str):
            results = self.reader.readtext(image_input)
        else:
            # Handle PIL image or file buffer
            import numpy as np
            img_np = np.array(Image.open(image_input))
            results = self.reader.readtext(img_np)

        extracted_text = [text for (_, text, prob) in results if prob > 0.3]
        return " ".join(extracted_text)

    def parse_receipt_data(self, raw_text: str) -> dict:
        """
        Extracts key entities using Regex pattern matching.
        """
        parsed = {
            'extracted_serial': None,
            'extracted_date': None,
            'extracted_amount': None,
            'raw_text': raw_text
        }

        # 1. Regex Pattern for Serial Numbers (e.g., SN-123456, SA-991823, LG-543210)
        serial_match = re.search(r'\b[A-Z]{2,4}[-\s]?\d{5,8}\b', raw_text)
        if serial_match:
            parsed['extracted_serial'] = serial_match.group(0).replace(" ", "")

        # 2. Regex Pattern for Dates (YYYY-MM-DD or DD/MM/YYYY)
        date_match = re.search(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b', raw_text)
        if date_match:
            parsed['extracted_date'] = date_match.group(0)

        # 3. Regex Pattern for Amounts ($400, $1200.00, etc.)
        amount_match = re.search(r'\$\s?\d+(?:\.\d{2})?', raw_text)
        if amount_match:
            parsed['extracted_amount'] = amount_match.group(0)

        return parsed


if __name__ == '__main__':
    processor = ReceiptOCRProcessor()
    sample_text = "OFFICIAL RECEIPT Brand: Samsung Serial: SA-991823 Date: 2023-05-12 Total Amount: $850.00"
    data = processor.parse_receipt_data(sample_text)
    print("OCR Parse Test Result:", data)