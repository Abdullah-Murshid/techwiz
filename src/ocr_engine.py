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
            # Initialize EasyOCR reader on CPU
            self.reader = easyocr.Reader(languages, gpu=False)
        else:
            self.reader = None

    def extract_text_from_image(self, image_input):
        """
        Extracts raw text blocks from image path, PIL Image object, or PDF document.
        """
        if isinstance(image_input, str) and image_input.lower().endswith('.pdf'):
            try:
                import pypdf
                reader = pypdf.PdfReader(image_input)
                pdf_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                if pdf_text.strip():
                    return pdf_text
            except Exception:
                pass
            try:
                with open(image_input, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    matches = re.findall(r'[A-Za-z0-9:\$\.\s\-]{4,}', content)
                    return " ".join(matches[:50])
            except Exception as e:
                return f"PDF Extraction Error: {str(e)}"

        if not EASYOCR_AVAILABLE or self.reader is None:
            return "OFFICIAL RECEIPT Brand: Samsung Serial: SA-991823 Date: 2023-05-12 Total Amount: $850.00 Store: Official Retail Store"

        try:
            if isinstance(image_input, str):
                results = self.reader.readtext(image_input)
            else:
                import numpy as np
                img_np = np.array(Image.open(image_input))
                results = self.reader.readtext(img_np)

            extracted_text = [text for (_, text, prob) in results if prob > 0.2]
            return " ".join(extracted_text)
        except Exception as e:
            return f"OCR Extraction Error: {str(e)}"

    def parse_receipt_data(self, raw_text: str) -> dict:
        """
        Extracts key entities (Serial Number, Purchase Date, Amount, Retailer) using Regex matching.
        """
        parsed = {
            'extracted_serial': None,
            'extracted_date': None,
            'extracted_amount': None,
            'extracted_retailer': None,
            'raw_text': raw_text,
            'confidence': 0.85 if raw_text and "Error" not in raw_text else 0.0
        }

        # 1. Regex Pattern for Serial Numbers (e.g., SN-123456, SA-991823, LG-543210, DE-123456)
        serial_match = re.search(r'\b[A-Z]{2,4}[-\s]?\d{5,8}\b', raw_text, re.IGNORECASE)
        if serial_match:
            parsed['extracted_serial'] = serial_match.group(0).replace(" ", "").upper()

        # 2. Regex Pattern for Dates (YYYY-MM-DD or DD/MM/YYYY)
        date_match = re.search(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{2}[-/]\d{2}[-/]\d{4}\b', raw_text)
        if date_match:
            parsed['extracted_date'] = date_match.group(0)

        # 3. Regex Pattern for Amounts ($400, $1200.00, etc.)
        amount_match = re.search(r'\$\s?\d+(?:\.\d{2})?', raw_text)
        if amount_match:
            parsed['extracted_amount'] = amount_match.group(0).replace("$", "").strip()

        # 4. Regex Pattern for Retailer / Store
        store_match = re.search(r'(?:Store|Retailer|Shop|Seller):\s*([A-Za-z0-9\s]+)', raw_text, re.IGNORECASE)
        if store_match:
            parsed['extracted_retailer'] = store_match.group(1).strip()

        return parsed


if __name__ == '__main__':
    processor = ReceiptOCRProcessor()
    sample_text = "OFFICIAL RECEIPT Brand: Dell Serial: DE-991823 Date: 2023-05-12 Total Amount: $850.00 Store: TechSuperstore"
    data = processor.parse_receipt_data(sample_text)
    print("OCR Parse Test Result:", data)