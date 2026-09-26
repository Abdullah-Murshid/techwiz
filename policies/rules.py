import os
import json
import hashlib
from datetime import datetime

class RuleEngine:
    def __init__(self, policy_path=os.path.join('policies', 'warranty_policies.json')):
        self.policy_path = policy_path
        self.policies = self._load_policies()
        self.seen_receipt_hashes = set()
        self.seen_serials = set()

    def _load_policies(self):
        if os.path.exists(self.policy_path):
            try:
                with open(self.policy_path, 'r') as f:
                    data = json.load(f)
                    return data.get('categories', {})
            except Exception as e:
                print(f"[Warning] Failed to load policy JSON: {e}")
        
        # Fallback inline dictionary if file unreadable
        return {
            "Default": {
                "standard_warranty_months": 12,
                "covered_faults": ["Motor Failure", "Screen Damage", "Power Failure", "Water Leakage", "Overheating", "Board Defect"],
                "excluded_faults": ["User Misuse"],
                "required_documents": ["Has_Receipt"],
                "allow_unauthorized_repairs": False,
                "require_serial_match": True
            }
        }

    def compute_document_hash(self, doc_bytes_or_str) -> str:
        """Computes MD5 hash for document deduplication."""
        if isinstance(doc_bytes_or_str, str):
            doc_bytes = doc_bytes_or_str.encode('utf-8')
        else:
            doc_bytes = doc_bytes_or_str
        return hashlib.md5(doc_bytes).hexdigest()

    def register_processed_claim(self, claim_data: dict):
        """Registers claim serial and document hash to track duplicates across session."""
        serial = claim_data.get('Serial_Number')
        if serial:
            self.seen_serials.add(serial)
            
        r_hash = claim_data.get('Receipt_Hash')
        if r_hash:
            self.seen_receipt_hashes.add(r_hash)

    def evaluate_claim(self, claim_data: dict, ml_pred_class: str = "Valid Claim", ml_confidence: float = 1.0) -> dict:
        """
        Evaluates externalized business policy rules, date contradictions, 
        document hashes, and policy exclusions.
        """
        flags = []
        rules_passed = []
        rules_failed = []
        required_evidence = []
        
        category = claim_data.get('Product_Category', 'Default')
        cat_policy = self.policies.get(category, self.policies.get('Default', {}))
        
        status = ml_pred_class
        action_required = "Proceed with automated routing."

        # 1. Contradiction Detection (Date Logic Conflict)
        date_contradiction = claim_data.get('Date_Contradiction_Flag', False)
        purchase_date_str = claim_data.get('Purchase_Date')
        claim_date_str = claim_data.get('Claim_Date')
        
        if not date_contradiction and purchase_date_str and claim_date_str:
            try:
                p_dt = datetime.strptime(str(purchase_date_str), '%Y-%m-%d')
                c_dt = datetime.strptime(str(claim_date_str), '%Y-%m-%d')
                if c_dt < p_dt:
                    date_contradiction = True
            except Exception:
                pass

        if date_contradiction:
            msg = "CRITICAL: Claim date precedes purchase date (Chronology Contradiction)."
            flags.append(msg)
            rules_failed.append("Chronological Date Logic")
            status = "Invalid Claim"
            action_required = "Automatic Rejection due to invalid transaction timeline."
        else:
            rules_passed.append("Chronological Date Logic")

        # 2. Warranty Expiry Check
        rem_warranty = float(claim_data.get('Remaining_Warranty_Months', 0))
        prod_age = float(claim_data.get('Product_Age_Months', 0))
        max_warranty = cat_policy.get('standard_warranty_months', 24)
        
        if rem_warranty <= 0 or prod_age > max_warranty:
            msg = f"POLICY: Product warranty expired (Product Age: {prod_age} mos > Max Warranty: {max_warranty} mos)."
            flags.append(msg)
            rules_failed.append("Warranty Duration Coverage")
            if status != "Invalid Claim":
                status = "Invalid Claim"
                action_required = "Automatic Rejection due to expired warranty period."
        else:
            rules_passed.append("Warranty Duration Coverage")

        # 3. Fault Coverage & Excluded Damage Check
        fault = claim_data.get('Fault_Type', 'General Defect')
        covered_faults = cat_policy.get('covered_faults', [])
        excluded_faults = cat_policy.get('excluded_faults', [])
        
        if fault in excluded_faults:
            msg = f"COVERAGE: Fault '{fault}' is explicitly excluded under {category} policy."
            flags.append(msg)
            rules_failed.append("Fault Exclusions Check")
            status = "Invalid Claim"
            action_required = f"Automatic Rejection due to non-covered damage category ({fault})."
        elif covered_faults and fault not in covered_faults:
            msg = f"COVERAGE: Fault '{fault}' is not listed under covered issues for {category}."
            flags.append(msg)
            rules_failed.append("Fault Exclusions Check")
            status = "Manual Review"
            action_required = "Route to human reviewer for non-standard fault coverage assessment."
        else:
            rules_passed.append("Fault Exclusions Check")

        # 4. Mandatory Proof of Purchase & Document Completeness
        has_receipt = claim_data.get('Has_Receipt', True)
        if not has_receipt:
            msg = "PROOF OF PURCHASE: Missing proof of purchase (Receipt)."
            flags.append(msg)
            rules_failed.append("Proof of Purchase Requirement")
            status = "Invalid Claim"
            action_required = "Automatic Rejection due to missing mandatory receipt."
            required_evidence.append("Official Purchase Receipt / Retailer Invoice")
        else:
            rules_passed.append("Proof of Purchase Requirement")

        req_docs = cat_policy.get('required_documents', ['Has_Receipt'])
        for doc_key in req_docs:
            if doc_key != 'Has_Receipt' and not claim_data.get(doc_key, True):
                doc_name = doc_key.replace("Has_", "").replace("_", " ")
                msg = f"MISSING DOCUMENT: Required document '{doc_name}' is missing."
                flags.append(msg)
                rules_failed.append(f"Required Document ({doc_name})")
                required_evidence.append(doc_name)
                if status != "Invalid Claim":
                    status = "Manual Review"
                    action_required = f"Request missing supporting evidence ({doc_name})."

        # 5. Duplicate Claim / Document Hash Detection
        receipt_hash = claim_data.get('Receipt_Hash')
        dup_flag = claim_data.get('Duplicate_Claim_Flag', False)
        
        if (receipt_hash and receipt_hash in self.seen_receipt_hashes) or dup_flag or receipt_hash == "DUP_HASH_REF_001":
            msg = "SECURITY: Duplicate document hash / claim serial detected (Potential Fraud Risk)."
            flags.append(msg)
            rules_failed.append("Duplicate Document & Claim Hash Check")
            if status != "Invalid Claim":
                status = "Manual Review"
                action_required = "Hold claim for fraud team audit due to duplicate record flag."
        else:
            rules_passed.append("Duplicate Document & Claim Hash Check")

        # 6. Serial Number Matching
        serial_match = claim_data.get('Serial_Number_Match', True)
        if cat_policy.get('require_serial_match', True) and not serial_match:
            msg = "VERIFICATION: Serial number mismatch between receipt and physical device."
            flags.append(msg)
            rules_failed.append("Serial Number Verification")
            if status != "Invalid Claim":
                status = "Manual Review"
                action_required = "Route to manual review for serial number verification."
                required_evidence.append("Clear Photo of Device Serial Number Plate")
        else:
            rules_passed.append("Serial Number Verification")

        # 7. Previous Unauthorized Repairs
        unauth_repair = claim_data.get('Previous_Unauthorized_Repairs', False) or claim_data.get('Repair_History') == 'Unauthorized Repair'
        if unauth_repair and not cat_policy.get('allow_unauthorized_repairs', False):
            msg = "POLICY: Uncertified prior repair detected; voids standard manufacturer coverage."
            flags.append(msg)
            rules_failed.append("Unauthorized Repair Compliance")
            status = "Invalid Claim"
            action_required = "Automatic Rejection due to uncertified prior repair."
        else:
            rules_passed.append("Unauthorized Repair Compliance")

        # 8. ML Confidence Check
        if ml_confidence < 0.75 and status == "Valid Claim":
            msg = f"ML CONFIDENCE: Model prediction score below threshold ({ml_confidence*100:.1f}%)."
            flags.append(msg)
            rules_failed.append("ML Confidence Threshold")
            status = "Manual Review"
            action_required = "Secondary human review required due to model uncertainty."
        else:
            rules_passed.append("ML Confidence Threshold")

        # Final Status Normalization: Map internal 'Manual Review' -> 'Manual Review Required'
        final_verdict = status
        if final_verdict == "Manual Review":
            final_verdict = "Manual Review Required"
            
        return {
            'final_status': final_verdict,
            'original_ml_class': ml_pred_class,
            'ml_confidence': float(ml_confidence),
            'rule_flags': flags,
            'rules_passed': rules_passed,
            'rules_failed': rules_failed,
            'required_evidence': required_evidence,
            'action_required': action_required
        }


if __name__ == '__main__':
    engine = RuleEngine()
    
    sample_claim = {
        'Claim_ID': 'CLM-TEST-99',
        'Product_Category': 'Washing Machine',
        'Fault_Type': 'Motor Failure',
        'Remaining_Warranty_Months': 12.0,
        'Product_Age_Months': 12.0,
        'Has_Receipt': True,
        'Has_Warranty_Card': True,
        'Has_Damage_Photo': True,
        'Serial_Number_Match': True,
        'Previous_Unauthorized_Repairs': False,
        'Duplicate_Claim_Flag': False,
        'Date_Contradiction_Flag': False
    }
    
    res = engine.evaluate_claim(sample_claim, "Valid Claim", 0.95)
    print("Rule Engine Evaluation Test Result:\n", res)