import pandas as pd

class RuleEngine:
    def __init__(self):
        pass

    def evaluate_claim(self, claim_data: dict, ml_pred_class: str, ml_confidence: float) -> dict:
        """
        Applies explicit business rules to validate, reject, or flag claims.
        Override/adjust predictions based on policy boundary conditions.
        """
        flags = []
        status = ml_pred_class
        action_required = "Proceed with normal automated routing."
        
        # Rule 1: Date Contradiction (Hard Rejection)
        if claim_data.get('Date_Contradiction_Flag', False):
            flags.append("CRITICAL: Claim date precedes purchase date.")
            status = "Invalid Claim"
            action_required = "Automatic Rejection due to invalid transaction chronology."
            
        # Rule 2: Expired Warranty (Hard Rejection)
        elif claim_data.get('Remaining_Warranty_Months', 0) <= 0:
            flags.append("POLICY: Product warranty period has expired.")
            status = "Invalid Claim"
            action_required = "Automatic Rejection due to expired warranty."

        # Rule 3: Serial Number Mismatch (Manual Review Mandatory)
        elif not claim_data.get('Serial_Number_Match', True):
            flags.append("WARNING: Serial number mismatch between receipt and claim.")
            status = "Manual Review"
            action_required = "Route to human claims auditor for serial verification."

        # Rule 4: Duplicate Claim Submission
        elif claim_data.get('Duplicate_Claim_Flag', False):
            flags.append("WARNING: Duplicate claim detected for serial number.")
            status = "Manual Review"
            action_required = "Hold claim and verify prior payout records."

        # Rule 5: Low Confidence ML Score Routing
        elif ml_confidence < 0.75:
            flags.append(f"ML CONFIDENCE: Model confidence below threshold ({ml_confidence*100:.1f}%).")
            status = "Manual Review"
            action_required = "Secondary human review required due to model uncertainty."

        return {
            'final_status': status,
            'original_ml_class': ml_pred_class,
            'confidence': ml_confidence,
            'rule_flags': flags,
            'action_required': action_required
        }