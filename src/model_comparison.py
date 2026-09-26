class ModelComparisonEngine:
    def __init__(self, strong_threshold=0.10, acceptable_threshold=0.20, min_certainty_threshold=0.60):
        self.strong_threshold = strong_threshold
        self.acceptable_threshold = acceptable_threshold
        self.min_certainty_threshold = min_certainty_threshold

    def compare_models(self, python_pred: str, python_conf: float, tm_pred: str, tm_conf: float) -> dict:
        """
        Compares Python Tabular Classifier vs Teachable Machine Image Classifier output.
        Computes |python_confidence - tm_confidence| and classifies match status.
        """
        delta = abs(python_conf - tm_conf)
        is_agreement = (python_pred == tm_pred)
        
        if not is_agreement:
            status = "Model Disagreement"
            explanation = f"Model class mismatch: Python predicted '{python_pred}' ({python_conf*100:.1f}%) whereas Teachable Machine predicted '{tm_pred}' ({tm_conf*100:.1f}%)."
        elif python_conf < self.min_certainty_threshold or tm_conf < self.min_certainty_threshold:
            status = "Uncertain Result"
            explanation = f"Both models predicted '{python_pred}', but confidence score is below certainty threshold ({self.min_certainty_threshold*100:.0f}%)."
        elif delta <= self.strong_threshold:
            status = "Strong Match"
            explanation = f"Both models strongly agree on '{python_pred}' with a minimal confidence delta of {delta*100:.1f}%."
        elif delta <= self.acceptable_threshold:
            status = "Acceptable Match"
            explanation = f"Both models agree on '{python_pred}' with an acceptable confidence delta of {delta*100:.1f}%."
        else:
            status = "Weak Match"
            explanation = f"Both models agree on '{python_pred}', but exhibit a large confidence delta of {delta*100:.1f}%."


        return {
            'comparison_status': status,
            'is_agreement': is_agreement,
            'python_prediction': python_pred,
            'python_confidence': float(python_conf),
            'tm_prediction': tm_pred,
            'tm_confidence': float(tm_conf),
            'confidence_delta': round(float(delta), 4),
            'explanation': explanation
        }


if __name__ == '__main__':
    engine = ModelComparisonEngine()
    
    # Quick sanity checks
    print("Test 1 (Strong Match):", engine.compare_models("Valid Claim", 0.95, "Valid Claim", 0.92))
    print("Test 2 (Disagreement):", engine.compare_models("Valid Claim", 0.90, "Invalid Claim", 0.85))
    print("Test 3 (Uncertain):", engine.compare_models("Valid Claim", 0.55, "Valid Claim", 0.58))
