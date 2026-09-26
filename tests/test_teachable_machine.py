import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.teachable_machine_engine import TeachableMachineClassifier

def test_module_3_teachable_machine():
    tm = TeachableMachineClassifier()
    assert tm.labels == ['Valid Claim', 'Invalid Claim', 'Manual Review'], f"Labels mismatch: {tm.labels}"
    assert tm.image_size == 224, f"Image size mismatch: {tm.image_size}"

    # Create dummy card image for test
    dummy_img_path = 'data/test_card_temp.png'
    os.makedirs('data', exist_ok=True)
    img = Image.new('RGB', (600, 420), color=(240, 240, 240))
    img.save(dummy_img_path)

    res = tm.predict(dummy_img_path)

    assert 'predicted_class' in res, "Missing predicted_class"
    assert res['predicted_class'] in ['Valid Claim', 'Invalid Claim', 'Manual Review'], f"Invalid predicted class: {res['predicted_class']}"
    assert 'confidence' in res, "Missing confidence score"
    assert 0.0 <= res['confidence'] <= 1.0, f"Confidence out of bounds: {res['confidence']}"
    assert 'class_probabilities' in res, "Missing class_probabilities"
    assert len(res['class_probabilities']) == 3, f"Expected 3 class probabilities, got {len(res['class_probabilities'])}"

    prob_sum = sum(res['class_probabilities'].values())
    assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities sum to {prob_sum}, expected 1.0"

    # Cleanup temp file
    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path)

    print("Module 3 (Teachable Machine Classifier) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_module_3_teachable_machine()
