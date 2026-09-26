import os
import pandas as pd

def test_dataset_generator():
    assert os.path.exists('data/full_dataset.csv'), "full_dataset.csv missing"
    assert os.path.exists('data/train/train_claims.csv'), "train_claims.csv missing"
    assert os.path.exists('data/val/val_claims.csv'), "val_claims.csv missing"
    assert os.path.exists('data/test/test_claims.csv'), "test_claims.csv missing"

    df = pd.read_csv('data/full_dataset.csv')
    assert len(df) == 1500, f"Expected 1500 records, got {len(df)}"

    class_counts = df['Claim_Class'].value_counts()
    assert class_counts['Valid Claim'] == 500, f"Expected 500 Valid Claim, got {class_counts.get('Valid Claim')}"
    assert class_counts['Invalid Claim'] == 500, f"Expected 500 Invalid Claim, got {class_counts.get('Invalid Claim')}"
    assert class_counts['Manual Review'] == 500, f"Expected 500 Manual Review, got {class_counts.get('Manual Review')}"

    # Verify repair history and receipt hash columns exist
    assert 'Repair_History' in df.columns, "Repair_History column missing"
    assert 'Receipt_Hash' in df.columns, "Receipt_Hash column missing"

    train_df = pd.read_csv('data/train/train_claims.csv')
    val_df = pd.read_csv('data/val/val_claims.csv')
    test_df = pd.read_csv('data/test/test_claims.csv')

    assert len(train_df) == 1050, f"Expected 1050 train records, got {len(train_df)}"
    assert len(val_df) == 225, f"Expected 225 val records, got {len(val_df)}"
    assert len(test_df) == 225, f"Expected 225 test records, got {len(test_df)}"

    print("Module 1 (Dataset Generator) Unit Tests Passed Successfully!")

if __name__ == '__main__':
    test_dataset_generator()
