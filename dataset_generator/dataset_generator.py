import os
import random
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from sklearn.model_selection import train_test_split

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

TOTAL_RECORDS = 1500
CLASS_COUNT = 500

PRODUCT_CATEGORIES = {
    'Washing Machine': {'brands': ['LG', 'Samsung', 'Whirlpool'], 'price_range': (400, 1200), 'warranty_months': 24},
    'Laptop': {'brands': ['Dell', 'HP', 'Lenovo', 'Apple'], 'price_range': (600, 2500), 'warranty_months': 12},
    'Refrigerator': {'brands': ['Samsung', 'LG', 'Bosch'], 'price_range': (500, 2000), 'warranty_months': 24},
    'Smartphone': {'brands': ['Apple', 'Samsung', 'Google'], 'price_range': (300, 1400), 'warranty_months': 12}
}

FAULTS = ['Motor Failure', 'Screen Damage', 'Power Failure', 'Water Leakage', 'Overheating', 'Board Defect']
REPAIR_HISTORIES = ['None', 'Authorized Center Repair', 'Unauthorized Repair']

def generate_record(claim_id, target_class):
    category = random.choice(list(PRODUCT_CATEGORIES.keys()))
    brand = random.choice(PRODUCT_CATEGORIES[category]['brands'])
    price = random.randint(*PRODUCT_CATEGORIES[category]['price_range'])
    std_warranty = PRODUCT_CATEGORIES[category]['warranty_months']
    
    # Base Dates
    purchase_date = fake.date_between(start_date='-3y', end_date='-1y')
    serial_number = f"{brand[:2].upper()}-{random.randint(100000, 999999)}"
    
    # Document Hash Simulation
    doc_hash_base = f"{brand}_{serial_number}_{price}_{purchase_date}"
    receipt_hash = hashlib.md5((doc_hash_base + "_receipt").encode('utf-8')).hexdigest()[:16]
    
    # Logic based on target_class
    if target_class == 'Valid Claim':
        # Valid: Product within warranty, valid receipt, matching serial, no unauthorized repairs
        days_after_purchase = random.randint(30, max(31, (std_warranty * 30) - 30))
        claim_date = purchase_date + timedelta(days=days_after_purchase)
        product_age_months = round(days_after_purchase / 30.0, 1)
        remaining_warranty_months = round(max(0.1, std_warranty - product_age_months), 1)
        
        has_receipt = True
        has_warranty_card = True
        has_damage_photo = True
        serial_match = True
        repair_history = random.choice(['None', 'Authorized Center Repair'])
        previous_unauthorized_repairs = False
        duplicate_claim = False
        date_contradiction = False
        missing_docs_count = 0

    elif target_class == 'Invalid Claim':
        # Invalid: Expired warranty, missing mandatory receipt, or unauthorized repair
        invalid_type = random.choice(['expired', 'missing_receipt', 'unauthorized_repair'])
        
        if invalid_type == 'expired':
            days_after_purchase = (std_warranty * 30) + random.randint(30, 365)
            has_receipt = True
            repair_history = random.choice(['None', 'Authorized Center Repair'])
            previous_unauthorized_repairs = False
        elif invalid_type == 'missing_receipt':
            days_after_purchase = random.randint(30, std_warranty * 30)
            has_receipt = False
            repair_history = random.choice(['None', 'Authorized Center Repair'])
            previous_unauthorized_repairs = False
        else: # unauthorized_repair
            days_after_purchase = random.randint(30, std_warranty * 30)
            has_receipt = True
            repair_history = 'Unauthorized Repair'
            previous_unauthorized_repairs = True

        claim_date = purchase_date + timedelta(days=days_after_purchase)
        product_age_months = round(days_after_purchase / 30.0, 1)
        remaining_warranty_months = max(0.0, round(std_warranty - product_age_months, 1))
        
        has_warranty_card = random.choice([True, False])
        has_damage_photo = True
        serial_match = random.choice([True, False])
        duplicate_claim = False
        date_contradiction = False
        missing_docs_count = (0 if has_receipt else 1) + (0 if has_warranty_card else 1)

    else: # Manual Review
        # Borderline: Serial mismatch, date contradictions, duplicate flags, missing key photo
        days_after_purchase = random.randint(30, std_warranty * 30)
        claim_date = purchase_date + timedelta(days=days_after_purchase)
        
        manual_type = random.choice(['serial_mismatch', 'date_contradiction', 'duplicate_flag', 'missing_photo'])
        
        has_receipt = True
        has_warranty_card = True
        has_damage_photo = True
        serial_match = True
        repair_history = 'None'
        previous_unauthorized_repairs = False
        duplicate_claim = False
        date_contradiction = False

        if manual_type == 'serial_mismatch':
            serial_match = False
        elif manual_type == 'date_contradiction':
            # Claim date precedes purchase date
            claim_date = purchase_date - timedelta(days=random.randint(5, 30))
            date_contradiction = True
        elif manual_type == 'duplicate_flag':
            duplicate_claim = True
            receipt_hash = "DUP_HASH_REF_001" # Re-used hash
        elif manual_type == 'missing_photo':
            has_damage_photo = False

        diff_days = (claim_date - purchase_date).days
        product_age_months = round(diff_days / 30.0, 1)
        remaining_warranty_months = round(std_warranty - max(0, product_age_months), 1)
        missing_docs_count = (0 if has_damage_photo else 1)

    return {
        'Claim_ID': f"CLM-{claim_id:04d}",
        'Customer_Name': fake.name(),
        'Product_Category': category,
        'Brand': brand,
        'Model_Number': f"{brand[:3].upper()}-{random.randint(100, 999)}",
        'Serial_Number': serial_number,
        'Purchase_Price': price,
        'Purchase_Date': purchase_date.strftime('%Y-%m-%d'),
        'Claim_Date': claim_date.strftime('%Y-%m-%d'),
        'Product_Age_Months': product_age_months,
        'Warranty_Duration_Months': std_warranty,
        'Remaining_Warranty_Months': remaining_warranty_months,
        'Fault_Type': random.choice(FAULTS),
        'Repair_History': repair_history,
        'Has_Receipt': has_receipt,
        'Has_Warranty_Card': has_warranty_card,
        'Has_Damage_Photo': has_damage_photo,
        'Serial_Number_Match': serial_match,
        'Previous_Unauthorized_Repairs': previous_unauthorized_repairs,
        'Duplicate_Claim_Flag': duplicate_claim,
        'Date_Contradiction_Flag': date_contradiction,
        'Missing_Documents_Count': missing_docs_count,
        'Receipt_Hash': receipt_hash,
        'Claim_Class': target_class
    }

def main():
    records = []
    claim_counter = 1
    
    for cls in ['Valid Claim', 'Invalid Claim', 'Manual Review']:
        for _ in range(CLASS_COUNT):
            records.append(generate_record(claim_counter, cls))
            claim_counter += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Stratified Split: 70% Train (1050), 15% Val (225), 15% Test (225)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['Claim_Class'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['Claim_Class'])

    # Ensure target output directories exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Save CSV files
    df.to_csv('data/full_dataset.csv', index=False)
    train_df.to_csv('data/train/train_claims.csv', index=False)
    val_df.to_csv('data/val/val_claims.csv', index=False)
    test_df.to_csv('data/test/test_claims.csv', index=False)

    print(f"Dataset generated successfully!")
    print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == '__main__':
    main()