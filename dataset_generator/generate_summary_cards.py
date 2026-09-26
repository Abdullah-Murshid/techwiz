import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

CARD_WIDTH = 600
CARD_HEIGHT = 420

def format_date(date_str, format_type=1):
    try:
        dt = datetime.strptime(str(date_str), '%Y-%m-%d')
        if format_type == 1:
            return dt.strftime('%Y-%m-%d')
        elif format_type == 2:
            return dt.strftime('%d/%m/%Y')
        else:
            return dt.strftime('%b %d, %Y')
    except Exception:
        return str(date_str)

def create_card_image(row, output_path, variation=1):
    # Select background & styling based on variation
    if variation == 1:
        bg_color = (245, 247, 250)
        card_bg = (255, 255, 255)
        text_color = (30, 41, 59)
        accent_color = (37, 99, 235)
        date_fmt = 1
        x_offset = 40
    else:
        bg_color = (238, 242, 246)
        card_bg = (250, 253, 255)
        text_color = (15, 23, 42)
        accent_color = (13, 148, 136)
        date_fmt = 2
        x_offset = 45

    image = Image.new('RGB', (CARD_WIDTH, CARD_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Draw Inner Card Frame
    draw.rectangle([20, 20, CARD_WIDTH - 20, CARD_HEIGHT - 20], fill=card_bg, outline=accent_color, width=2)

    # Header
    draw.text((x_offset, 35), f"WARRANTY CLAIM SUMMARY | ID: {row['Claim_ID']}", fill=accent_color)
    draw.line([(x_offset, 60), (CARD_WIDTH - x_offset, 60)], fill=accent_color, width=2)

    p_date = format_date(row.get('Purchase_Date', '2023-01-01'), date_fmt)
    c_date = format_date(row.get('Claim_Date', '2023-06-01'), date_fmt)

    # Claim Information (Excludes ML prediction or confidence scores)
    lines = [
        f"Product Category: {row['Product_Category']}",
        f"Brand & Model: {row['Brand']} ({row['Model_Number']})",
        f"Serial Number: {row['Serial_Number']}",
        f"Purchase Date: {p_date} | Claim Date: {c_date}",
        f"Product Age: {row['Product_Age_Months']} Mos | Warranty Rem: {row['Remaining_Warranty_Months']} Mos",
        f"Reported Fault: {row['Fault_Type']}",
        f"Repair History: {row.get('Repair_History', 'None')}",
        f"Receipt Available: {'YES' if row['Has_Receipt'] else 'NO'} | Serial Match: {'YES' if row['Serial_Number_Match'] else 'NO'}",
        f"Missing Documents Count: {row['Missing_Documents_Count']}"
    ]

    y_pos = 75
    for line in lines:
        draw.text((x_offset, y_pos), line, fill=text_color)
        y_pos += 30

    # Footer
    draw.text((x_offset, CARD_HEIGHT - 40), f"AssureX Verification Artifact | Claim ID: {row['Claim_ID']} [Var #{variation}]", fill=(100, 116, 139))

    image.save(output_path)

def generate_cards_for_split(csv_path, output_dir, is_train=False):
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        cls_folder = os.path.join(output_dir, str(row['Claim_Class']).replace(" ", "_"))
        os.makedirs(cls_folder, exist_ok=True)
        
        # Variation 1
        img_name_1 = f"{row['Claim_ID']}_v1.png"
        create_card_image(row, os.path.join(cls_folder, img_name_1), variation=1)

        # Variation 2 for training data (to reach >= 2,100 training images)
        if is_train:
            img_name_2 = f"{row['Claim_ID']}_v2.png"
            create_card_image(row, os.path.join(cls_folder, img_name_2), variation=2)

def main():
    print("Generating Claim Summary Cards...")
    generate_cards_for_split('data/train/train_claims.csv', 'data/train/cards', is_train=True)
    generate_cards_for_split('data/val/val_claims.csv', 'data/val/cards', is_train=False)
    generate_cards_for_split('data/test/test_claims.csv', 'data/test/cards', is_train=False)
    print("All Claim Summary Cards created successfully!")

if __name__ == '__main__':
    main()