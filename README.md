# AssureX Claim Verification Engine 🚀

An end-to-end, AI-powered claim verification engine designed to automate electronic claim processing. The system combines **EasyOCR** for document extraction, an **XGBoost machine learning model** for pattern classification, and a deterministic **Rule Engine** for policy compliance enforcement.

---

## 🛠 System Architecture & Data Flow

```
+-------------------+      +-----------------------+
|  Input Claim Data | ---> |  OCR Processing Module |
|  (JSON / Images)  |      | (Extract Text/Serial)  |
+-------------------+      +-----------------------+
                                       |
                                       v
                            +-----------------------+
                            |   XGBoost ML Model    |
                            |  (Predicts Validity)  |
                            +-----------------------+
                                       |
                                       v
                            +-----------------------+
                            |  Policy Rule Engine   |
                            | (Hard Rejections /    |
                            |   Manual Reviews)     |
                            +-----------------------+
                                       |
                                       v
                            +-----------------------+
                            |     Final Verdict     |
                            | (Valid/Invalid/Review)|
                            +-----------------------+
```

---

## 📋 Prerequisites & Requirements

- **Python:** 3.10 or higher
- **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

---

## ⚙️ Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Abdullah-Murshid/assurex-claim-engine.git
cd assurex-claim-engine
```

### 2. Create and Activate a Virtual Environment

```bash
# Create a virtual environment
uv venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## 📁 Project Structure

```
assurex-claim-engine/
├── data/
│   ├── raw/               # Raw training datasets & images
│   ├── processed/         # Feature-engineered CSV files
│   └── test/              # Holdout evaluation sets (test_claims.csv)
├── models/
│   ├── xgboost_model.json # Trained XGBoost model artifact
│   └── label_encoder.pkl  # Categorical encoders
├── reports/
│   ├── 30_claim_model_comparison.csv
│   └── 30_claim_model_comparison.md
├── src/
│   ├── __init__.py
│   ├── preprocess.py      # Feature engineering & preprocessing
│   ├── train.py           # Model training script
│   ├── ocr_engine.py      # EasyOCR extraction pipeline
│   ├── rule_engine.py     # Business rule enforcement
│   ├── pipeline.py        # E2E unified orchestration engine
│   └── generate_report.py # Evaluation & comparison report generator
├── tests/
│   └── test_full_system.py # End-to-end unit and integration tests
├── AI_USAGE.md            # Log of AI usage and assistance
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

---

## 🚀 Step-by-Step Execution Guide

**Step 1: Preprocess Data & Train ML Model**

```bash
uv run python src/train.py
```

**Step 2: Run E2E Verification Engine on a Sample Claim**

```bash
uv run python src/pipeline.py
```

**Step 3: Run Automated Test Suite**

```bash
uv run python tests/test_full_system.py
```

**Step 4: Generate 30-Claim Evaluation Report**

```bash
uv run python src/generate_report.py
```

---

## 📊 Key Features & Decision Logic

- **OCR Extraction** (`src/ocr_engine.py`): Automatically extracts invoice numbers, serials, purchase dates, and prices from receipt photos using EasyOCR.

- **ML Classification** (`src/train.py`): Scores claim risk and outputs a predicted class (Valid Claim, Invalid Claim, Manual Review) with a confidence probability.

- **Deterministic Policy Rules** (`src/rule_engine.py`): Overrides ML outcomes when strict policy criteria are violated:
  - **Hard Rejection:** Expired warranty period, missing core receipt document.
  - **Manual Review:** Serial number mismatches, date contradictions, flagged previous unauthorized repairs.

---

## 📜 Dependencies (`requirements.txt`)

```
torch
torchvision
easyocr
xgboost
scikit-learn
pandas
numpy
tabulate
pytest
```


## auth credentials 
# admin
# admin123 

## user 
# samad-r
# 12345