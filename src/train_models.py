import sys
import os
import json
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.preprocessing import ClaimPreprocessor

def train_and_evaluate():
    os.makedirs('model', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. Load Data Splits
    train_df = pd.read_csv('data/train/train_claims.csv')
    val_df = pd.read_csv('data/val/val_claims.csv')
    test_df = pd.read_csv('data/test/test_claims.csv')
    
    # 2. Preprocess Data
    preprocessor = ClaimPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    y_val = preprocessor.label_encoder.transform(val_df['Claim_Class'])
    y_test = preprocessor.label_encoder.transform(test_df['Claim_Class'])
    
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    
    # 3. Define Models to Train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')

    best_val_f1 = -1.0
    best_model_name = ""
    best_model = None
    results_summary = []
    
    report_text = "ASSUREX CLAIM ENGINE - TABULAR MODEL EVALUATION REPORT\n"
    report_text += "=" * 65 + "\n\n"
    
    # 4. Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        test_prec_macro = precision_score(y_test, y_test_pred, average='macro')
        test_rec_macro = recall_score(y_test, y_test_pred, average='macro')
        
        results_summary.append({
            'Model': name,
            'Train Acc': train_acc,
            'Val Acc': val_acc,
            'Val F1 (Macro)': val_f1_macro,
            'Test Acc': test_acc,
            'Test F1 (Macro)': test_f1_macro,
            'Test Precision': test_prec_macro,
            'Test Recall': test_rec_macro
        })
        
        report_text += f"Model Algorithm: {name}\n"
        report_text += f"  Train Accuracy:   {train_acc * 100:.2f}%\n"
        report_text += f"  Val Accuracy:     {val_acc * 100:.2f}%\n"
        report_text += f"  Val F1 (Macro):   {val_f1_macro:.4f}\n"
        report_text += f"  Test Accuracy:    {test_acc * 100:.2f}%\n"
        report_text += f"  Test F1 (Macro):  {test_f1_macro:.4f}\n\n"
        
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model_name = name
            best_model = model

    report_text += f"SELECTED BEST MODEL: {best_model_name} (Val F1 Macro: {best_val_f1:.4f})\n"
    
    # 5. Detailed Test Set Classification Report
    y_test_pred = best_model.predict(X_test)
    class_names = preprocessor.label_encoder.classes_
    report_text += "\nDetailed Test Set Classification Report:\n"
    report_text += classification_report(y_test, y_test_pred, target_names=class_names)
    
    # Save Text Report
    with open('reports/tabular_model_report.txt', 'w') as f:
        f.write(report_text)
        
    # Save Model Artifact
    joblib.dump(best_model, 'model/best_tabular_model.joblib')
    
    # Save Metadata JSON for Model Version Tracking (Req Constraint)
    model_metadata = {
        'model_version': '1.0.0',
        'algorithm': best_model_name,
        'trained_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'classes': list(class_names),
        'metrics': [r for r in results_summary if r['Model'] == best_model_name][0],
        'feature_names': preprocessor.feature_names
    }
    
    with open('model/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=4)
        
    print(report_text)
    print("Training complete! Best model saved to model/best_tabular_model.joblib")

if __name__ == '__main__':
    train_and_evaluate()