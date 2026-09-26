import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class ClaimPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.feature_names = []
        self.version = "1.0.0"
        
    def _extract_features(self, df):
        df_processed = df.copy()
        
        # Ensure mandatory columns exist with default fallbacks if missing
        required_cols = {
            'Purchase_Price': 500.0,
            'Product_Age_Months': 6.0,
            'Warranty_Duration_Months': 12.0,
            'Remaining_Warranty_Months': 6.0,
            'Missing_Documents_Count': 0,
            'Has_Receipt': True,
            'Has_Warranty_Card': True,
            'Has_Damage_Photo': True,
            'Serial_Number_Match': True,
            'Previous_Unauthorized_Repairs': False,
            'Duplicate_Claim_Flag': False,
            'Date_Contradiction_Flag': False,
            'Product_Category': 'Unknown',
            'Brand': 'Unknown',
            'Fault_Type': 'General Defect',
            'Repair_History': 'None'
        }
        
        for col, default_val in required_cols.items():
            if col not in df_processed.columns:
                df_processed[col] = default_val

        # Derived Feature Engineering
        df_processed['Doc_Completeness_Score'] = (
            df_processed['Has_Receipt'].astype(int) + 
            df_processed['Has_Warranty_Card'].astype(int) + 
            df_processed['Has_Damage_Photo'].astype(int)
        )
        
        num_cols = [
            'Purchase_Price', 'Product_Age_Months', 'Warranty_Duration_Months',
            'Remaining_Warranty_Months', 'Missing_Documents_Count', 'Doc_Completeness_Score'
        ]
        cat_cols = ['Product_Category', 'Brand', 'Fault_Type', 'Repair_History']
        bool_cols = [
            'Has_Receipt', 'Has_Warranty_Card', 'Has_Damage_Photo', 
            'Serial_Number_Match'
        ]
        
        return df_processed, num_cols, cat_cols, bool_cols

    def fit_transform(self, df):
        df_processed, num_cols, cat_cols, bool_cols = self._extract_features(df)
        
        # Target Label Encoding
        y = self.label_encoder.fit_transform(df_processed['Claim_Class'])
        
        # Impute & Scale Numerics
        X_num = self.num_imputer.fit_transform(df_processed[num_cols])
        X_bool = df_processed[bool_cols].to_numpy(dtype=np.float64)
        
        # Impute & OHE Categoricals
        df_cat_imputed = self.cat_imputer.fit_transform(df_processed[cat_cols])
        X_cat = self.ohe.fit_transform(df_cat_imputed).astype(np.float64)
        
        # Combine & Standard Scale
        X_combined = np.hstack([X_num, X_bool, X_cat])
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # Store Feature Names
        ohe_feature_names = self.ohe.get_feature_names_out(cat_cols)
        self.feature_names = num_cols + bool_cols + list(ohe_feature_names)
        
        return X_scaled, y

    def transform(self, df):
        df_processed, num_cols, cat_cols, bool_cols = self._extract_features(df)
        
        X_num = self.num_imputer.transform(df_processed[num_cols])
        X_bool = df_processed[bool_cols].to_numpy(dtype=np.float64)
        
        df_cat_imputed = self.cat_imputer.transform(df_processed[cat_cols])
        X_cat = self.ohe.transform(df_cat_imputed).astype(np.float64)
        
        X_combined = np.hstack([X_num, X_bool, X_cat])
        X_scaled = self.scaler.transform(X_combined)
        
        return X_scaled

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    train_df = pd.read_csv('data/train/train_claims.csv')
    preprocessor = ClaimPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    print("Preprocessor updated successfully!")