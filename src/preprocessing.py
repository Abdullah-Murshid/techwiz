import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import joblib
import os

class ClaimPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = []
        
    def _extract_features(self, df):
        df_processed = df.copy()
        
        # Engineering Risk & Policy Flags
        df_processed['Warranty_Expired_Flag'] = (df_processed['Remaining_Warranty_Months'] <= 0).astype(int)
        df_processed['Doc_Completeness_Score'] = (
            df_processed['Has_Receipt'].astype(int) + 
            df_processed['Has_Warranty_Card'].astype(int) + 
            df_processed['Has_Damage_Photo'].astype(int)
        )
        
        num_cols = [
            'Purchase_Price', 'Product_Age_Months', 'Warranty_Duration_Months',
            'Remaining_Warranty_Months', 'Missing_Documents_Count', 'Doc_Completeness_Score'
        ]
        cat_cols = ['Product_Category', 'Brand', 'Fault_Type']
        bool_cols = [
            'Has_Receipt', 'Has_Warranty_Card', 'Has_Damage_Photo', 
            'Serial_Number_Match', 'Previous_Unauthorized_Repairs', 
            'Duplicate_Claim_Flag', 'Date_Contradiction_Flag', 'Warranty_Expired_Flag'
        ]
        
        X_num = df_processed[num_cols].to_numpy(dtype=np.float64)
        X_bool = df_processed[bool_cols].to_numpy(dtype=np.float64)
        
        return df_processed, X_num, X_bool, cat_cols

    def fit_transform(self, df):
        df_processed, X_num, X_bool, cat_cols = self._extract_features(df)
        
        y = self.label_encoder.fit_transform(df_processed['Claim_Class'])
        X_cat = self.ohe.fit_transform(df_processed[cat_cols]).astype(np.float64)
        
        # Ensure purely 2D Float Array
        X_combined = np.hstack([X_num, X_bool, X_cat])
        X_scaled = self.scaler.fit_transform(X_combined)
        
        return X_scaled, y

    def transform(self, df):
        df_processed, X_num, X_bool, cat_cols = self._extract_features(df)
        X_cat = self.ohe.transform(df_processed[cat_cols]).astype(np.float64)
        
        # Ensure purely 2D Float Array
        X_combined = np.hstack([X_num, X_bool, X_cat])
        X_scaled = self.scaler.transform(X_combined)
        
        return X_scaled

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    train_df = pd.read_csv('data/train/train_claims.csv')
    preprocessor = ClaimPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    print("Preprocessor updated successfully.")