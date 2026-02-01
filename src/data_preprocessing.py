# src/preprocessing.py - VERSION COMPLÈTE CORRIGÉE

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

# ========== FONCTION DE BASE ==========

def clean_data_base(df):
    """Votre fonction clean_data"""
    df_clean = df.copy()
    
    # Supprimer colonnes ID
    cols_to_drop = []
    for col in ['id', 'CustomerId', 'Surname']:
        if col in df_clean.columns:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    return df_clean


# ========== TRANSFORMER 1 : NETTOYAGE (CORRIGÉ) ==========

class DataCleaner(BaseEstimator, TransformerMixin):
    """Nettoyage basique"""
    
    def fit(self, X, y=None):
        self.is_fitted_ = True  # ← Obligatoire pour sklearn
        return self
    
    def transform(self, X):
        return clean_data_base(X)


def create_cleaning_pipeline():
    return Pipeline([
        ('cleaner', DataCleaner())
    ])


# ========== TRANSFORMER 2 : FEATURE ENGINEERING (CORRIGÉ) ==========

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crée les features"""
    
    def fit(self, X, y=None):
        # Mémoriser les bins calculés sur train
        self.bins_ = {}  # ← Attribut avec underscore
        
        if 'Age' in X.columns:
            self.bins_['Age'] = [0, 30, 40, 50, 100]
            self.age_labels_ = ['Young', 'Adult', 'Middle', 'Senior']
        
        if 'Tenure' in X.columns:
            self.bins_['Tenure'] = [-1, 2, 5, 10]
            self.tenure_labels_ = ['New', 'Regular', 'Loyal']
        
        if 'CreditScore' in X.columns:
            self.bins_['CreditScore'] = [0, 600, 700, 850]
            self.credit_labels_ = ['Poor', 'Good', 'Excellent']
        
        return self
    
    def transform(self, X):
        df_new = X.copy()
        
        # 1. BalanceSalaryRatio
        if 'Balance' in df_new.columns and 'EstimatedSalary' in df_new.columns:
            df_new['BalanceSalaryRatio'] = df_new['Balance'] / (df_new['EstimatedSalary'] + 1)
        
        # 2. AgeGroup
        if 'Age' in df_new.columns and 'Age' in self.bins_:
            df_new['AgeGroup'] = pd.cut(df_new['Age'], 
                                        bins=self.bins_['Age'],
                                        labels=self.age_labels_)
        
        # 3. TenureGroup
        if 'Tenure' in df_new.columns and 'Tenure' in self.bins_:
            df_new['TenureGroup'] = pd.cut(df_new['Tenure'],
                                           bins=self.bins_['Tenure'],
                                           labels=self.tenure_labels_)
        
        # 4. CreditScoreGroup
        if 'CreditScore' in df_new.columns and 'CreditScore' in self.bins_:
            df_new['CreditScoreGroup'] = pd.cut(df_new['CreditScore'],
                                                bins=self.bins_['CreditScore'],
                                                labels=self.credit_labels_)
        
        # 5. IsZeroBalance
        if 'Balance' in df_new.columns:
            df_new['IsZeroBalance'] = (df_new['Balance'] == 0).astype(int)
        
        # 6. HasMultipleProducts
        if 'NumOfProducts' in df_new.columns:
            df_new['HasMultipleProducts'] = (df_new['NumOfProducts'] > 1).astype(int)
        
        # 7. EngagementScore
        if all(col in df_new.columns for col in ['IsActiveMember', 'HasCrCard', 'NumOfProducts']):
            df_new['EngagementScore'] = (
                df_new['IsActiveMember'] + 
                df_new['HasCrCard'] + 
                (df_new['NumOfProducts'] / 4)
            )
        
        # 8. Age_Balance_Interaction
        if 'Age' in df_new.columns and 'Balance' in df_new.columns:
            df_new['Age_Balance_Interaction'] = df_new['Age'] * df_new['Balance'] / 100000
        
        # 9. TenureAgeRatio
        if 'Tenure' in df_new.columns and 'Age' in df_new.columns:
            df_new['TenureAgeRatio'] = df_new['Tenure'] / df_new['Age']
        
        return df_new


def create_feature_engineering_pipeline():
    return Pipeline([
        ('feature_engineer', FeatureEngineer())
    ])


# ========== TRANSFORMER 3 : ENCODAGE ==========

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode les variables catégorielles"""
    
    def fit(self, X, y=None):
        self.encoders_ = {}  # ← Attribut avec underscore
        self.onehot_columns_ = {}
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            unique_values = X[col].nunique()
            
            if unique_values == 2:
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders_[col] = ('label', le)
            
            elif unique_values <= 10:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                self.encoders_[col] = ('onehot', None)
                self.onehot_columns_[col] = dummies.columns.tolist()
            
            else:
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders_[col] = ('label', le)
        
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        
        for col, (method, encoder) in self.encoders_.items():
            if col not in X_encoded.columns:
                continue
            
            if method == 'label':
                X_encoded[col] = X_encoded[col].map(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                X_encoded[col] = encoder.transform(X_encoded[col])
            
            elif method == 'onehot':
                dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
                
                for dummy_col in self.onehot_columns_[col]:
                    if dummy_col not in dummies.columns:
                        dummies[dummy_col] = 0
                
                extra_cols = set(dummies.columns) - set(self.onehot_columns_[col])
                for c in extra_cols:
                    dummies = dummies.drop(columns=[c])
                
                dummies = dummies[self.onehot_columns_[col]]
                
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded = X_encoded.drop(columns=[col])
        
        return X_encoded


def create_encoding_pipeline():
    return Pipeline([
        ('encoder', CategoricalEncoder())
    ])


# ========== TRANSFORMER 4 : SCALING ==========

class FeatureScaler(BaseEstimator, TransformerMixin):
    """Normalise les features"""
    
    def __init__(self, method='standard'):
        self.method = method
    
    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler_ = StandardScaler()  # ← Attribut avec underscore
        elif self.method == 'robust':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError("method doit être 'standard' ou 'robust'")
        
        self.scaler_.fit(X)
        self.columns_ = X.columns.tolist()
        
        return self
    
    def transform(self, X):
        X_scaled = self.scaler_.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)


def create_scaling_pipeline(method='standard'):
    return Pipeline([
        ('scaler', FeatureScaler(method=method))
    ])