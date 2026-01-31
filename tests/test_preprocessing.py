"""
Tests unitaires pour le preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestDataLoading:
    """Tests pour le chargement des donnees"""
    
    def test_data_loader_import(self):
        """Test que le module data_loader peut etre importe"""
        try:
            from data_loader import DataLoader
            assert True
        except ImportError:
            pytest.fail("Impossible d'importer DataLoader")
    
    def test_data_loader_initialization(self):
        """Test l'initialisation du DataLoader"""
        from data_loader import DataLoader
        
        loader = DataLoader()
        assert loader.base_path is not None
        assert loader.data_raw_path is not None
        assert loader.data_processed_path is not None
    
    def test_find_csv_files(self):
        """Test la recherche de fichiers CSV"""
        from data_loader import DataLoader
        
        loader = DataLoader()
        csv_files = loader.find_csv_files('raw')
        assert isinstance(csv_files, list)


class TestFeatureEngineering:
    """Tests pour le feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Cree un jeu de donnees de test"""
        np.random.seed(42)
        return pd.DataFrame({
            'Age': np.random.randint(18, 80, 100),
            'Balance': np.random.uniform(0, 200000, 100),
            'EstimatedSalary': np.random.uniform(10000, 150000, 100),
            'Tenure': np.random.randint(0, 10, 100),
            'CreditScore': np.random.randint(300, 850, 100),
            'NumOfProducts': np.random.randint(1, 5, 100),
            'IsActiveMember': np.random.randint(0, 2, 100),
            'HasCrCard': np.random.randint(0, 2, 100)
        })
    
    def test_balance_salary_ratio(self, sample_data):
        """Test la creation du ratio Balance/Salary"""
        sample_data['BalanceSalaryRatio'] = sample_data['Balance'] / (sample_data['EstimatedSalary'] + 1)
        
        assert 'BalanceSalaryRatio' in sample_data.columns
        assert sample_data['BalanceSalaryRatio'].isna().sum() == 0
        assert (sample_data['BalanceSalaryRatio'] >= 0).all()
    
    def test_age_groups(self, sample_data):
        """Test la creation des groupes d'age"""
        sample_data['AgeGroup'] = pd.cut(
            sample_data['Age'],
            bins=[0, 30, 40, 50, 100],
            labels=['Young', 'Adult', 'Middle', 'Senior']
        )
        
        assert 'AgeGroup' in sample_data.columns
        assert sample_data['AgeGroup'].isna().sum() == 0
        assert len(sample_data['AgeGroup'].unique()) <= 4
    
    def test_zero_balance_flag(self, sample_data):
        """Test la creation du flag zero balance"""
        sample_data.loc[:10, 'Balance'] = 0
        sample_data['IsZeroBalance'] = (sample_data['Balance'] == 0).astype(int)
        
        assert 'IsZeroBalance' in sample_data.columns
        assert sample_data['IsZeroBalance'].isin([0, 1]).all()
        assert sample_data['IsZeroBalance'].sum() >= 10
    
    def test_engagement_score(self, sample_data):
        """Test la creation du score d'engagement"""
        sample_data['EngagementScore'] = (
            sample_data['IsActiveMember'] + 
            sample_data['HasCrCard'] + 
            (sample_data['NumOfProducts'] / 4)
        )
        
        assert 'EngagementScore' in sample_data.columns
        assert sample_data['EngagementScore'].isna().sum() == 0
        assert (sample_data['EngagementScore'] >= 0).all()


class TestDataCleaning:
    """Tests pour le nettoyage des donnees"""
    
    @pytest.fixture
    def dirty_data(self):
        """Cree un jeu de donnees avec des problemes"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'CustomerId': [101, 102, 103, 104, 105],
            'Age': [25, 30, np.nan, 40, 45],
            'Balance': [1000, 2000, 3000, 4000, 5000],
            'Exited': [0, 1, 0, 1, 0]
        })
    
    def test_remove_id_columns(self, dirty_data):
        """Test la suppression des colonnes ID"""
        columns_to_drop = ['id', 'CustomerId']
        cleaned = dirty_data.drop(columns=columns_to_drop)
        
        assert 'id' not in cleaned.columns
        assert 'CustomerId' not in cleaned.columns
        assert 'Age' in cleaned.columns
    
    def test_missing_values_detection(self, dirty_data):
        """Test la detection des valeurs manquantes"""
        missing = dirty_data.isnull().sum()
        
        assert missing['Age'] > 0
        assert missing['Balance'] == 0
    
    def test_duplicates_detection(self):
        """Test la detection des doublons"""
        data_with_duplicates = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [4, 4, 5, 6]
        })
        
        duplicates = data_with_duplicates.duplicated().sum()
        assert duplicates > 0


class TestBalancingTechniques:
    """Tests pour les techniques de reequilibrage"""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Cree un jeu de donnees desequilibre"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series([0] * 800 + [1] * 200)
        return X, y
    
    def test_manual_smote_import(self):
        """Test l'import de la classe ManualSMOTE"""
        try:
            from manual_balancing import ManualSMOTE
            assert True
        except ImportError:
            pytest.fail("Impossible d'importer ManualSMOTE")
    
    def test_manual_smote_balancing(self, imbalanced_data):
        """Test le reequilibrage avec SMOTE manuel"""
        from manual_balancing import ManualSMOTE
        
        X, y = imbalanced_data
        smote = ManualSMOTE(k_neighbors=5, random_state=42)
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        assert len(X_resampled) > len(X)
        assert len(y_resampled) > len(y)
        
        # Verifier l'equilibrage
        class_counts = y_resampled.value_counts()
        assert abs(class_counts[0] - class_counts[1]) <= 1
    
    def test_random_oversampling(self, imbalanced_data):
        """Test le random oversampling"""
        X, y = imbalanced_data
        
        minority_class = y.value_counts().idxmin()
        majority_class = y.value_counts().idxmax()
        
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        n_to_sample = len(y_majority) - len(y_minority)
        indices = np.random.choice(X_minority.index, size=n_to_sample, replace=True)
        X_minority_upsampled = X_minority.loc[indices]
        y_minority_upsampled = y_minority.loc[indices]
        
        X_resampled = pd.concat([X_majority, X_minority, X_minority_upsampled], ignore_index=True)
        y_resampled = pd.concat([y_majority, y_minority, y_minority_upsampled], ignore_index=True)
        
        assert len(X_resampled) > len(X)
        class_counts = y_resampled.value_counts()
        assert class_counts[0] == class_counts[1]


class TestEncoding:
    """Tests pour l'encodage des variables"""
    
    @pytest.fixture
    def categorical_data(self):
        """Cree des donnees avec variables categorielles"""
        return pd.DataFrame({
            'Geography': ['France', 'Spain', 'Germany', 'France', 'Spain'],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'NumericFeature': [1, 2, 3, 4, 5]
        })
    
    def test_label_encoding_binary(self, categorical_data):
        """Test le label encoding pour variable binaire"""
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        categorical_data['Gender_Encoded'] = le.fit_transform(categorical_data['Gender'])
        
        assert 'Gender_Encoded' in categorical_data.columns
        assert categorical_data['Gender_Encoded'].nunique() == 2
        assert categorical_data['Gender_Encoded'].isin([0, 1]).all()
    
    def test_onehot_encoding(self, categorical_data):
        """Test le one-hot encoding"""
        dummies = pd.get_dummies(categorical_data['Geography'], prefix='Geography', drop_first=True)
        
        assert len(dummies.columns) == 2  # 3 categories - 1
        assert dummies.values.sum() == len(categorical_data)


class TestScaling:
    """Tests pour la standardisation"""
    
    @pytest.fixture
    def numeric_data(self):
        """Cree des donnees numeriques"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100) * 100 + 50,
            'feature2': np.random.randn(100) * 10 + 5,
            'feature3': np.random.randn(100) * 1000 + 1000
        })
    
    def test_standard_scaler(self, numeric_data):
        """Test le StandardScaler"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        
        # Verifier que la moyenne est proche de 0
        assert np.allclose(scaled_df.mean(), 0, atol=1e-10)
        
        # Verifier que l'ecart-type est proche de 1
        assert np.allclose(scaled_df.std(), 1, atol=0.1)
    
    def test_robust_scaler(self, numeric_data):
        """Test le RobustScaler"""
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        
        # RobustScaler est plus resistant aux outliers
        assert scaled_df is not None
        assert len(scaled_df) == len(numeric_data)


def run_tests():
    """Execute tous les tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()