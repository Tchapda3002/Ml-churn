"""
Techniques de rééquilibrage manuelles
Alternative à imbalanced-learn en cas de problèmes de compatibilité
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class ManualSMOTE:
    """
    Implémentation simplifiée de SMOTE
    Synthetic Minority Over-sampling Technique
    """
    
    def __init__(self, k_neighbors=5, random_state=42):
        """
        Args:
            k_neighbors: Nombre de voisins à utiliser
            random_state: Seed pour la reproductibilité
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        """
        Applique SMOTE sur les données
        
        Args:
            X: Features (DataFrame ou array)
            y: Target (Series ou array)
        
        Returns:
            X_resampled, y_resampled
        """
        # Convertir en DataFrame si nécessaire
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Identifier la classe minoritaire
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        n_to_generate = n_majority - n_minority
        
        print(f"   Classe minoritaire ({minority_class}): {n_minority} échantillons")
        print(f"   Classe majoritaire ({majority_class}): {n_majority} échantillons")
        print(f"   Génération de {n_to_generate} échantillons synthétiques...")
        
        # Extraire les données de la classe minoritaire
        X_minority = X[y == minority_class].values
        
        # Trouver les k plus proches voisins
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(X_minority)
        
        # Générer les échantillons synthétiques
        synthetic_samples = []
        
        for _ in range(n_to_generate):
            # Choisir un échantillon aléatoire
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority[idx]
            
            # Trouver ses voisins
            neighbors_indices = nn.kneighbors([sample], return_distance=False)[0]
            # Exclure l'échantillon lui-même
            neighbors_indices = neighbors_indices[1:]
            
            # Choisir un voisin aléatoire
            neighbor_idx = np.random.choice(neighbors_indices)
            neighbor = X_minority[neighbor_idx]
            
            # Créer un échantillon synthétique
            alpha = np.random.random()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)
        
        # Combiner avec les données originales
        synthetic_samples = np.array(synthetic_samples)
        X_synthetic = pd.DataFrame(synthetic_samples, columns=X.columns)
        y_synthetic = pd.Series([minority_class] * len(synthetic_samples))
        
        X_resampled = pd.concat([X, X_synthetic], ignore_index=True)
        y_resampled = pd.concat([y, y_synthetic], ignore_index=True)
        
        print(f" Dataset rééquilibré: {X_resampled.shape}")
        
        return X_resampled, y_resampled


class ManualRandomOverSampler:
    """Random Over-sampling de la classe minoritaire"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        """
        Applique random over-sampling
        
        Args:
            X: Features
            y: Target
        
        Returns:
            X_resampled, y_resampled
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        # Nombre d'échantillons à générer
        n_to_sample = len(y_majority) - len(y_minority)
        
        # Dupliquer aléatoirement
        indices = np.random.choice(X_minority.index, size=n_to_sample, replace=True)
        X_minority_upsampled = X_minority.loc[indices]
        y_minority_upsampled = y_minority.loc[indices]
        
        # Combiner
        X_resampled = pd.concat([X_majority, X_minority, X_minority_upsampled], ignore_index=True)
        y_resampled = pd.concat([y_majority, y_minority, y_minority_upsampled], ignore_index=True)
        
        return X_resampled, y_resampled


class ManualRandomUnderSampler:
    """Random Under-sampling de la classe majoritaire"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        """
        Applique random under-sampling
        
        Args:
            X: Features
            y: Target
        
        Returns:
            X_resampled, y_resampled
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        
        # Échantillonner la classe majoritaire
        n_minority = len(y_minority)
        indices_majority = np.random.choice(X_majority.index, size=n_minority, replace=False)
        X_majority_downsampled = X_majority.loc[indices_majority]
        y_majority_downsampled = y_majority.loc[indices_majority]
        
        # Combiner
        X_resampled = pd.concat([X_majority_downsampled, X_minority], ignore_index=True)
        y_resampled = pd.concat([y_majority_downsampled, y_minority], ignore_index=True)
        
        return X_resampled, y_resampled


def create_balanced_datasets_manual(X, y, random_state=42):
    """
    Crée plusieurs versions du dataset avec différentes techniques de rééquilibrage
    Version manuelle sans dépendances externes
    
    Args:
        X: Features
        y: Target
        random_state: Seed
    
    Returns:
        dict avec les différentes versions
    """
    datasets = {}
    
    print("="*60)
    print("CRÉATION DES DATASETS RÉÉQUILIBRÉS (VERSION MANUELLE)")
    print("="*60)
    
    # 1. Original
    datasets['original'] = (X.copy(), y.copy())
    print(f"\n Original: {X.shape}")
    print(f"   Distribution: {y.value_counts().to_dict()}")
    
    # 2. SMOTE manuel
    print("\n Application de SMOTE (implémentation manuelle)...")
    try:
        smote = ManualSMOTE(k_neighbors=5, random_state=random_state)
        X_smote, y_smote = smote.fit_resample(X, y)
        datasets['smote'] = (X_smote, y_smote)
        print(f"   Distribution finale: {y_smote.value_counts().to_dict()}")
    except Exception as e:
        print(f" Erreur SMOTE: {str(e)[:50]}...")
    
    # 3. Random Over-sampling
    print("\n Application de Random Over-sampling...")
    oversampler = ManualRandomOverSampler(random_state=random_state)
    X_over, y_over = oversampler.fit_resample(X, y)
    datasets['oversampling'] = (X_over, y_over)
    print(f" Over-sampling: {X_over.shape}")
    print(f" Distribution: {y_over.value_counts().to_dict()}")
    
    # 4. Random Under-sampling
    print("\n Application de Random Under-sampling...")
    undersampler = ManualRandomUnderSampler(random_state=random_state)
    X_under, y_under = undersampler.fit_resample(X, y)
    datasets['undersampling'] = (X_under, y_under)
    print(f" Under-sampling: {X_under.shape}")
    print(f"   Distribution: {y_under.value_counts().to_dict()}")
    
    return datasets


# Pour utilisation directe
if __name__ == "__main__":
    # Exemple d'utilisation
    print("Test des techniques de rééquilibrage manuelles\n")
    
    # Créer des données de test
    from sklearn.datasets import make_classification
    
    X_test, y_test = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.8, 0.2],  # Déséquilibre
        random_state=42
    )
    
    X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(10)])
    y_test = pd.Series(y_test, name='target')
    
    print(f"Dataset de test: {X_test.shape}")
    print(f"Distribution originale: {y_test.value_counts().to_dict()}\n")
    
    # Tester les techniques
    datasets = create_balanced_datasets_manual(X_test, y_test)
    
    print("\n Test terminé!")