"""
Helper pour charger les données automatiquement
Gère les chemins relatifs/absolus et détecte automatiquement les fichiers
"""

import pandas as pd
import os
from pathlib import Path

class DataLoader:
    """Classe pour charger les données de manière robuste"""
    
    def __init__(self, base_path=None):
        """
        Initialise le loader
        
        Args:
            base_path: Chemin de base du projet (auto-détecté si None)
        """
        if base_path is None:
            # Détecter automatiquement le chemin du projet
            current_path = Path.cwd()
            
            # Si on est dans notebooks/, remonter d'un niveau
            if current_path.name == 'notebooks':
                self.base_path = current_path.parent
            else:
                self.base_path = current_path
        else:
            self.base_path = Path(base_path)
        
        self.data_raw_path = self.base_path / 'data' / 'raw'
        self.data_processed_path = self.base_path / 'data' / 'processed'
        
        print(f" Chemin de base détecté: {self.base_path}")
        print(f" Dossier raw: {self.data_raw_path}")
    
    def find_csv_files(self, directory='raw'):
        """
        Trouve tous les fichiers CSV dans un dossier
        
        Args:
            directory: 'raw' ou 'processed'
        
        Returns:
            Liste des fichiers CSV trouvés
        """
        if directory == 'raw':
            path = self.data_raw_path
        elif directory == 'processed':
            path = self.data_processed_path
        else:
            path = self.base_path / directory
        
        if not path.exists():
            print(f"  Dossier non trouvé: {path}")
            return []
        
        csv_files = list(path.glob('*.csv'))
        return csv_files
    
    def load_train_test(self, train_file='train.csv', test_file='test.csv'):
        """
        Charge les fichiers train et test
        
        Args:
            train_file: Nom du fichier train
            test_file: Nom du fichier test
        
        Returns:
            train_df, test_df
        """
        print("="*60)
        print(" CHARGEMENT DES DONNÉES")
        print("="*60)
        
        # Chercher les fichiers
        train_path = self.data_raw_path / train_file
        test_path = self.data_raw_path / test_file
        
        # Vérifier l'existence
        if not train_path.exists():
            # Essayer de trouver automatiquement
            print(f" {train_file} non trouvé, recherche automatique...")
            csv_files = self.find_csv_files('raw')
            
            if not csv_files:
                raise FileNotFoundError(
                    f"Aucun fichier CSV trouvé dans {self.data_raw_path}\n"
                    f"Assurez-vous que vos fichiers sont dans: {self.data_raw_path}"
                )
            
            print(f"\n Fichiers CSV disponibles:")
            for i, f in enumerate(csv_files, 1):
                print(f"   {i}. {f.name}")
            
            # Trouver train.csv
            train_candidates = [f for f in csv_files if 'train' in f.name.lower()]
            if train_candidates:
                train_path = train_candidates[0]
                print(f"\n Train détecté: {train_path.name}")
            else:
                train_path = csv_files[0]
                print(f"\n 'train.csv' non trouvé, utilisation de: {train_path.name}")
        
        if not test_path.exists():
            print(f" {test_file} non trouvé, recherche automatique...")
            csv_files = self.find_csv_files('raw')
            
            # Trouver test.csv
            test_candidates = [f for f in csv_files if 'test' in f.name.lower()]
            if test_candidates:
                test_path = test_candidates[0]
                print(f" Test détecté: {test_path.name}")
            else:
                print(f" Aucun fichier test trouvé")
                test_df = None
        
        # Charger les données
        print(f"\n Chargement de {train_path.name}...")
        train_df = pd.read_csv(train_path)
        print(f"  {train_df.shape[0]} lignes × {train_df.shape[1]} colonnes")
        
        if test_path.exists():
            print(f"\n Chargement de {test_path.name}...")
            test_df = pd.read_csv(test_path)
            print(f"  {test_df.shape[0]} lignes × {test_df.shape[1]} colonnes")
        else:
            test_df = None
            print("\n Pas de fichier test chargé")
        
        print("\n" + "="*60)
        
        return train_df, test_df
    
    def load_processed_data(self):
        """
        Charge les données préprocessées
        
        Returns:
            dict avec X_train, X_val, X_test, y_train, y_val
        """
        print("="*60)
        print(" CHARGEMENT DES DONNÉES PRÉPROCESSÉES")
        print("="*60)
        
        data = {}
        
        files_to_load = {
            'X_train': 'X_train.csv',
            'X_val': 'X_val.csv',
            'X_test': 'X_test.csv',
            'y_train': 'y_train.csv',
            'y_val': 'y_val.csv'
        }
        
        for key, filename in files_to_load.items():
            filepath = self.data_processed_path / filename
            
            if filepath.exists():
                data[key] = pd.read_csv(filepath)
                print(f" {filename}: {data[key].shape}")
            else:
                print(f" {filename} non trouvé")
                data[key] = None
        
        print("="*60)
        
        return data
    
    def get_data_info(self, df, name="Dataset"):
        """
        Affiche des informations sur le dataset
        
        Args:
            df: DataFrame
            name: Nom du dataset
        """
        print(f"\n{'='*60}")
        print(f" {name.upper()}")
        print('='*60)
        
        print(f"\n Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        print(f"\n Colonnes ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            unique = df[col].nunique()
            print(f"   {i:2d}. {col:25s} | {str(dtype):10s} | {unique:5d} valeurs uniques")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n Valeurs manquantes:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                print(f"   - {col}: {count} ({pct:.2f}%)")
        else:
            print("\n Aucune valeur manquante")
        
        print('='*60)


# ============================================================================
# FONCTION D'UTILISATION SIMPLE
# ============================================================================

def load_churn_data():
    """
    Fonction simple pour charger les données
    
    Returns:
        train_df, test_df
    
    Usage:
        >>> train_df, test_df = load_churn_data()
    """
    loader = DataLoader()
    train_df, test_df = loader.load_train_test()
    return train_df, test_df


def load_churn_processed():
    """
    Charge les données préprocessées
    
    Returns:
        dict avec X_train, X_val, X_test, y_train, y_val
    
    Usage:
        >>> data = load_churn_processed()
        >>> X_train = data['X_train']
    """
    loader = DataLoader()
    return loader.load_processed_data()


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    print(" Test du DataLoader\n")
    
    # Méthode 1: Simple
    print("\n" + "="*70)
    print("MÉTHODE 1: Utilisation simple")
    print("="*70)
    
    try:
        train_df, test_df = load_churn_data()
        print("\n Données chargées avec succès!")
        
        if train_df is not None:
            print("\n Aperçu du train:")
            print(train_df.head())
    
    except FileNotFoundError as e:
        print(f"\n Erreur: {e}")
        print("\n Solutions:")
        print("   1. Vérifiez que les fichiers CSV sont dans data/raw/")
        print("   2. Vérifiez que vous êtes dans le bon dossier")
        print("   3. Créez le dossier: mkdir -p data/raw")
    
    # Méthode 2: Avancée
    print("\n" + "="*70)
    print("MÉTHODE 2: Utilisation avancée")
    print("="*70)
    
    loader = DataLoader()
    csv_files = loader.find_csv_files('raw')
    
    if csv_files:
        print(f"\n {len(csv_files)} fichier(s) CSV trouvé(s):")
        for f in csv_files:
            print(f"   - {f.name}")
    else:
        print("\n Aucun fichier CSV trouvé")
        print(f"   Placez vos fichiers dans: {loader.data_raw_path}")