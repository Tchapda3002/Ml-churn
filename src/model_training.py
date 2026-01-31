"""
Scripts de modelisation reutilisables avec pipelines sklearn
Projet Churn Prediction - Fortuneo Banque
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import joblib
from pathlib import Path
import time


class ChurnModelTrainer:
    """
    Classe pour entrainer et evaluer des modeles de churn
    """
    
    def __init__(self, models_dict, scale_features=True):
        """
        Args:
            models_dict: Dictionnaire {nom_modele: instance_modele}
            scale_features: Si True, applique StandardScaler
        """
        self.models_dict = models_dict
        self.scale_features = scale_features
        self.trained_models = {}
        self.results = []
        self.pipelines = {}
        
    def create_pipeline(self, model, model_name):
        """
        Cree un pipeline sklearn
        
        Args:
            model: Instance du modele
            model_name: Nom du modele
            
        Returns:
            Pipeline sklearn
        """
        steps = []
        
        if self.scale_features:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('classifier', model))
        
        pipeline = Pipeline(steps)
        return pipeline
    
    def train_model(self, model, X_train, y_train, model_name):
        """
        Entraine un modele
        
        Args:
            model: Modele a entrainer
            X_train, y_train: Donnees d'entrainement
            model_name: Nom du modele
            
        Returns:
            Modele entraine, temps d'entrainement
        """
        print(f"\nEntrainement de {model_name}...")
        
        pipeline = self.create_pipeline(model, model_name)
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"  Temps d'entrainement: {train_time:.2f} secondes")
        
        self.pipelines[model_name] = pipeline
        return pipeline, train_time
    
    def evaluate_model(self, pipeline, X_train, y_train, X_val, y_val, model_name):
        """
        Evalue un modele sur train et validation
        
        Args:
            pipeline: Pipeline sklearn entraine
            X_train, y_train: Donnees d'entrainement
            X_val, y_val: Donnees de validation
            model_name: Nom du modele
            
        Returns:
            dict avec les metriques
        """
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        results = {
            'model_name': model_name,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba)
        }
        
        return results
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Entraine tous les modeles
        
        Args:
            X_train, y_train: Donnees d'entrainement
            X_val, y_val: Donnees de validation
            
        Returns:
            DataFrame avec les resultats
        """
        print("="*60)
        print("ENTRAINEMENT DE TOUS LES MODELES")
        print("="*60)
        
        for model_name, model in self.models_dict.items():
            try:
                pipeline, train_time = self.train_model(
                    model, X_train, y_train, model_name
                )
                
                results = self.evaluate_model(
                    pipeline, X_train, y_train, X_val, y_val, model_name
                )
                
                results['train_time'] = train_time
                self.results.append(results)
                self.trained_models[model_name] = pipeline
                
                print(f"  Val ROC-AUC: {results['val_roc_auc']:.4f}")
                
            except Exception as e:
                print(f"  Erreur avec {model_name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('val_roc_auc', ascending=False)
        
        return results_df
    
    def cross_validate(self, model_name, X, y, cv=5, scoring='roc_auc'):
        """
        Effectue une cross-validation
        
        Args:
            model_name: Nom du modele a valider
            X, y: Donnees completes
            cv: Nombre de folds
            scoring: Metrique de scoring
            
        Returns:
            Scores de cross-validation
        """
        if model_name not in self.pipelines:
            print(f"Modele {model_name} non entraine")
            return None
        
        pipeline = self.pipelines[model_name]
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        print(f"\nCross-validation pour {model_name} ({cv} folds)...")
        
        scores = cross_val_score(
            pipeline, X, y, 
            cv=cv_strategy, 
            scoring=scoring, 
            n_jobs=-1
        )
        
        print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"  Moyenne: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def get_best_model(self):
        """
        Retourne le meilleur modele base sur val_roc_auc
        
        Returns:
            nom_modele, pipeline, resultats
        """
        if not self.results:
            print("Aucun modele entraine")
            return None, None, None
        
        results_df = pd.DataFrame(self.results)
        best_idx = results_df['val_roc_auc'].idxmax()
        best_result = results_df.loc[best_idx]
        best_model_name = best_result['model_name']
        best_pipeline = self.pipelines[best_model_name]
        
        return best_model_name, best_pipeline, best_result
    
    def save_model(self, model_name, save_path):
        """
        Sauvegarde un modele
        
        Args:
            model_name: Nom du modele a sauvegarder
            save_path: Chemin de sauvegarde
        """
        if model_name not in self.pipelines:
            print(f"Modele {model_name} non trouve")
            return
        
        pipeline = self.pipelines[model_name]
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(pipeline, save_path)
        print(f"Modele sauvegarde: {save_path}")
    
    def load_model(self, load_path, model_name=None):
        """
        Charge un modele sauvegarde
        
        Args:
            load_path: Chemin du modele
            model_name: Nom optionnel pour le modele
        """
        pipeline = joblib.load(load_path)
        
        if model_name is None:
            model_name = Path(load_path).stem
        
        self.pipelines[model_name] = pipeline
        print(f"Modele charge: {model_name}")
        
        return pipeline


class ModelOptimizer:
    """
    Classe pour optimiser les hyperparametres
    """
    
    def __init__(self, model, param_grid, scoring='roc_auc', cv=3, n_iter=20):
        """
        Args:
            model: Modele de base
            param_grid: Grille de parametres
            scoring: Metrique d'optimisation
            cv: Nombre de folds
            n_iter: Nombre d'iterations pour RandomizedSearch
        """
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def optimize(self, X_train, y_train, search_type='random'):
        """
        Optimise les hyperparametres
        
        Args:
            X_train, y_train: Donnees d'entrainement
            search_type: 'random' ou 'grid'
            
        Returns:
            Meilleur modele, meilleurs parametres
        """
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        
        print("="*60)
        print("OPTIMISATION DES HYPERPARAMETRES")
        print("="*60)
        
        if search_type == 'random':
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=1
            )
        
        print(f"\nType de recherche: {search_type}")
        print(f"Nombre de combinaisons: {self.n_iter if search_type == 'random' else 'all'}")
        
        start_time = time.time()
        search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        print(f"\nTemps d'optimisation: {optimization_time:.2f} secondes")
        print(f"Meilleur score ({self.scoring}): {self.best_score:.4f}")
        print(f"\nMeilleurs parametres:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_model, self.best_params


def get_default_models():
    """
    Retourne un dictionnaire de modeles par defaut
    
    Returns:
        dict {nom_modele: instance_modele}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
    }
    
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, 
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
    except ImportError:
        pass
    
    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            iterations=100, learning_rate=0.1,
            random_state=42, verbose=0
        )
    except ImportError:
        pass
    
    return models


def get_param_grids():
    """
    Retourne les grilles de parametres pour chaque modele
    
    Returns:
        dict {nom_modele: param_grid}
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [10, 20, 30]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 10],
            'num_leaves': [31, 50, 70]
        },
        'CatBoost': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8]
        }
    }
    
    return param_grids


if __name__ == "__main__":
    print("Module de modelisation charge")
    print("Utilisation:")
    print("  from model_training import ChurnModelTrainer, get_default_models")
    print("  models = get_default_models()")
    print("  trainer = ChurnModelTrainer(models)")
    print("  results = trainer.train_all_models(X_train, y_train, X_val, y_val)")