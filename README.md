# Churn Prediction - Fortuneo Banque

Projet de Machine Learning pour predire le churn des clients d'une banque en ligne.

## Objectif

Identifier les clients susceptibles de quitter la banque afin de prendre des mesures preventives ciblees pour les retenir.

## Dataset

**Source**: [Kaggle - Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

**Caracteristiques**:
- Train set: ~165,000 observations
- Variables: 14 features + 1 target (Exited)
- Desequilibre: ~20% de churn

**Variables**:
- `CreditScore`: Score de credit du client
- `Geography`: Pays (France, Spain, Germany)
- `Gender`: Genre (Male/Female)
- `Age`: Age du client
- `Tenure`: Anciennete (annees)
- `Balance`: Solde du compte
- `NumOfProducts`: Nombre de produits
- `HasCrCard`: Possession carte de credit
- `IsActiveMember`: Membre actif
- `EstimatedSalary`: Salaire estime
- **`Exited`**: Variable cible (0=Reste, 1=Parti)

## Structure du Projet

```
fortuneo-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/              # Donnees brutes (train.csv, test.csv)
│   └── processed/        # Donnees preprocessees
│
├── notebooks/
│   ├── churn_01_analyse_exploratoire.ipynb
│   ├── churn_02_preprocessing.ipynb
│   └── churn_03_modelisation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── manual_balancing.py
│   └── model_training.py
│
├── models/
│   └── saved_models/     # Modeles entraines sauvegardes
│
└── tests/                # Tests unitaires
```

## Quick Start

### 1. Cloner et installer

```bash
git clone https://github.com/votre-username/fortuneo-churn-prediction.git
cd fortuneo-churn-prediction
pip install -r requirements.txt
```

### 2. Telecharger les donnees

Placez `train.csv` et `test.csv` dans `data/raw/`

### 3. Executer les notebooks

```bash
jupyter notebook notebooks/
```

Executez dans l'ordre:
1. `churn_01_analyse_exploratoire.ipynb`
2. `churn_02_preprocessing.ipynb`
3. `churn_03_modelisation.ipynb`
4. `churn_04_evaluation_finale.ipynb`

### 4. Lancer l'API

```bash
cd deployment/
python app.py
```

L'API sera disponible sur http://localhost:8000

### 5. Tester l'API

```bash
python test_api.py
```

Ou visitez http://localhost:8000/docs pour la documentation interactive.

## Structure du Projet

### Performance des Modeles

| Modele | Val Accuracy | Val F1-Score | Val ROC-AUC |
|--------|-------------|-------------|------------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX |
| Gradient Boosting | 0.XXX | 0.XXX | 0.XXX |
| XGBoost | 0.XXX | 0.XXX | 0.XXX |
| LightGBM | 0.XXX | 0.XXX | 0.XXX |

### Variables les Plus Importantes

1. Age
2. NumOfProducts
3. IsActiveMember
4. Balance
5. Geography

## Methodologie

### 1. Analyse Exploratoire
- Distribution des variables
- Correlation avec le churn
- Detection des outliers
- Analyse des desequilibres

### 2. Feature Engineering
- `BalanceSalaryRatio`: Ratio solde/salaire
- `AgeGroup`: Groupes d'age
- `TenureGroup`: Groupes d'anciennete
- `CreditScoreGroup`: Groupes de score
- `IsZeroBalance`: Indicateur solde nul
- `HasMultipleProducts`: Indicateur multi-produits
- `EngagementScore`: Score d'engagement
- `Age_Balance_Interaction`: Interaction age-solde
- `TenureAgeRatio`: Ratio anciennete/age

### 3. Gestion du Desequilibre
- **SMOTE manuel**: Implementation complete sans dependances
- Random Oversampling
- Random Undersampling
- Comparaison des approches

### 4. Modelisation
- **Pipelines sklearn** pour reproductibilite
- Cross-validation stratifiee
- Optimisation hyperparametres (RandomizedSearchCV)
- Evaluation multi-metriques

## Technologies

- **Python 3.8+**
- **Pandas** - Manipulation de donnees
- **NumPy** - Calcul numerique
- **Scikit-learn** - Machine Learning
- **XGBoost, LightGBM, CatBoost** - Gradient Boosting
- **Matplotlib, Seaborn** - Visualisation
- **Jupyter** - Notebooks interactifs

## Points Techniques Importants

### Compatibilite NumPy 2.0

Le projet gere automatiquement les problemes de compatibilite entre NumPy 2.0 et certaines bibliotheques (Plotly, imbalanced-learn).

**Solutions implementees**:
- Detection automatique de la disponibilite de Plotly
- Implementation manuelle de SMOTE
- Fallback vers Matplotlib/Seaborn

### SMOTE Manuel

Implementation complete de SMOTE sans dependance a imbalanced-learn:
- Utilise KNN pour trouver les voisins
- Genere des echantillons synthetiques
- Garantit un equilibrage parfait

## Prochaines Etapes

- [ ] Tests unitaires complets
- [ ] API de prediction (FastAPI)
- [ ] Dashboard de monitoring (Streamlit)
- [ ] Containerisation (Docker)
- [ ] CI/CD (GitHub Actions)
- [ ] Documentation API
- [ ] Tracking MLflow

## Contributeurs

- Saer Ndao - Senior Data Scientist

## Licence

Ce projet est sous licence MIT.

## Contact

Pour toute question ou suggestion:
- Email: saerndao469@gmail.com
- LinkedIn: [Mon Profil](https://www.linkedin.com/in/saër-ndao-686208289/)
- GitHub: [Mon GitHub](https://github.com/ndaosaer)

---

**Note**: Ce projet a ete developpe dans le cadre d'une mission pour Fortuneo Banque visant a reduire le churn client grace au Machine Learning.