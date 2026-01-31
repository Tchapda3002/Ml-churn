fortuneo-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── churn_01_analyse_exploratoire.ipynb
│   ├── churn_02_preprocessing.ipynb
│   ├── churn_03_modelisation.ipynb
│   └── churn_04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── models/
│   └── saved_models/
├── config/
│   └── config.yaml
├── mlflow/
│   └── mlruns/
├── deployment/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
└── presentation/
    └── churn_presentation.pdf