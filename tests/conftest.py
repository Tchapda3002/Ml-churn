"""
Fixtures pytest pour les tests de l'API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le chemin racine au PYTHONPATH
ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))


@pytest.fixture(scope="module")
def client():
    """
    Client de test FastAPI.
    Note: Les modèles doivent être présents pour que l'API démarre.
    """
    try:
        from api.app.main import app
        with TestClient(app) as test_client:
            yield test_client
    except Exception as e:
        pytest.skip(f"Impossible de charger l'API: {e}")


@pytest.fixture
def sample_customer():
    """Données d'un client exemple pour les tests"""
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 125000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 85000.0
    }


@pytest.fixture
def sample_customer_high_risk():
    """Client à haut risque pour les tests"""
    return {
        "CreditScore": 400,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 65,
        "Tenure": 1,
        "Balance": 0.0,
        "NumOfProducts": 4,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 20000.0
    }


@pytest.fixture
def sample_customer_with_target():
    """Client avec variable cible pour validation"""
    return {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 125000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 85000.0,
        "Exited": 0
    }


@pytest.fixture
def batch_customers(sample_customer, sample_customer_high_risk):
    """Liste de clients pour les tests batch"""
    return [sample_customer, sample_customer_high_risk]
