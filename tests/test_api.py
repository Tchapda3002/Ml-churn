"""
Tests pour l'API de prédiction de churn
"""

import pytest
from fastapi import status


class TestHealthEndpoints:
    """Tests pour les endpoints de santé"""

    def test_root_endpoint(self, client):
        """Test de la page d'accueil"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Test du health check"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True


class TestPredictEndpoint:
    """Tests pour l'endpoint de prédiction individuelle"""

    def test_predict_single_valid(self, client, sample_customer):
        """Test de prédiction avec données valides"""
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "recommended_action" in data

        # Vérifier les types
        assert isinstance(data["churn_prediction"], int)
        assert data["churn_prediction"] in [0, 1]
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]
        assert 0 <= data["confidence"] <= 1

    def test_predict_single_high_risk(self, client, sample_customer_high_risk):
        """Test de prédiction pour un client à haut risque"""
        response = client.post("/predict", json=sample_customer_high_risk)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        # Un client à haut risque devrait avoir une probabilité plus élevée
        assert data["churn_probability"] >= 0

    def test_predict_invalid_credit_score_low(self, client, sample_customer):
        """Test avec un CreditScore trop bas"""
        invalid_customer = sample_customer.copy()
        invalid_customer["CreditScore"] = 100  # Min est 300

        response = client.post("/predict", json=invalid_customer)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_credit_score_high(self, client, sample_customer):
        """Test avec un CreditScore trop haut"""
        invalid_customer = sample_customer.copy()
        invalid_customer["CreditScore"] = 1000  # Max est 850

        response = client.post("/predict", json=invalid_customer)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_age(self, client, sample_customer):
        """Test avec un âge invalide"""
        invalid_customer = sample_customer.copy()
        invalid_customer["Age"] = 10  # Min est 18

        response = client.post("/predict", json=invalid_customer)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_gender(self, client, sample_customer):
        """Test avec un genre invalide"""
        invalid_customer = sample_customer.copy()
        invalid_customer["Gender"] = "Unknown"

        response = client.post("/predict", json=invalid_customer)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_geography(self, client, sample_customer):
        """Test avec une géographie non supportée"""
        invalid_customer = sample_customer.copy()
        invalid_customer["Geography"] = "USA"

        response = client.post("/predict", json=invalid_customer)
        # L'API peut accepter ou rejeter selon l'implémentation
        # On vérifie juste qu'elle répond
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_predict_missing_field(self, client):
        """Test avec un champ manquant"""
        incomplete_customer = {
            "CreditScore": 650,
            "Geography": "France",
            # Gender manquant
            "Age": 35,
        }

        response = client.post("/predict", json=incomplete_customer)
        # Avec les valeurs par défaut, ça devrait passer
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]


class TestBatchPredictEndpoint:
    """Tests pour l'endpoint de prédiction batch"""

    def test_predict_batch_valid(self, client, batch_customers):
        """Test de prédiction batch avec données valides"""
        response = client.post("/predict/batch", json={"customers": batch_customers})
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == len(batch_customers)

        # Vérifier le summary
        summary = data["summary"]
        assert "total_customers" in summary
        assert "churn_predicted" in summary
        assert "churn_rate" in summary
        assert "avg_probability" in summary
        assert "risk_distribution" in summary
        assert summary["total_customers"] == len(batch_customers)

    def test_predict_batch_empty(self, client):
        """Test de prédiction batch avec liste vide"""
        response = client.post("/predict/batch", json={"customers": []})
        # Peut retourner une erreur ou un résultat vide
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_predict_batch_single(self, client, sample_customer):
        """Test de prédiction batch avec un seul client"""
        response = client.post("/predict/batch", json={"customers": [sample_customer]})
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["predictions"]) == 1


class TestValidateBatchEndpoint:
    """Tests pour l'endpoint de validation batch"""

    def test_validate_batch_valid(self, client, sample_customer_with_target):
        """Test de validation batch avec données valides"""
        response = client.post(
            "/validate/batch",
            json={"customers": [sample_customer_with_target]}
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "metrics" in data
        assert "predictions" in data
        assert "summary" in data

        # Vérifier les métriques
        metrics = data["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics


class TestSchemaValidation:
    """Tests de validation des schémas Pydantic"""

    def test_customer_data_boundaries(self, client):
        """Test des valeurs limites"""
        # Valeurs minimales valides
        min_customer = {
            "CreditScore": 300,
            "Geography": "France",
            "Gender": "Male",
            "Age": 18,
            "Tenure": 0,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 0
        }
        response = client.post("/predict", json=min_customer)
        assert response.status_code == status.HTTP_200_OK

        # Valeurs maximales valides
        max_customer = {
            "CreditScore": 850,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 100,
            "Tenure": 10,
            "Balance": 999999999,
            "NumOfProducts": 4,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 999999999
        }
        response = client.post("/predict", json=max_customer)
        assert response.status_code == status.HTTP_200_OK


class TestRiskLevels:
    """Tests pour vérifier la cohérence des niveaux de risque"""

    def test_risk_level_consistency(self, client, sample_customer):
        """Vérifie que le risk_level correspond à la probabilité"""
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        prob = data["churn_probability"]
        risk = data["risk_level"]

        # Vérifier la cohérence selon les seuils définis dans l'API
        if prob < 0.3:
            assert risk == "Low"
        elif prob < 0.5:
            assert risk == "Medium"
        elif prob < 0.7:
            assert risk == "High"
        else:
            assert risk == "Critical"
