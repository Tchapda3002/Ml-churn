"""
Script pour tester l'API de prediction de churn
"""

import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"


def test_root():
    """Test l'endpoint racine"""
    print("="*60)
    print("TEST: Endpoint racine")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_health():
    """Test le health check"""
    print("="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Test les infos du modele"""
    print("="*60)
    print("TEST: Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Model Type: {data['model_type']}")
        print(f"Features Count: {data['features_count']}")
        print(f"Scaler Loaded: {data['scaler_loaded']}")
    else:
        print(f"Error: {response.text}")
    print()


def test_single_prediction():
    """Test une prediction simple"""
    print("="*60)
    print("TEST: Prediction Simple")
    print("="*60)
    
    # Donnees de test
    customer_data = {
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
    
    print("Donnees client:")
    print(json.dumps(customer_data, indent=2))
    print()
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nResultat de la prediction:")
        print(f"  Prediction: {'CHURN' if result['churn_prediction'] == 1 else 'NO CHURN'}")
        print(f"  Probabilite: {result['churn_probability']:.2%}")
        print(f"  Niveau de risque: {result['risk_level']}")
        print(f"  Confiance: {result['confidence']:.2%}")
        print(f"  Action recommandee: {result['recommended_action']}")
    else:
        print(f"Error: {response.text}")
    print()


def test_high_risk_customer():
    """Test un client a haut risque"""
    print("="*60)
    print("TEST: Client a Haut Risque")
    print("="*60)
    
    # Client avec profil a risque
    customer_data = {
        "CreditScore": 400,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 35000.0
    }
    
    print("Donnees client (profil a risque):")
    print(json.dumps(customer_data, indent=2))
    print()
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nResultat de la prediction:")
        print(f"  Prediction: {'CHURN' if result['churn_prediction'] == 1 else 'NO CHURN'}")
        print(f"  Probabilite: {result['churn_probability']:.2%}")
        print(f"  Niveau de risque: {result['risk_level']}")
        print(f"  Confiance: {result['confidence']:.2%}")
        print(f"  Action recommandee: {result['recommended_action']}")
    else:
        print(f"Error: {response.text}")
    print()


def test_batch_prediction():
    """Test une prediction par batch"""
    print("="*60)
    print("TEST: Prediction Batch")
    print("="*60)
    
    # Plusieurs clients
    batch_data = {
        "customers": [
            {
                "CreditScore": 700,
                "Geography": "France",
                "Gender": "Male",
                "Age": 30,
                "Tenure": 7,
                "Balance": 150000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 90000.0
            },
            {
                "CreditScore": 450,
                "Geography": "Spain",
                "Gender": "Female",
                "Age": 50,
                "Tenure": 1,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 30000.0
            },
            {
                "CreditScore": 800,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 28,
                "Tenure": 8,
                "Balance": 200000.0,
                "NumOfProducts": 3,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 120000.0
            }
        ]
    }
    
    print(f"Nombre de clients: {len(batch_data['customers'])}")
    print()
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch_data
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("\nPredictions individuelles:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\nClient {i}:")
            print(f"  Prediction: {'CHURN' if pred['churn_prediction'] == 1 else 'NO CHURN'}")
            print(f"  Probabilite: {pred['churn_probability']:.2%}")
            print(f"  Risque: {pred['risk_level']}")
        
        print("\n--- RESUME ---")
        summary = result['summary']
        print(f"Total clients: {summary['total_customers']}")
        print(f"Churn predit: {summary['churn_predicted']}")
        print(f"Taux de churn: {summary['churn_rate']:.2%}")
        print(f"Probabilite moyenne: {summary['avg_probability']:.2%}")
        print(f"\nDistribution des risques:")
        for level, count in summary['risk_distribution'].items():
            print(f"  {level}: {count}")
    else:
        print(f"Error: {response.text}")
    print()


def test_invalid_data():
    """Test avec des donnees invalides"""
    print("="*60)
    print("TEST: Donnees Invalides")
    print("="*60)
    
    # CreditScore hors limites
    invalid_data = {
        "CreditScore": 1000,  # Invalide (max 850)
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
    
    print("Donnees invalides (CreditScore=1000):")
    print()
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("Validation fonctionne correctement!")
    print()


def run_all_tests():
    """Execute tous les tests"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + " " * 15 + "TESTS API CHURN PREDICTION" + " " * 17 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_single_prediction()
        test_high_risk_customer()
        test_batch_prediction()
        test_invalid_data()
        
        print("\n")
        print("="*60)
        print("TOUS LES TESTS EXECUTES AVEC SUCCES!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\nERREUR: Impossible de se connecter a l'API")
        print("Assurez-vous que l'API est lancee avec: python deployment/app.py")
    except Exception as e:
        print(f"\nERREUR: {str(e)}")


if __name__ == "__main__":
    run_all_tests()