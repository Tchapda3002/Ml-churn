"""
API FastAPI pour predictions de churn
Fortuneo Banque - Churn Prediction Service
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser l'application
app = FastAPI(
    title="Churn Prediction API",
    description="API de prediction du churn client pour Fortuneo Banque",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Chemins
BASE_PATH = Path(__file__).parent.parent
MODELS_PATH = BASE_PATH / 'models' / 'saved_models'

# Variable globale pour le modele
model = None
scaler = None
feature_names = None


class CustomerData(BaseModel):
    """Schema pour les donnees client"""
    CreditScore: int = Field(..., ge=300, le=850, description="Score de credit (300-850)")
    Geography: str = Field(..., description="Pays (France, Spain, Germany)")
    Gender: str = Field(..., description="Genre (Male, Female)")
    Age: int = Field(..., ge=18, le=100, description="Age (18-100)")
    Tenure: int = Field(..., ge=0, le=10, description="Anciennete en annees (0-10)")
    Balance: float = Field(..., ge=0, description="Solde du compte")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Nombre de produits (1-4)")
    HasCrCard: int = Field(..., ge=0, le=1, description="Possession carte (0 ou 1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Membre actif (0 ou 1)")
    EstimatedSalary: float = Field(..., ge=0, description="Salaire estime")
    
    @validator('Geography')
    def validate_geography(cls, v):
        valid_countries = ['France', 'Spain', 'Germany']
        if v not in valid_countries:
            raise ValueError(f'Geography doit etre parmi {valid_countries}')
        return v
    
    @validator('Gender')
    def validate_gender(cls, v):
        valid_genders = ['Male', 'Female']
        if v not in valid_genders:
            raise ValueError(f'Gender doit etre parmi {valid_genders}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Schema pour la reponse de prediction"""
    customer_id: Optional[str] = None
    churn_prediction: int = Field(..., description="Prediction (0=Reste, 1=Parti)")
    churn_probability: float = Field(..., description="Probabilite de churn (0-1)")
    risk_level: str = Field(..., description="Niveau de risque (Low, Medium, High, Critical)")
    confidence: float = Field(..., description="Confiance de la prediction")
    recommended_action: str = Field(..., description="Action recommandee")


class BatchPredictionRequest(BaseModel):
    """Schema pour les predictions par batch"""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Schema pour la reponse batch"""
    predictions: List[PredictionResponse]
    summary: dict


def load_model_artifacts():
    """Charge le modele et les artefacts necessaires"""
    global model, scaler, feature_names
    
    try:
        # Charger le modele
        model_files = list(MODELS_PATH.glob('best_model_*.pkl'))
        if not model_files:
            raise FileNotFoundError("Aucun modele trouve")
        
        model_path = model_files[0]
        model = joblib.load(model_path)
        logger.info(f"Modele charge: {model_path}")
        
        # Charger le scaler
        scaler_path = BASE_PATH / 'models' / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Scaler charge")
        
        # Charger les noms de features
        processed_path = BASE_PATH / 'data' / 'processed'
        if (processed_path / 'X_train.csv').exists():
            sample_data = pd.read_csv(processed_path / 'X_train.csv', nrows=1)
            feature_names = sample_data.columns.tolist()
            logger.info(f"{len(feature_names)} features chargees")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modele: {str(e)}")
        return False


def preprocess_customer_data(customer: CustomerData) -> pd.DataFrame:
    """
    Preprocess les donnees client pour prediction
    
    Args:
        customer: Donnees client
        
    Returns:
        DataFrame preprocesse
    """
    # Convertir en DataFrame
    data = pd.DataFrame([customer.dict()])
    
    # Feature Engineering (reprendre les features du notebook 02)
    
    # 1. BalanceSalaryRatio
    data['BalanceSalaryRatio'] = data['Balance'] / (data['EstimatedSalary'] + 1)
    
    # 2. AgeGroup
    data['AgeGroup'] = pd.cut(data['Age'], 
                              bins=[0, 30, 40, 50, 100],
                              labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    # 3. TenureGroup
    data['TenureGroup'] = pd.cut(data['Tenure'],
                                 bins=[-1, 2, 5, 10],
                                 labels=['New', 'Regular', 'Loyal'])
    
    # 4. CreditScoreGroup
    data['CreditScoreGroup'] = pd.cut(data['CreditScore'],
                                      bins=[0, 600, 700, 850],
                                      labels=['Poor', 'Good', 'Excellent'])
    
    # 5. IsZeroBalance
    data['IsZeroBalance'] = (data['Balance'] == 0).astype(int)
    
    # 6. HasMultipleProducts
    data['HasMultipleProducts'] = (data['NumOfProducts'] > 1).astype(int)
    
    # 7. EngagementScore
    data['EngagementScore'] = (
        data['IsActiveMember'] + 
        data['HasCrCard'] + 
        (data['NumOfProducts'] / 4)
    )
    
    # 8. Age_Balance_Interaction
    data['Age_Balance_Interaction'] = data['Age'] * data['Balance'] / 100000
    
    # 9. TenureAgeRatio
    data['TenureAgeRatio'] = data['Tenure'] / data['Age']
    
    # Encodage des variables categorielles
    # Geography
    if 'Geography' in data.columns:
        geography_dummies = pd.get_dummies(data['Geography'], prefix='Geography', drop_first=True)
        data = pd.concat([data, geography_dummies], axis=1)
        data = data.drop(columns=['Geography'])
    
    # Gender
    if 'Gender' in data.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        data['Gender'] = le.fit_transform(data['Gender'])
    
    # AgeGroup, TenureGroup, CreditScoreGroup
    for col in ['AgeGroup', 'TenureGroup', 'CreditScoreGroup']:
        if col in data.columns:
            col_dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, col_dummies], axis=1)
            data = data.drop(columns=[col])
    
    # S'assurer que toutes les features necessaires sont presentes
    if feature_names is not None:
        for col in feature_names:
            if col not in data.columns:
                data[col] = 0
        
        # Reordonner les colonnes
        data = data[feature_names]
    
    return data


def get_risk_level(probability: float) -> str:
    """Determine le niveau de risque selon la probabilite"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.7:
        return "High"
    else:
        return "Critical"


def get_recommended_action(probability: float, risk_level: str) -> str:
    """Recommande une action selon le niveau de risque"""
    actions = {
        "Low": "Communication standard, newsletter mensuelle",
        "Medium": "Email personnalise avec avantages exclusifs",
        "High": "Appel telephonique du conseiller + offre speciale",
        "Critical": "Action immediate - Contact manager + reduction tarifaire"
    }
    return actions.get(risk_level, "Monitoring regulier")


@app.on_event("startup")
async def startup_event():
    """Charge le modele au demarrage"""
    logger.info("Demarrage de l'API...")
    success = load_model_artifacts()
    if not success:
        logger.error("ERREUR: Impossible de charger le modele")
    else:
        logger.info("API prete!")


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Churn Prediction API - Fortuneo Banque",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_count": len(feature_names) if feature_names else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predit le churn pour un client
    
    Args:
        customer: Donnees du client
        
    Returns:
        Prediction avec probabilite et recommandations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    
    try:
        # Preprocessing
        features = preprocess_customer_data(customer)
        
        # Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Determiner le niveau de risque
        risk_level = get_risk_level(probability)
        
        # Confiance (distance au seuil)
        confidence = abs(probability - 0.5) * 2
        
        # Action recommandee
        recommended_action = get_recommended_action(probability, risk_level)
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            risk_level=risk_level,
            confidence=float(confidence),
            recommended_action=recommended_action
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prediction: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predit le churn pour plusieurs clients
    
    Args:
        request: Liste de clients
        
    Returns:
        Predictions pour tous les clients avec resume
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    
    try:
        predictions = []
        
        for i, customer in enumerate(request.customers):
            # Preprocessing
            features = preprocess_customer_data(customer)
            
            # Prediction
            prediction = model.predict(features)[0]
            
            probability = model.predict_proba(features)[0][1]

            risk_level = get_risk_level(probability)
            confidence = abs(probability - 0.5) * 2
            recommended_action = get_recommended_action(probability, risk_level)
            
            predictions.append(PredictionResponse(
                customer_id=f"customer_{i+1}",
                churn_prediction=int(prediction),
                churn_probability=float(probability),
                risk_level=risk_level,
                confidence=float(confidence),
                recommended_action=recommended_action
            ))
        
        # Resume
        total_customers = len(predictions)
        churn_predicted = sum(1 for p in predictions if p.churn_prediction == 1)
        avg_probability = np.mean([p.churn_probability for p in predictions])
        
        risk_distribution = {}
        for level in ["Low", "Medium", "High", "Critical"]:
            count = sum(1 for p in predictions if p.risk_level == level)
            risk_distribution[level] = count
        
        summary = {
            "total_customers": total_customers,
            "churn_predicted": churn_predicted,
            "churn_rate": churn_predicted / total_customers if total_customers > 0 else 0,
            "avg_probability": float(avg_probability),
            "risk_distribution": risk_distribution
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prediction batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prediction: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Retourne les informations sur le modele"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    
    return {
        "model_type": type(model).__name__,
        "features_count": len(feature_names) if feature_names else 0,
        "features": feature_names if feature_names else [],
        "scaler_loaded": scaler is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)