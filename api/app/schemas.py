# api/app/schemas.py


from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    """Une ligne client sans variable cible (pour /predict)"""

    CreditScore: int = Field(
        default=650,  # ← AJOUT : valeur par défaut
        ge=300,
        le=850,
        example=650,  # ← CORRIGÉ : example au lieu de examples
        description="Score de crédit (300-850)",
    )

    Geography: str = Field(
        default="France",  # ← AJOUT
        example="France",  # ← CORRIGÉ
        description="Pays (France, Spain, Germany)",
    )

    Gender: str = Field(
        default="Male",  # ← AJOUT
        example="Male",  # ← CORRIGÉ
        description="Genre (Male, Female)",
    )

    Age: int = Field(
        default=35,  # ← AJOUT
        ge=18,
        le=100,
        example=35,  # ← CORRIGÉ
        description="Âge du client",
    )

    Tenure: int = Field(
        default=5,  # ← AJOUT
        ge=0,
        le=10,
        example=5,  # ← CORRIGÉ
        description="Ancienneté en années",
    )

    Balance: float = Field(
        default=125000.0,  # ← AJOUT
        ge=0,
        example=125000.0,  # ← CORRIGÉ
        description="Solde du compte",
    )

    NumOfProducts: int = Field(
        default=2,  # ← AJOUT
        ge=1,
        le=4,
        example=2,  # ← CORRIGÉ
        description="Nombre de produits (1-4)",
    )

    HasCrCard: int = Field(
        default=1,  # ← AJOUT
        ge=0,
        le=1,
        example=1,  # ← CORRIGÉ
        description="Carte de crédit (0 ou 1)",
    )

    IsActiveMember: int = Field(
        default=1,  # ← AJOUT
        ge=0,
        le=1,
        example=1,  # ← CORRIGÉ
        description="Membre actif (0 ou 1)",
    )

    EstimatedSalary: float = Field(
        default=85000.0,  # ← AJOUT
        ge=0,
        example=85000.0,  # ← CORRIGÉ
        description="Salaire estimé",
    )

    @field_validator("Gender")  # ← CORRIGÉ : field_validator au lieu de validator
    @classmethod
    def validate_gender(cls, v):
        valid_genders = ["Male", "Female"]
        if v not in valid_genders:
            raise ValueError(f"Gender doit être parmi {valid_genders}")
        return v


class CustomerDataWithTarget(CustomerData):
    """Une ligne client avec variable cible (pour /validate)"""

    Exited: int = Field(
        ...,
        ge=0,
        le=1,
        example=1,  # ← CORRIGÉ
        description="Variable cible : Exited (0=reste, 1=parti)",
    )


class PredictionResponse(BaseModel):
    """Résultat d'une prédiction pour un client"""

    customer_id: str | None = None
    churn_prediction: int = Field(..., description="Prediction (0=Reste, 1=Parti)")
    churn_probability: float = Field(..., description="Probabilité de Exited (0-1)")
    risk_level: str = Field(..., description="Niveau de risque (Low, Medium, High, Critical)")
    confidence: float = Field(..., description="Confiance de la prédiction")
    recommended_action: str = Field(..., description="Action recommandée")


class BatchPredictionRequest(BaseModel):
    """Liste de clients pour prédiction"""

    customers: list[CustomerData]

    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    # Client 1 : Profil stable, risque faible
                    {
                        "CreditScore": 720,
                        "Geography": "France",
                        "Gender": "Male",
                        "Age": 35,
                        "Tenure": 8,
                        "Balance": 150000.0,
                        "NumOfProducts": 2,
                        "HasCrCard": 1,
                        "IsActiveMember": 1,
                        "EstimatedSalary": 85000.0,
                    },
                    # Client 2 : Profil moyen, risque moyen
                    {
                        "CreditScore": 650,
                        "Geography": "Germany",
                        "Gender": "Female",
                        "Age": 42,
                        "Tenure": 3,
                        "Balance": 80000.0,
                        "NumOfProducts": 1,
                        "HasCrCard": 1,
                        "IsActiveMember": 0,
                        "EstimatedSalary": 60000.0,
                    },
                    # Client 3 : Profil à risque, risque élevé
                    {
                        "CreditScore": 450,
                        "Geography": "Spain",
                        "Gender": "Female",
                        "Age": 68,
                        "Tenure": 1,
                        "Balance": 0.0,
                        "NumOfProducts": 1,
                        "HasCrCard": 0,
                        "IsActiveMember": 0,
                        "EstimatedSalary": 25000.0,
                    },
                ]
            }
        }


# ----------------------------
# Requête batch VALIDATION
# ----------------------------
class BatchValidationRequest(BaseModel):
    """Liste de clients avec target pour validation"""

    customers: list[CustomerDataWithTarget]

    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    # Client 1 : Reste (Exited = 0) - Profil stable
                    {
                        "CreditScore": 720,
                        "Geography": "France",
                        "Gender": "Male",
                        "Age": 35,
                        "Tenure": 8,
                        "Balance": 150000.0,
                        "NumOfProducts": 2,
                        "HasCrCard": 1,
                        "IsActiveMember": 1,
                        "EstimatedSalary": 85000.0,
                        "Exited": 0,  # ← Reste
                    },
                    # Client 2 : Reste (Exited = 0) - Profil moyen mais reste
                    {
                        "CreditScore": 650,
                        "Geography": "Germany",
                        "Gender": "Female",
                        "Age": 42,
                        "Tenure": 3,
                        "Balance": 80000.0,
                        "NumOfProducts": 1,
                        "HasCrCard": 1,
                        "IsActiveMember": 0,
                        "EstimatedSalary": 60000.0,
                        "Exited": 0,  # ← Reste
                    },
                    # Client 3 : Part (Exited = 1) - Profil à risque
                    {
                        "CreditScore": 450,
                        "Geography": "Spain",
                        "Gender": "Female",
                        "Age": 68,
                        "Tenure": 1,
                        "Balance": 0.0,
                        "NumOfProducts": 1,
                        "HasCrCard": 0,
                        "IsActiveMember": 0,
                        "EstimatedSalary": 25000.0,
                        "Exited": 1,  # ← Part
                    },
                ]
            }
        }


# ----------------------------
# Réponse batch
# ----------------------------
class BatchPredictionResponse(BaseModel):
    """Résultats de prédiction pour un batch"""

    predictions: list[PredictionResponse]
    summary: dict
