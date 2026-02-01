# api/app/serializer.py

from typing import Union, List
import pandas as pd
from fastapi import UploadFile, HTTPException
import io
import logging
from .schemas import CustomerData, CustomerDataWithTarget  # ← AJOUT

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]


def customer_to_df(customer: CustomerData) -> pd.DataFrame:
    """
    Convertit un objet CustomerData en DataFrame
    """
    data = customer.model_dump(exclude_none=True)
    return pd.DataFrame([data])


def customers_to_df(customers: List[CustomerData]) -> pd.DataFrame:
    """
    Convertit une liste de CustomerData en DataFrame
    """
    data = [customer.model_dump(exclude_none=True) for customer in customers]
    return pd.DataFrame(data)


def load_data(input_data: Union[pd.DataFrame, UploadFile, list, dict]) -> pd.DataFrame:
    """
    Convertit les données d'entrée en DataFrame prête pour le pipeline.
    """
    df = None

    try:
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        
        elif isinstance(input_data, (list, dict)):
            df = pd.DataFrame(input_data if isinstance(input_data, list) else [input_data])
        
        elif isinstance(input_data, UploadFile):
            content = input_data.file.read()
            try:
                if input_data.filename.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(content))
                elif input_data.filename.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(io.BytesIO(content))
                else:
                    raise HTTPException(status_code=400, detail="Format non supporté")
            finally:
                input_data.file.close()
        
        else:
            raise HTTPException(status_code=400, detail="Type de données non supporté")

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing_cols}")
        
        df = df[REQUIRED_COLUMNS]

        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Dataset vide")
        
        return df

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lecture données: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")