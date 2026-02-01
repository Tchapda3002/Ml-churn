# api/app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import logging
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import sys
from fastapi import HTTPException
import numpy as np
from .schemas import (
    CustomerData,
    CustomerDataWithTarget,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchValidationRequest
)
from .serializer import customer_to_df, customers_to_df, load_data

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fastapi import FastAPI, HTTPException, UploadFile, File  
import pandas as pd
import time
from datetime import datetime, timedelta
from fastapi.responses import FileResponse

# =========================
# CONFIGURATION PATH
# =========================
BASE_PATH = Path(__file__).parent.parent.parent
MODELS_DIR: Path = BASE_PATH / "models" 

# Dossier pour sauvegarder les fichiers
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas

# Seuil pour affichage inline
INLINE_THRESHOLD = 10  # Si <= 10 lignes → afficher dans JSON
                       # Si > 10 lignes → seulement fichier

# Durée de conservation des fichiers en minutes
MAX_FILE_AGE_MINUTES = 15 # Garder les fichiers 5 minutes

# Ajouter le chemin racine au sys.path pour pouvoir importer src
sys.path.append(str(BASE_PATH))
import src

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# FICHIERS DES PIPELINES ET MODÈLE
# =========================
PIPELINE_CLEANING: Path = MODELS_DIR / "pipeline_cleaning.pkl"
PIPELINE_FEATURES: Path = MODELS_DIR / "pipeline_features.pkl"
PIPELINE_ENCODING: Path = MODELS_DIR / "pipeline_encoding.pkl"
PIPELINE_SCALING: Path = MODELS_DIR / "pipeline_scaling.pkl"
SAVE_MODELS_DIR: Path = MODELS_DIR / "saved_models"
MODEL: Path = next(SAVE_MODELS_DIR.glob("best_model_*.pkl"), None)  

if MODEL is None:
    logger.error(f"Aucun modèle trouvé dans {SAVE_MODELS_DIR} correspondant à 'best_model_*.pkl'")
    sys.exit(1)

# =========================
# VARIABLES GLOBALES
# =========================
pipeline_cleaning = None
pipeline_features = None
pipeline_encoding = None
pipeline_scaling = None
model = None


def save_predictions_to_file(predictions: list, prefix: str = "predictions") -> dict:
    """
    Sauvegarde les predictions dans un fichier CSV
    
    Args:
        predictions: Liste de dictionnaires (les prédictions)
        prefix: Préfixe du nom de fichier ("predictions" ou "validation")
    
    Returns:
        dict avec infos sur le fichier créé
    """
    # 1. Générer un nom de fichier unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    filepath = OUTPUTS_DIR / filename
    
    # 2. Convertir la liste de dict en DataFrame
    df = pd.DataFrame(predictions)
    
    # 3. Sauvegarder en CSV
    df.to_csv(filepath, index=False)
    
    # 4. Récupérer la taille du fichier
    file_size = filepath.stat().st_size
    
    # 5. Retourner les infos
    return {
        "filename": filename,
        "path": str(filepath),
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "created_at": datetime.now().isoformat()
    }

def cleanup_old_files():
    """
    Supprime les fichiers de plus de MAX_FILE_AGE_HOURS heures
    
    Appelé au démarrage de l'API pour éviter l'accumulation de fichiers
    """
    now = datetime.now()
    deleted_count = 0
    total_size_deleted = 0
    
    # Parcourir tous les fichiers CSV dans outputs/
    for file in OUTPUTS_DIR.glob("*.csv"):
        # 1. Récupérer la date de création du fichier
        file_time = datetime.fromtimestamp(file.stat().st_mtime)
        
        # 2. Calculer l'âge du fichier
        age = now - file_time
        
        # 3. Si le fichier est trop vieux
        if age > timedelta(minutes=MAX_FILE_AGE_MINUTES):
            try:
                # Récupérer la taille avant suppression
                file_size = file.stat().st_size
                
                # Supprimer le fichier
                file.unlink()
                
                # Compter
                deleted_count += 1
                total_size_deleted += file_size
                
                logger.info(f"Fichier supprime: {file.name} (age: {age})")
                
            except Exception as e:
                logger.error(f"Erreur suppression {file.name}: {e}")
    
    # 4. Afficher un résumé
    if deleted_count > 0:
        size_mb = round(total_size_deleted / (1024 * 1024), 2)
        logger.info(f"Nettoyage termine: {deleted_count} fichiers supprimes ({size_mb} MB)")
    else:
        logger.info("Nettoyage: Aucun fichier à supprimer")

# =========================
# LIFESPAN DE L'API
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cette fonction s'exécute au démarrage et à l'arrêt de l'API
    """
    global pipeline_cleaning, pipeline_features, pipeline_encoding, pipeline_scaling, model

    logger.info("Démarrage de l'API...")
    logger.info("Nettoyage des vieux fichiers...")
    cleanup_old_files()

    try:
        # Charger chaque pipeline
        logger.info("Chargement des pipelines et du modèle...")

        pipeline_cleaning = joblib.load(PIPELINE_CLEANING)
        logger.info("Pipeline cleaning chargé")

        pipeline_features = joblib.load(PIPELINE_FEATURES)
        logger.info("Pipeline features chargé")

        pipeline_encoding = joblib.load(PIPELINE_ENCODING)
        logger.info("Pipeline encoding chargé")

        pipeline_scaling = joblib.load(PIPELINE_SCALING)
        logger.info("Pipeline scaling chargé")

        model = joblib.load(MODEL)
        logger.info(f"Modèle chargé depuis : {MODEL}")

        logger.info("Tous les modèles sont chargés et prêts !")

    except FileNotFoundError as fnf_error:
        logger.error(f"Fichier introuvable : {fnf_error}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles : {e}")
        sys.exit(1)

    yield  # L'API est maintenant opérationnelle

    logger.info("Arrêt de l'API...")

# =========================
# CREATION DE L'APPLICATION FASTAPI
# =========================
app = FastAPI(
    title="Exited Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# =========================
# MIDDLEWARE CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    """Page d'accueil de l'API"""
    return {"message": "Bienvenue sur l'API de prédiction de Exited"}

@app.get("/health")
def health_check():
    """Vérifie si l'API fonctionne et si les modèles sont chargés"""
    models_loaded = all([
        pipeline_cleaning is not None,
        pipeline_features is not None,
        pipeline_encoding is not None,
        pipeline_scaling is not None,
        model is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "version": "1.0.0",
        "models_loaded": models_loaded
    }

# --- utils.py ou en haut de main.py ---

def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque selon la probabilité de Exited"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.7:
        return "High"
    else:
        return "Critical"


def get_recommended_action(probability: float, risk_level: str) -> str:
    """Retourne l'action recommandée selon le niveau de risque"""
    actions = {
        "Low": "Communication standard, newsletter mensuelle",
        "Medium": "Email personnalisé avec avantages exclusifs",
        "High": "Appel téléphonique du conseiller + offre spéciale",
        "Critical": "Action immédiate - Contact manager + réduction tarifaire"
    }
    return actions.get(risk_level, "Monitoring régulier")



# ========== ENDPOINT PREDICT (UN CLIENT) ==========
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):  # ← CORRIGÉ : utilise CustomerData
    """
    Prédit le Exited pour UN client
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Convertir CustomerData en DataFrame
        df = customer_to_df(customer)
        
        # Pipelines
        df = pipeline_cleaning.transform(df)
        df = pipeline_features.transform(df)
        df = pipeline_encoding.transform(df)
        df = pipeline_scaling.transform(df)
        
        # Prédiction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        risk_level = get_risk_level(probability)
        confidence = abs(probability - 0.5) * 2
        recommended_action = get_recommended_action(probability, risk_level)

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            risk_level=risk_level,
            confidence=float(confidence),
            recommended_action=recommended_action
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# ========== ENDPOINT PREDICT BATCH ==========
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):  # ← CORRIGÉ
    """
    Prédit le churn pour PLUSIEURS clients
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Convertir liste de CustomerData en DataFrame
        df = customers_to_df(request.customers)
        
        # Pipelines
        df = pipeline_cleaning.transform(df)
        df = pipeline_features.transform(df)
        df = pipeline_encoding.transform(df)
        df = pipeline_scaling.transform(df)
        
        # Prédictions
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        
        predictions = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            risk_level = get_risk_level(prob)
            confidence = abs(prob - 0.5) * 2
            recommended_action = get_recommended_action(prob, risk_level)
            
            predictions.append(PredictionResponse(
                customer_id=f"customer_{i+1}",
                churn_prediction=int(pred),
                churn_probability=float(prob),
                risk_level=risk_level,
                confidence=float(confidence),
                recommended_action=recommended_action
            ))
        
        # Résumé
        total = len(predictions)
        churn_count = sum(p.churn_prediction for p in predictions)
        avg_prob = np.mean([p.churn_probability for p in predictions])
        
        risk_dist = {}
        for level in ["Low", "Medium", "High", "Critical"]:
            risk_dist[level] = sum(1 for p in predictions if p.risk_level == level)
        
        summary = {
            "total_customers": total,
            "churn_predicted": churn_count,
            "churn_rate": churn_count / total if total > 0 else 0,
            "avg_probability": float(avg_prob),
            "risk_distribution": risk_dist
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# ========== ENDPOINT VALIDATE BATCH ==========
def safe_metric(value):
    """
    Convertit NaN/Inf en valeur JSON valide
    """
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


@app.post("/validate/batch")
async def validate_batch(request: BatchValidationRequest):
    """
    Valide le modèle sur un batch de clients avec target
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Convertir en DataFrame
        df = customers_to_df(request.customers)
        
        # Vérifier que Churn existe
        if 'Exited' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"La colonne 'Exited' est manquante. Colonnes : {df.columns.tolist()}"
            )
        
        # Séparer target
        y_true = df['Exited'].values  # ← Convertir en numpy array
        X = df.drop(columns=['Exited'])
        
        logger.info(f"Validation de {len(y_true)} clients")
        logger.info(f"Distribution y_true: Classe 0={sum(y_true==0)}, Classe 1={sum(y_true==1)}")
        
        # Pipelines
        X = pipeline_cleaning.transform(X)
        X = pipeline_features.transform(X)
        X = pipeline_encoding.transform(X)
        X = pipeline_scaling.transform(X)
        
        # Prédictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        logger.info(f"Distribution y_pred: Classe 0={sum(y_pred==0)}, Classe 1={sum(y_pred==1)}")
        
        # Calculer métriques avec gestion des NaN
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)  # ← zero_division=0
            recall = recall_score(y_true, y_pred, zero_division=0)        # ← zero_division=0
            f1 = f1_score(y_true, y_pred, zero_division=0)                # ← zero_division=0
            
            # ROC-AUC peut échouer si une seule classe
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"ROC-AUC non calculable: {e}")
                roc_auc = 0.0
        
        except Exception as e:
            logger.error(f"Erreur calcul métriques: {e}")
            accuracy = precision = recall = f1 = roc_auc = 0.0
        
        # Convertir en float sûr (remplace NaN par 0)
        metrics = {
            "accuracy": safe_metric(accuracy),
            "precision": safe_metric(precision),
            "recall": safe_metric(recall),
            "f1_score": safe_metric(f1),
            "roc_auc": safe_metric(roc_auc)
        }
        
        # Détails par client
        predictions = []
        for i, (pred, prob) in enumerate(zip(y_pred, y_proba)):
            risk_level = get_risk_level(prob)
            confidence = abs(prob - 0.5) * 2
            
            predictions.append({
                "customer_id": str(i + 1),
                "y_true": int(y_true[i]),
                "y_pred": int(pred),
                "probability": safe_metric(prob),        # ← Sécurisé
                "confidence": safe_metric(confidence),    # ← Sécurisé
                "risk_level": risk_level,
                "correct": bool(y_true[i] == pred)
            })
        
        logger.info(f"✅ Validation réussie: Accuracy={metrics['accuracy']:.4f}")
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "summary": {
                "total": len(y_true),
                "correct": int(sum(y_true == y_pred)),
                "incorrect": int(sum(y_true != y_pred)),
                "accuracy_pct": f"{metrics['accuracy']*100:.2f}%"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
# api/app/main.py

@app.post("/predict/file", tags=["Prediction"])
async def predict_file(file: UploadFile = File(...)):
    """
    Predire le Exited avec un fichier CSV/Excel
    
    - Si <= 10 lignes : Renvoie predictions + summary + fichier
    - Si > 10 lignes : Renvoie summary + fichier uniquement
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        start_total = time.time()
        
        logger.info(f"Prediction avec fichier: {file.filename}")
        
        # 1. LECTURE DU FICHIER
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Format non supporte")

        row_count = len(df)
        logger.info(f"Fichier charge: {row_count} lignes")

        # 2. VERIFICATIONS
        if 'Exited' in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Le fichier ne doit PAS contenir 'Exited'. Utilisez /validate"
            )

        from .serializer import REQUIRED_COLUMNS
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing_cols}")

        X = df[REQUIRED_COLUMNS]

        # 3. PIPELINES
        logger.info("Application des pipelines...")
        X = pipeline_cleaning.transform(X)
        X = pipeline_features.transform(X)
        X = pipeline_encoding.transform(X)
        X = pipeline_scaling.transform(X)

        # 4. PREDICTIONS
        logger.info("Predictions...")
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # 5. CREER LES RESULTATS
        predictions = []
        for i, (pred, prob) in enumerate(zip(y_pred, y_proba)):
            risk_level = get_risk_level(prob)
            confidence = abs(prob - 0.5) * 2
            recommended_action = get_recommended_action(prob, risk_level)
            
            predictions.append({
                "customer_id": str(i + 1),
                "churn_prediction": int(pred),
                "churn_probability": safe_metric(prob),
                "risk_level": risk_level,
                "confidence": safe_metric(confidence),
                "recommended_action": recommended_action
            })

        # 6. CALCULER LE SUMMARY
        total = len(predictions)
        churn_count = sum(p["churn_prediction"] for p in predictions)
        avg_prob = np.mean([p["churn_probability"] for p in predictions])
        
        risk_dist = {}
        for level in ["Low", "Medium", "High", "Critical"]:
            risk_dist[level] = sum(1 for p in predictions if p["risk_level"] == level)

        summary = {
            "total_customers": total,
            "churn_predicted": churn_count,
            "churn_rate": round(churn_count / total if total > 0 else 0, 4),
            "avg_probability": round(float(avg_prob), 4),
            "risk_distribution": risk_dist
        }

        # 7. SAUVEGARDER LE FICHIER (TOUJOURS)
        file_info = save_predictions_to_file(predictions, prefix="predictions")
        
        processing_time = round(time.time() - start_total, 2)
        logger.info(f"Predictions terminees en {processing_time}s")

        # 8. DECISION : Inline ou File only
        if row_count <= INLINE_THRESHOLD:
            # CAS 1 : Peu de lignes → Afficher tout
            logger.info(f"Mode INLINE ({row_count} lignes)")
            return {
                "status": "completed",
                "mode": "inline",
                "row_count": row_count,
                "predictions": predictions,
                "summary": summary,
                "file": file_info,
                "processing_time_seconds": processing_time
            }
        else:
            # CAS 2 : Beaucoup de lignes → Seulement summary + fichier
            logger.info(f"Mode FILE_ONLY ({row_count} lignes)")
            return {
                "status": "completed",
                "mode": "file_only",
                "row_count": row_count,
                "summary": summary,
                "file": file_info,
                "message": f"Trop de resultats ({row_count} lignes). Telechargez le fichier.",
                "processing_time_seconds": processing_time
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
# api/app/main.py

@app.post("/validate/file", tags=["Validation"])
async def validate_file(file: UploadFile = File(...)):
    """
    Valider le modele avec un fichier CSV/Excel
    
    - Si <= 10 lignes : Renvoie predictions + metriques + fichier
    - Si > 10 lignes : Renvoie metriques + fichier uniquement
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        start_total = time.time()
        
        logger.info(f"Validation avec fichier: {file.filename}")
        
        # 1. LECTURE DU FICHIER
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Format non supporte")

        row_count = len(df)
        logger.info(f"Fichier charge: {row_count} lignes")

        # 2. VERIFICATIONS
        if 'Exited' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Colonne 'Exited' manquante. Utilisez /predict/file pour prediction"
            )

        # 3. SEPARER X et y
        y_true = df['Exited'].values
        X = df.drop(columns=['Exited'])

        # 4. PIPELINES
        X = pipeline_cleaning.transform(X)
        X = pipeline_features.transform(X)
        X = pipeline_encoding.transform(X)
        X = pipeline_scaling.transform(X)

        # 5. PREDICTIONS
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # 6. CALCULER METRIQUES
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                logger.warning("ROC-AUC non calculable")
                roc_auc = 0.0
        except Exception:
            accuracy = precision = recall = f1 = roc_auc = 0.0

        metrics = {
            "accuracy": safe_metric(accuracy),
            "precision": safe_metric(precision),
            "recall": safe_metric(recall),
            "f1_score": safe_metric(f1),
            "roc_auc": safe_metric(roc_auc)
        }

        # 7. DETAILS PAR CLIENT
        predictions = []
        for i, (true_val, pred, prob) in enumerate(zip(y_true, y_pred, y_proba)):
            risk_level = get_risk_level(prob)
            confidence = abs(prob - 0.5) * 2
            
            predictions.append({
                "customer_id": str(i + 1),
                "y_true": int(true_val),
                "y_pred": int(pred),
                "probability": safe_metric(prob),
                "confidence": safe_metric(confidence),
                "risk_level": risk_level,
                "correct": bool(true_val == pred)
            })

        # 8. SUMMARY
        summary = {
            "total": len(y_true),
            "correct": int(sum(y_true == y_pred)),
            "incorrect": int(sum(y_true != y_pred)),
            "accuracy_pct": f"{metrics['accuracy']*100:.2f}%"
        }

        # 9. SAUVEGARDER LE FICHIER (TOUJOURS)
        file_info = save_predictions_to_file(predictions, prefix="validation")
        
        processing_time = round(time.time() - start_total, 2)
        logger.info(f"Validation terminee en {processing_time}s")

        # 10. DECISION : Inline ou File only
        if row_count <= INLINE_THRESHOLD:
            # CAS 1 : Peu de lignes → Afficher tout
            logger.info(f"Mode INLINE ({row_count} lignes)")
            return {
                "status": "completed",
                "mode": "inline",
                "row_count": row_count,
                "metrics": metrics,
                "predictions": predictions,
                "summary": summary,
                "file": file_info,
                "processing_time_seconds": processing_time
            }
        else:
            # CAS 2 : Beaucoup de lignes → Seulement metriques + fichier
            logger.info(f"Mode FILE_ONLY ({row_count} lignes)")
            return {
                "status": "completed",
                "mode": "file_only",
                "row_count": row_count,
                "metrics": metrics,
                "summary": summary,
                "file": file_info,
                "message": f"Trop de resultats ({row_count} lignes). Telechargez le fichier.",
                "processing_time_seconds": processing_time
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    

@app.get("/downloads/{filename}", tags=["Downloads"])
async def download_file(filename: str):
    """
    Telecharger un fichier de resultats genere
    
    Usage:
    - Apres /predict/file, recuperer le fichier predictions_XXX.csv
    - Apres /validate, recuperer le fichier validation_XXX.csv
    """
    # 1. Construire le chemin complet du fichier
    filepath = OUTPUTS_DIR / filename
    
    # 2. Verifier que le fichier existe
    if not filepath.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Fichier '{filename}' introuvable"
        )
    
    # 3. Verifier que c'est bien un fichier CSV (securite)
    if not filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers CSV peuvent etre telecharges"
        )
    
    # 4. Renvoyer le fichier
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="text/csv"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
