# Deploiement de l'API Churn Prediction

Guide complet pour deployer l'API de prediction de churn.

## Architecture

```
Client (Web/Mobile)
       |
       v
  FastAPI (Port 8000)
       |
       v
  Modele ML (Pickle)
       |
       v
  Predictions
```

## Installation

### 1. Pre-requis

```bash
Python 3.8+
pip
```

### 2. Installer les dependances

```bash
cd deployment/
pip install -r requirements.txt
```

### 3. Verifier la presence du modele

```bash
# Le modele doit etre dans models/saved_models/
ls ../models/saved_models/best_model_*.pkl
```

## Lancer l'API

### Mode Developpement

```bash
python app.py
```

Ou avec uvicorn directement:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Mode Production

```bash
uvicorn app:app --workers 4 --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

## Documentation Interactive

Une fois l'API lancee, accedez a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints Disponibles

### 1. Health Check

```bash
GET /health
```

Verifie que l'API fonctionne.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "features_count": 24
}
```

### 2. Prediction Simple

```bash
POST /predict
```

Predit le churn pour un client.

**Request Body**:
```json
{
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
```

**Response**:
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.23,
  "risk_level": "Low",
  "confidence": 0.54,
  "recommended_action": "Communication standard, newsletter mensuelle"
}
```

### 3. Prediction Batch

```bash
POST /predict/batch
```

Predit le churn pour plusieurs clients.

**Request Body**:
```json
{
  "customers": [
    {
      "CreditScore": 700,
      "Geography": "France",
      ...
    },
    {
      "CreditScore": 450,
      "Geography": "Spain",
      ...
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [...],
  "summary": {
    "total_customers": 2,
    "churn_predicted": 1,
    "churn_rate": 0.5,
    "avg_probability": 0.45,
    "risk_distribution": {
      "Low": 1,
      "High": 1
    }
  }
}
```

### 4. Informations Modele

```bash
GET /model/info
```

Retourne les informations sur le modele charge.

## Tests

### Script de test automatique

```bash
python test_api.py
```

### Tests manuels avec curl

```bash
# Health check
curl http://localhost:8000/health

# Prediction simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Tests avec Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

print(response.json())
```

## Niveaux de Risque

L'API categorise les clients selon leur probabilite de churn:

| Probabilite | Niveau | Action Recommandee |
|------------|--------|-------------------|
| < 30% | Low | Communication standard |
| 30-50% | Medium | Email personnalise |
| 50-70% | High | Appel telephonique + offre |
| > 70% | Critical | Action immediate manager |

## Monitoring

### Logs

Les logs de l'API sont affiches dans la console:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Modele charge: best_model_xgboost.pkl
INFO:     API prete!
```

### Metriques a Surveiller

- Temps de reponse moyen
- Nombre de predictions par heure
- Distribution des predictions (churn vs non-churn)
- Taux d'erreurs

## Deploiement Production

### Docker (Recommande)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Construire et lancer:

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

### Avec Gunicorn

```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Variables d'Environnement

```bash
export MODEL_PATH=/path/to/model.pkl
export PORT=8000
export LOG_LEVEL=info
```

## Securite

### Authentification (a implementer)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
```

### Rate Limiting (a implementer)

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("5/minute")
def predict(...):
    ...
```

## Troubleshooting

### Probleme: Modele non trouve

```
FileNotFoundError: Aucun modele trouve
```

**Solution**: Assurez-vous d'avoir entraine et sauvegarde le modele avec le notebook 03.

### Probleme: Port deja utilise

```
ERROR: Address already in use
```

**Solution**: Changez le port ou tuez le processus existant:

```bash
# Changer le port
uvicorn app:app --port 8001

# Ou tuer le processus
lsof -ti:8000 | xargs kill -9
```

### Probleme: Import error

```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution**: Installez les dependances:

```bash
pip install -r requirements.txt
```

## Performance

### Optimisations

1. **Batch Processing**: Utilisez `/predict/batch` pour plusieurs predictions
2. **Caching**: Implemente un cache pour les predictions frequentes
3. **Workers**: Augmentez le nombre de workers en production
4. **Compression**: Activez la compression gzip

### Benchmarks Typiques

- Prediction simple: ~50ms
- Batch 100 clients: ~200ms
- Throughput: ~500 predictions/seconde

## Contact & Support

Pour toute question ou probleme:
- Email: support@fortuneo-ml.fr
- Documentation: http://localhost:8000/docs
- Issues: GitHub Issues

---

**Version**: 1.0.0
**Derniere mise a jour**: Janvier 2026