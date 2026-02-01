FROM python:3.13-slim

WORKDIR /app

# Dépendances système si besoin


# Copier requirements
COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY api ./api
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
