import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from src.inference import load_model, predict


app = FastAPI(
    title="Breast Cancer MLP Classifier (PyTorch)",
    description=(
        "API для бинарной классификации опухолей молочной железы (benign / malignant) "
        "с использованием обученной MLP (PyTorch). "
        "Модель ожидает 30 числовых признаков, масштабирование проводится через "
        "StandardScaler. Для обучения использовался BCEWithLogitsLoss."
    ),
    version="1.0.0"
)
# -------------------------------
# Request Body Schema
# -------------------------------
class CancerFeatures(BaseModel):
    features: list[float] = Field(
        ..., 
        description="Список из 30 числовых признаков Breast Cancer dataset",
        example=[14.3, 21.5, 92.4, 600.1, 0.11, 0.15, 0.1, 0.08, 0.19, 0.06,
                 0.25, 1.4, 2.5, 18.0, 0.001, 0.01, 0.02, 0.005, 0.02, 0.003,
                 17.0, 30.0, 120.0, 850.0, 0.15, 0.25, 0.2, 0.12, 0.35, 0.08]
    )

# -------------------------
# Load Model Once on Startup
# -------------------------

model, scaler = load_model()
MODEL_VERSION = "v1.0.0"
START_TIME = datetime.now().isoformat()

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
def predict_cancer(data: CancerFeatures):
    # Log incoming request
    print(f"[{datetime.now().isoformat()}] Incoming request: {data.dict()}")

    # convert list → numpy row 
    sample = np.array(data.features).reshape(1, -1)
    
    return predict(model, scaler, sample)
