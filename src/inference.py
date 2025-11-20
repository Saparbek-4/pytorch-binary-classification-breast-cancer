import torch
import joblib
import numpy as np
from src.model import BinaryClassifier

def load_model(model_path="saved_models/model.pth", scaler_path="saved_models/scaler.pkl"):
    # Load scaler
    scaler = joblib.load(scaler_path)

    # Init model
    model = BinaryClassifier(in_features=scaler.mean_.shape[0])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler


def predict(model, scaler, sample: np.ndarray):
    sample = scaler.transform(sample)

    tensor = torch.tensor(sample, dtype=torch.float32)
    
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        pred = int(prob > 0.5)

    return {"logit": logit.item(), "prob": prob, "class": pred}

