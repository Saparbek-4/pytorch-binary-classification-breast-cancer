# 🧬 Breast Cancer Binary Classification
### PyTorch + FastAPI | End-to-End ML Engineering Project

A production-ready machine learning system for **binary cancer diagnosis** (benign vs. malignant) based on the **Breast Cancer Wisconsin (Diagnostic)** dataset.  
The project implements a full ML workflow:

- custom PyTorch MLP model  
- training loop with metrics & visualizations  
- serialized model & scaler  
- FastAPI inference service  
- clean, modular architecture

This repository demonstrates **strong ML engineering skills** and proper separation between training and production inference.

---

## 📁 Project Structure

```
binary_classification_project/
│
├── notebooks/
│   ├── training.ipynb          
│   └── eda.ipynb               
│
├── saved_models/
│   ├── model.pth               
│   └── scaler.pkl              
│
├── src/
│   ├── dataset.py              
│   ├── model.py                
│   ├── train.py                
│   ├── metrics.py              
│   └── inference.py            
│
├── visuals/                    
│   ├── loss.png
│   └── confusion_matrix.png
│
├── app.py                      
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Create a virtual environment

```bash
conda create -n bcancer python=3.10
conda activate bcancer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎓 Training the Model

```bash
python src/train.py
```

Outputs:

- `saved_models/model.pth` — trained model  
- `saved_models/scaler.pkl` — StandardScaler  
- `visuals/loss.png` — loss curve  
- `visuals/confusion_matrix.png` — confusion matrix  

---

## 🌐 Running the API

```bash
python -m uvicorn app:app --reload
```

Swagger UI:  
http://127.0.0.1:8000/docs

---

## 📡 API Endpoint — POST /predict

### Request example:

```json
{
  "features": [30 numerical values...]
}
```

### Response example:

```json
{
  "logit": -11.298,
  "prob": 0.00012,
  "class": 0
}
```

---

## 🔧 Features

| Feature | Description |
|--------|-------------|
| PyTorch MLP | 2-layer feed-forward neural network |
| Binary classification | benign vs malignant |
| Feature scaling | StandardScaler |
| Evaluation | accuracy, precision, recall, F1 |
| FastAPI | production-ready inference |
| Input validation | Pydantic models |
| Serialization | model + scaler |
| Diagnostics | loss curve + confusion matrix |

---

A strong end‑to‑end ML Engineering project demonstrating model training, evaluation, and a production inference API.
