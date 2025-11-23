# -*- coding: utf-8 -*-
"""
API de churn usando el mejor modelo:
RandomForest + ADASYN con n_estimators=300, max_depth=10, min_samples_split=5

Formas de ejecución:

1) Ejecutando directamente el .py (recomendado para ti ahora):
    python api_churn_rf_adasyn.py
   -> Levanta uvicorn en host 0.0.0.0 y puerto 8001

2) Ejecutando con uvicorn desde consola:
    uvicorn api_churn_rf_adasyn:app --host 0.0.0.0 --port 8001 --reload

Requisitos:
    - El archivo "Bank Customer Churn Prediction.csv" debe estar en el mismo directorio.
"""

import warnings
warnings.filterwarnings("ignore")

from typing import List

import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

import uvicorn


# ========= Configuración general =========
DATA_PATH   = "Bank Customer Churn Prediction.csv"
TARGET_COL  = "churn"
NUM_COLS    = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
CAT_COLS    = ["country", "gender"]
ID_COLS     = ["customer_id"]  # se excluye del entrenamiento

TEST_SIZE     = 0.2
RANDOM_STATE  = 42


# ========= Esquemas de entrada para la API =========
class CustomerFeatures(BaseModel):
    credit_score: float
    age: float
    tenure: float
    balance: float
    products_number: float
    estimated_salary: float
    country: str
    gender: str


class CustomerFeaturesBatch(BaseModel):
    instances: List[CustomerFeatures]


# ========= Funciones auxiliares =========
def make_ohe_dense():
    """OneHotEncoder compatible con imblearn (dense)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Compatibilidad con versiones anteriores de sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocess(num_cols, cat_cols):
    numeric_proc = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_proc = Pipeline(steps=[("ohe", make_ohe_dense())])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_proc, num_cols),
            ("cat", categorical_proc, cat_cols),
        ],
        remainder="drop"
    )
    return preprocess


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "AUC": float(roc_auc_score(y_true, y_prob[:, 1])),
        "F1": float(f1_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_best_rf_adasyn():
    """
    Entrena el mejor modelo encontrado:
    RF_ADASYN_ne300_md10_mss5
    y devuelve el pipeline entrenado + métricas.
    """
    # --- Cargar datos ---
    df = pd.read_csv(DATA_PATH)
    print("Dimensiones del dataset:", df.shape)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + ID_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocess = make_preprocess(NUM_COLS, CAT_COLS)

    # Hiperparámetros del mejor modelo
    cfg = {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5}

    rf_adasyn = RandomForestClassifier(
        **cfg,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("adasyn", ADASYN(random_state=RANDOM_STATE)),
        ("model", rf_adasyn),
    ])

    # Entrenar
    pipeline.fit(X_train, y_train)

    # Métricas de validación
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("✅ Modelo RF_ADASYN_ne300_md10_mss5 entrenado")
    print("📊 Métricas de validación:", metrics)

    return pipeline, metrics


# ========= Entrenar modelo al cargar el módulo =========
print("🚀 Entrenando modelo RF_ADASYN_ne300_md10_mss5 para la API...")
MODEL, MODEL_METRICS = train_best_rf_adasyn()
print("✅ Modelo listo para servir predicciones.")


# ========= Definición de la API =========
app = FastAPI(
    title="Bank Churn API - RF_ADASYN",
    description=(
        "API para predecir churn de clientes bancarios usando "
        "RandomForest + ADASYN (ne300, md10, mss5)."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Endpoint de salud."""
    return {
        "status": "ok",
        "model": "RF_ADASYN_ne300_md10_mss5",
        "metrics": MODEL_METRICS,
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):
    """
    Predicción de churn para un solo cliente.
    Devuelve:
        - churn_probability: probabilidad de que churn = 1
        - churn_prediction: 0 o 1 (umbral 0.5)
    """
    data = pd.DataFrame([customer.dict()])
    proba = MODEL.predict_proba(data)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "churn_probability": float(proba),
        "churn_prediction": pred,
    }


@app.post("/predict_batch")
def predict_batch(batch: CustomerFeaturesBatch):
    """
    Predicción de churn para un lote de clientes.
    """
    records = [c.dict() for c in batch.instances]
    data = pd.DataFrame(records)

    probas = MODEL.predict_proba(data)[:, 1]
    preds = (probas >= 0.5).astype(int)

    results = []
    for i, (p, prob) in enumerate(zip(preds, probas)):
        results.append(
            {
                "index": i,
                "churn_probability": float(prob),
                "churn_prediction": int(p),
            }
        )

    return {"results": results}


# ========= Punto de entrada para ejecutar y dejarlo escuchando =========
if __name__ == "__main__":
    # Host 0.0.0.0 para que responda a la IP pública de la VM
    # Puerto 8001 como pediste
    print("🌐 Iniciando servidor en 0.0.0.0:8001 ...")
    uvicorn.run(
        "api_churn_rf_adasyn:app",
        host="0.0.0.0",
        port=8001,
        reload=False  # puedes poner True en desarrollo si quieres autoreload
    )
