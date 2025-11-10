# -*- coding: utf-8 -*-
"""
Entrena modelos para churn bancario y registra métricas/artefactos en MLflow.
Modelos: Regresión Logística, RF + ADASYN, RF + SMOTEENN.
Sin visualizaciones.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# === Datos ===
DATA_PATH = "Bank Customer Churn Prediction.csv"  # ajusta si tu CSV tiene otro nombre
TARGET_COL = "churn"
NUM_COLS = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
CAT_COLS = ["country", "gender"]
ID_COLS  = ["customer_id"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# === Scikit-learn / imblearn ===
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# === MLflow ===
import mlflow
import mlflow.sklearn

# -----------------------------
# Configuración de MLflow
# -----------------------------
# Cambia localhost por la IP del servidor MLflow si aplica
mlflow.set_tracking_uri("http://localhost:8050")
mlflow.set_experiment("bank-churn-local-tests")
print("✔️ MLflow configurado")

# -----------------------------
# Funciones auxiliares
# -----------------------------
def make_preprocess(num_cols, cat_cols):
    numeric_proc = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_proc = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_proc, num_cols),
            ("cat", categorical_proc, cat_cols),
        ]
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

def train_eval_log(nombre_modelo, pipeline, X_train, y_train, X_test, y_test, extra_params=None):
    """
    Entrena, evalúa y registra en MLflow (parámetros, métricas, modelo).
    """
    with mlflow.start_run(run_name=nombre_modelo):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)

        try:
            model_params = pipeline.named_steps["model"].get_params()
        except Exception:
            model_params = {}

        if extra_params:
            model_params.update(extra_params)

        for k, v in model_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                mlflow.log_param(k, str(v))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{nombre_modelo}")

        print(f"✅ '{nombre_modelo}' registrado | {metrics}")
        return metrics

# -----------------------------
# Ejecución principal
# -----------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    print("Dimensiones del dataset:", df.shape)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + ID_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    preprocess = make_preprocess(NUM_COLS, CAT_COLS)

    # --- Modelo 1: Regresión Logística balanceada ---
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    pipe_lr = Pipeline(steps=[("preprocess", preprocess), ("model", logreg)])
    train_eval_log("LogReg_Balanceada", pipe_lr, X_train, y_train, X_test, y_test)

    # --- Modelo 2: RF + ADASYN ---
    adasyn = ADASYN(random_state=RANDOM_STATE)
    rf_adasyn = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    pipe_rf_adasyn = ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("adasyn", adasyn),
        ("model", rf_adasyn),
    ])
    train_eval_log("RandomForest_ADASYN", pipe_rf_adasyn, X_train, y_train, X_test, y_test)

    # --- Modelo 3: RF + SMOTEENN ---
    smoteenn = SMOTEENN(random_state=RANDOM_STATE)
    rf_smoteenn = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    pipe_rf_smoteenn = ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("smoteenn", smoteenn),
        ("model", rf_smoteenn),
    ])
    train_eval_log("RandomForest_SMOTEENN", pipe_rf_smoteenn, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
