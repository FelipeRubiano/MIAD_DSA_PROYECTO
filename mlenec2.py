# -*- coding: utf-8 -*-
"""
Entrena modelos de churn y registra métricas/parámetros/modelos en MLflow.
Incluye:
- Regresión Logística balanceada
- RandomForest + ADASYN (ejemplos de hiperparámetros)
- RandomForest + SMOTEENN (múltiples combinaciones para experimentar)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# ========= Configuración general =========
DATA_PATH   = "Bank Customer Churn Prediction.csv"
TARGET_COL  = "churn"
NUM_COLS    = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
CAT_COLS    = ["country", "gender"]
ID_COLS     = ["customer_id"]

TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# ========= Configuración MLflow =========
mlflow.set_tracking_uri("http://localhost:8050")
mlflow.set_experiment("bank-churn-local-tests")
print("✔️ MLflow configurado correctamente")

# ========= Funciones =========
def make_ohe_dense():
    """OneHotEncoder compatible con imblearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
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

def train_eval_log(nombre_modelo, pipeline, X_train, y_train, X_test, y_test, extra_params=None):
    """Entrena, evalúa y registra métricas y modelo en MLflow."""
    with mlflow.start_run(run_name=nombre_modelo):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Registrar parámetros
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

        # Firma e input_example
        sample_X = X_test.head(5)
        sample_pred = pipeline.predict_proba(sample_X)
        signature = infer_signature(sample_X, sample_pred)

        mlflow.sklearn.log_model(
            pipeline,
            name=f"model_{nombre_modelo}",
            signature=signature,
            input_example=sample_X
        )

        print(f"✅ '{nombre_modelo}' registrado | {metrics}")
        return metrics

# ========= Ejecución principal =========
def main():
    # --- Cargar datos ---
    df = pd.read_csv(DATA_PATH)
    print("Dimensiones del dataset:", df.shape)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + ID_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    preprocess = make_preprocess(NUM_COLS, CAT_COLS)

    # -------- Modelo 1: Regresión Logística balanceada --------
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    pipe_lr = Pipeline(steps=[("preprocess", preprocess), ("model", logreg)])
    train_eval_log("LogReg_Balanceada", pipe_lr, X_train, y_train, X_test, y_test)

    # -------- Modelo 2: RandomForest + ADASYN --------
    rf_adasyn_configs = [
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 500, "max_depth": 15, "min_samples_split": 5},
    ]
    for cfg in rf_adasyn_configs:
        rf_adasyn = RandomForestClassifier(**cfg, random_state=RANDOM_STATE, n_jobs=-1)
        pipe_rf_adasyn = ImbPipeline(steps=[
            ("preprocess", preprocess),
            ("adasyn", ADASYN(random_state=RANDOM_STATE)),
            ("model", rf_adasyn),
        ])
        name = f"RF_ADASYN_ne{cfg['n_estimators']}_md{cfg['max_depth']}_mss{cfg['min_samples_split']}"
        train_eval_log(name, pipe_rf_adasyn, X_train, y_train, X_test, y_test, extra_params=cfg)

    # -------- Modelo 3: RandomForest + SMOTEENN --------
    rf_smoteenn_configs = [
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": 15, "min_samples_split": 4, "min_samples_leaf": 3, "max_features": "log2"},
        {"n_estimators": 500, "max_depth": 20, "min_samples_split": 3, "min_samples_leaf": 2, "max_features": None},
        {"n_estimators": 700, "max_depth": 25, "min_samples_split": 5, "min_samples_leaf": 4, "max_features": "sqrt"},
        {"n_estimators": 800, "max_depth": 30, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "log2"},
        {"n_estimators": 1000, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    ]

    for cfg in rf_smoteenn_configs:
        rf_smoteenn = RandomForestClassifier(**cfg, random_state=RANDOM_STATE, n_jobs=-1)
        pipe_rf_smoteenn = ImbPipeline(steps=[
            ("preprocess", preprocess),
            ("smoteenn", SMOTEENN(random_state=RANDOM_STATE)),
            ("model", rf_smoteenn),
        ])
        name = f"RF_SMOTEENN_ne{cfg['n_estimators']}_md{cfg['max_depth']}_mss{cfg['min_samples_split']}_msl{cfg['min_samples_leaf']}_mf{cfg['max_features']}"
        train_eval_log(name, pipe_rf_smoteenn, X_train, y_train, X_test, y_test, extra_params=cfg)

if __name__ == "__main__":
    main()
