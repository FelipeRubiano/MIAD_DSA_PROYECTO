# src/train.py
import os
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import joblib
import yaml

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# MLflow
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict):
    d = cfg["data"]
    data_dir = Path(d["data_dir"])
    X_train = pd.read_csv(data_dir / d["X_train"])
    X_test  = pd.read_csv(data_dir / d["X_test"])
    y_train = pd.read_csv(data_dir / d["y_train"]).squeeze()
    y_test  = pd.read_csv(data_dir / d["y_test"]).squeeze()
    return X_train, X_test, y_train, y_test


def compute_scale_pos_weight(y):
    counter = Counter(y)
    # evita división por cero
    n_pos = counter.get(1, 0)
    n_neg = counter.get(0, 0)
    return (n_neg / n_pos) if n_pos > 0 else 1.0


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def train(cfg_path: str = "configs/config.yaml"):
    cfg = load_config(cfg_path)

    # ----- carga datos -----
    X_train, X_test, y_train, y_test = load_data(cfg)

    # ----- balanceo -----
    use_smote = cfg["balance"]["use_smote"]
    use_spw   = cfg["balance"]["use_scale_pos_weight"]

    if use_smote and use_spw:
        print("Aviso: use_smote=true y use_scale_pos_weight=true. Usaré SMOTE y fijaré scale_pos_weight=1.0.")
        use_spw = False

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        scale_pos_weight = 1.0
    else:
        X_train_res, y_train_res = X_train, y_train
        scale_pos_weight = compute_scale_pos_weight(y_train) if use_spw else 1.0
        if use_spw:
            print(f"scale_pos_weight calculado: {scale_pos_weight:.4f}")

    # ----- modelo -----
    params = cfg["model"]["params"].copy()
    params["scale_pos_weight"] = scale_pos_weight  # forzamos coherencia con balanceo
    model = XGBClassifier(**params)

    # ----- entrenamiento -----
    model.fit(X_train_res, y_train_res)

    # ----- evaluación -----
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = float(cfg["training"]["test_threshold"])

    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cls_rep = classification_report(y_test, y_pred, digits=4)

    print("\nMATRIZ DE CONFUSIÓN:\n", cm)
    print("\nCLASIFICACIÓN:\n", cls_rep)
    print(f"AUC-ROC: {auc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")

    # ----- artefactos -----
    art_dir = Path(cfg["artifacts"]["dir"])
    ensure_dir(art_dir)
    model_path = art_dir / cfg["artifacts"]["model_filename"]
    metrics_path = art_dir / cfg["artifacts"]["metrics_filename"]

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "auc": auc,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nModelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")

    # ----- MLflow 
    if cfg.get("mlflow", {}).get("enabled", False):
        if mlflow is None:
            print("MLflow no está instalado en el entorno. Omite logging.")
            return

        mlcfg = cfg["mlflow"]
        mlflow.set_tracking_uri(mlcfg.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(mlcfg.get("experiment_name", "default"))

        with mlflow.start_run(run_name="xgb_smote" if use_smote else "xgb_spw"):
            # log params
            mlflow.log_params({
                **{f"xgb__{k}": v for k, v in params.items()},
                "use_smote": use_smote,
                "use_scale_pos_weight": cfg["balance"]["use_scale_pos_weight"],
                "threshold": threshold,
            })
            # log metrics
            mlflow.log_metrics({
                "auc": float(auc),
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
            })
            # log artifacts
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
            print("📦 Artefactos y métricas registrados en MLflow.")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Ruta al archivo de configuración")
    args = parser.parse_args()

    # Opcional: normalizar a ruta absoluta
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()

    train(str(cfg_path))
