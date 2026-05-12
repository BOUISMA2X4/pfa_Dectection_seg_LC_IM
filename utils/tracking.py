# MLflow experiment tracking
# Enregistre params, métriques, modèle

import mlflow
import mlflow.pytorch
from contextlib import contextmanager

@contextmanager
def start_run(run_name: str, params: dict):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        yield

def log_epoch(metrics: dict, step: int) -> None:
    mlflow.log_metrics(metrics, step=step)

def log_model(model, name: str) -> None:
    mlflow.pytorch.log_model(model, name)