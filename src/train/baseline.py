"""
Trains the baseline Smart Router model: TF-IDF + Logistic Regression,
predicting the coarse `route` label (which agent should handle the message).

This is intentionally a fast, cheap-to-retrain baseline -- the point of
Phase 0 is proving the pipeline shape works end to end before introducing
a heavier model (e.g. fine-tuned DistilBERT) in a later phase.

Hyperparameters come from params.yaml so DVC can track/diff them across
experiments (`dvc exp run --set-param train.C=1.0`). Every run is also
logged to MLflow (params, metrics, the fitted pipeline as a model artifact,
and a new Model Registry version) so runs are comparable in the MLflow UI
independent of DVC.
 
Usage:
    python src/train/baseline.py
    dvc repro train                          # via the DVC pipeline
    dvc exp run --set-param train.C=1.0      # a one-off experiment variant
"""
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.mlflow_config import resolve_tracking_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PARAMS_PATH = ROOT / "params.yaml"

LABEL_COL = "route"


def load_params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text())


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def build_pipeline(train_params: dict) -> Pipeline:
    class_weight = train_params["class_weight"]
    if class_weight == "None":
        class_weight = None

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, train_params["ngram_max"]),
                    min_df=2,
                    max_features=train_params["max_features"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=train_params["C"],
                    class_weight=class_weight,  # fallback_human is a small class
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    params = load_params()
    train_params = params["train"]
    mlflow_params = params["mlflow"]

    tracking_uri = resolve_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    train_df = load_split("train")
    val_df = load_split("val")

    logger.info(f"Training on {len(train_df)} examples, validating on {len(val_df)}")
    logger.info(f"Params: {train_params}")

    with mlflow.start_run() as run:
        mlflow.log_params(train_params)
        mlflow.log_params({"n_train": len(train_df)})

        pipeline = build_pipeline(train_params)
        pipeline.fit(train_df["text"], train_df[LABEL_COL])

        val_preds = pipeline.predict(val_df["text"])
        val_acc = accuracy_score(val_df[LABEL_COL], val_preds)
        val_f1 = f1_score(val_df[LABEL_COL], val_preds, average="macro")

        logger.info(f"Validation accuracy: {val_acc:.4f}")
        logger.info(f"Validation macro-F1:  {val_f1:.4f}")

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_macro_f1", val_f1)

        # Log the fitted pipeline as an MLflow model AND register it in the
        # Model Registry in one call. It lands as a new version with no
        # stage/alias set -- evaluate.py decides afterwards whether it's
        # good enough to become the new "champion".
        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=mlflow_params["registered_model_name"],
        )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "baseline_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"Saved local copy of the pipeline to {model_path}")

        metrics = {
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "n_train": len(train_df),
            "mlflow_run_id": run.info.run_id,
            "mlflow_model_uri": model_info.model_uri,
            "registered_model_version": model_info.registered_model_version,
        }

        metrics_path = MODELS_DIR / "train_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Saved metrics to {metrics_path}")
        logger.info(
            f"MLflow run: {run.info.run_id} | "
            f"registered as '{mlflow_params['registered_model_name']}'"
            f"v{model_info.registered_model_version}"
        )


if __name__ == "__main__":
    main()
