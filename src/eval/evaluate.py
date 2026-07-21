"""
Evaluates the most recently trained model version against the held-out
test set, logs the results to its MLflow run, and decides whether it
should become the new "champion" (the version served in production).
 
Promotion logic (kept deliberately simple):
    - If no version currently holds the `champion` alias, promote this one.
    - Else compare test_macro_f1 against the champion's test_macro_f1
      (read from the champion's own run) and promote only if strictly
      better. The outgoing champion gets re-aliased to `previous_champion`
      so you can always roll back to it by hand.
 
Uses MLflow's alias system (set_registered_model_alias) rather than the
older Staging/Production "stages" API, which MLflow has deprecated in
favor of aliases + tags.
 
Usage:
    python src/eval/evaluate.py
"""
import json
import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.mlflow_config import resolve_tracking_uri

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PARAMS_PATH = ROOT / "params.yaml"
LABEL_COL = "route"

CHAMPION_ALIAS = "champion"
PREVIOUS_CHAMPION_ALIAS = "previous_champion"

def load_params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text())

def evaluate_model(pipeline, test_df: pd.DataFrame) -> dict:
    preds = pipeline.predict(test_df["text"])
    y_true = test_df[LABEL_COL]

    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    report = classification_report(y_true, preds, output_dict=True)
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, preds, labels=labels).tolist()

    return {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "n_test": len(test_df),
        "per_class_report": report,
        "confusion_matrix": {"labels": labels, "matrix": cm}
    }

def maybe_promote(client: MlflowClient, model_name: str, new_version: str, new_macro_f1: float):
    """Compare the just-trained version against the current champion and
    promote it via alias if it's better (or if there's no champion yet)."""
    try:
        champion_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        champion_version = None

    if champion_version is None:
        logger.info(f"No existing champion for '{model_name}' -- promoting v{new_version}.")
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version)
        return True, None
    
    champion_run = client.get_run(champion_version.run_id)
    champion_f1 = champion_run.data.metrics.get("test_macro_f1")
 
    if champion_f1 is None:
        logger.warning(
            f"Champion v{champion_version.version} has no test_macro_f1 logged yet "
            f"(likely never evaluated) -- promoting v{new_version} by default."
        )
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version)
        return True, champion_version.version
 
    logger.info(
        f"Challenger v{new_version} test_macro_f1={new_macro_f1:.4f} vs "
        f"champion v{champion_version.version} test_macro_f1={champion_f1:.4f}"
    )
 
    if new_macro_f1 > champion_f1:
        client.set_registered_model_alias(
            model_name, PREVIOUS_CHAMPION_ALIAS, champion_version.version
        )
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version)
        logger.info(f"Promoted v{new_version} to '{CHAMPION_ALIAS}'.")
        return True, champion_version.version
    else:
        logger.info(
            f"v{new_version} did not beat the champion -- leaving "
            f"v{champion_version.version} as '{CHAMPION_ALIAS}'."
        )
        return False, champion_version.version 


def main():
    params = load_params()
    mlflow_params = params["mlflow"]
    model_name = mlflow_params['registered_model_name']

    tracking_uri = resolve_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    train_metrics_path = MODELS_DIR / "train_metrics.json"
    if not train_metrics_path.exists():
        raise RuntimeError("No train_metrics.json found -- run train/baseline.py first.")
    train_metrics = json.loads(train_metrics_path.read_text())
    run_id = train_metrics['mlflow_run_id']
    new_version = str(train_metrics['registered_model_version'])

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    pipeline = joblib.load(MODELS_DIR / "baseline_pipeline.joblib")

    results = evaluate_model(pipeline, test_df)
    logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"Test macro-f1: {results['test_macro_f1']:.4f}")
    logger.info("\n" + classification_report(test_df[LABEL_COL], pipeline.predict(test_df['text'])))

    # Log eval metrics onto the SAME run that produced this model version,
    # so a champion's run always has both val_* and test_* metrics together.
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", results['test_accuracy'])
        mlflow.log_metric("test_macro_f1", results['test_macro_f1'])

    promoted, previous_version = maybe_promote(
        client, model_name, new_version, results["test_macro_f1"]
    )

    output_path = MODELS_DIR / "eval_metrics.json"
    results["promoted_to_champion"] = promoted
    results["model_version"] = new_version
    results["previous_champion_version"] = previous_version
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved evaluation results to {output_path}")

if __name__ == "__main__":
    main()
