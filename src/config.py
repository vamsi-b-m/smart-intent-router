from pathlib import Path
import os
import yaml

from src.mlflow_config import resolve_tracking_uri

ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = ROOT / "params.yaml"

_params = yaml.safe_load(PARAMS_FILE.read_text())

TRAIN_CONFIG = _params["train"]

MLFLOW_CONFIG = {
    "tracking_uri": resolve_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            _params["mlflow"]["tracking_uri"],
        )
    ),
    "experiment_name": os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        _params['mlflow']['experiment_name'],
    ),
    "registered_model_name": os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        _params["mlflow"]["registered_model_name"],
    ),
    "model_alias": os.getenv(
        "MLFLOW_MODEL_ALIAS",
        "champion",
    )
}