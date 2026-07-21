"""
FastAPI wrapper around the Smart Router model.

Loads the model tagged with the `champion` alias straight from the MLflow
Model Registry (models:/<name>@champion) rather than a local joblib file --
whatever evaluate.py most recently promoted is what gets served. Falls back
to the local joblib copy if MLflow is unreachable/misconfigured, so the app
still works standalone (e.g. before Phase 1 tooling is set up).

Endpoints:
    GET  /health          -> liveness check + which model version is loaded
    POST /route            -> {"text": "..."} -> {"route", "confidence", "top_k"}

Usage:
    uvicorn src.serve.app:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from src.mlflow_config import resolve_tracking_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"
FALLBACK_MODEL_PATH = ROOT / "models" / "baseline_pipeline.joblib"

ml_models = {}


def load_champion_from_registry() -> tuple:
    """Returns (pipeline, version_label). Raises if unavailable."""
    params = yaml.safe_load(PARAMS_PATH.read_text())
    mlflow_params = params["mlflow"]

    tracking_uri = resolve_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    model_name = mlflow_params["registered_model_name"]

    client = MlflowClient()
    champion = client.get_model_version_by_alias(model_name, "champion")
    model_uri = f"models:/{model_name}@champion"

    logger.info(f"Loading champion model: {model_uri} (version {champion.version})")
    # sklearn Pipelines round-trip through mlflow.sklearn.load_model with
    # predict_proba intact, which mlflow.pyfunc's generic wrapper does not
    # expose -- so we load via the sklearn flavor specifically.
    pipeline = mlflow.sklearn.load_model(model_uri)
    return pipeline, f"{model_name}@champion (v{champion.version})"


def load_fallback() -> tuple:
    logger.warning(
        f"Falling back to local joblib file at {FALLBACK_MODEL_PATH} "
        "(MLflow registry unavailable or no champion set yet)."
    )
    if not FALLBACK_MODEL_PATH.exists():
        raise RuntimeError(
            "No MLflow champion and no local fallback model found. "
            "Run `python src/train/baseline.py` and `python src/eval/evaluate.py` first."
        )
    return joblib.load(FALLBACK_MODEL_PATH), "local_fallback"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline, version_label = load_champion_from_registry()
    except Exception as e:
        logger.warning(f"Could not load champion from MLflow registry: {e}")
        pipeline, version_label = load_fallback()

    ml_models["pipeline"] = pipeline
    ml_models["version_label"] = version_label
    logger.info(f"Model loaded ({version_label}), ready to serve.")
    yield
    ml_models.clear()


app = FastAPI(title="Smart Router", version="0.2.0", lifespan=lifespan)


class RouteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["I need to reset my password"])


class RoutePrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    route: str
    confidence: float
    top_k: list[dict]
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "pipeline" in ml_models,
        "model_version": ml_models.get("version_label"),
    }


@app.post("/route", response_model=RoutePrediction)
def route(request: RouteRequest):
    pipeline = ml_models.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    proba = pipeline.predict_proba([request.text])[0]
    classes = pipeline.classes_
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    top_class, top_conf = ranked[0]
    top_k = [{"route": c, "confidence": round(float(p), 4)} for c, p in ranked[:3]]

    logger.info(f"text={request.text!r} -> route={top_class} conf={top_conf:.4f}")

    return RoutePrediction(
        route=top_class,
        confidence=round(float(top_conf), 4),
        top_k=top_k,
        model_version=ml_models["version_label"],
    )