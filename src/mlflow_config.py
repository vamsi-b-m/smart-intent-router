"""
Shared helpers for MLflow config so every entrypoint (train, evaluate,
serve) resolves the tracking URI identically -- regardless of which
directory the command happens to be run from.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_tracking_uri(tracking_uri: str) -> str:
    """
    A relative sqlite URI (e.g. "sqlite:///mlflow.db") resolves relative to
    the current working directory, NOT the project root -- so running the
    same command from two different directories silently reads/writes two
    different databases. This anchors it to the project root instead.

    Non-sqlite URIs (http://, https://, a real mlflow server, etc.) are
    passed through unchanged.
    """
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return tracking_uri

    path_part = tracking_uri[len(prefix):]
    path = Path(path_part)
    if path.is_absolute():
        return tracking_uri

    absolute_path = (PROJECT_ROOT / path).resolve()
    return f"{prefix}{absolute_path}"
