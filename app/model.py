"""Model loading and inference.

The pipeline (scaler + classifier) is loaded once per process from MODEL_PATH
(default models/model.joblib) and reused by every request.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / os.environ.get("MODEL_PATH", "models/model.joblib")
META_PATH = MODEL_PATH.with_name("model_meta.json")

_lock = threading.Lock()
_pipeline = None
_meta: dict = {}


def load() -> None:
    global _pipeline, _meta
    if _pipeline is not None:
        return
    with _lock:
        if _pipeline is None:
            _meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
            pipeline = joblib.load(MODEL_PATH)
            # One dummy prediction warms up sklearn's internal validation path.
            pipeline.predict_proba(np.zeros((1, len(_meta.get("features", [])) or pipeline.n_features_in_)))
            _pipeline = pipeline


def available() -> bool:
    try:
        load()
        return True
    except Exception:
        return False


def meta() -> dict:
    load()
    return _meta


def version() -> str:
    return meta().get("model_version", "v1")


def features() -> list[str]:
    return meta()["features"]


def predict(sample: dict) -> tuple[str, float]:
    """Return (predicted_status, confidence) for one telemetry sample."""
    load()
    x = np.array([[sample[f] for f in _meta["features"]]], dtype=float)
    proba = _pipeline.predict_proba(x)[0]
    best = int(np.argmax(proba))
    return str(_pipeline.classes_[best]), float(proba[best])
