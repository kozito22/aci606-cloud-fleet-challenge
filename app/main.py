from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import db, metrics, model


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm-up: load the model and create tables before the first request so
    # the first /predict does not pay the loading cost. Failures here are not
    # fatal: /health reports them and each module retries lazily.
    for warm_up in (model.load, db.init_db):
        try:
            warm_up()
        except Exception as exc:  # pragma: no cover - logged for the operator
            print(f"startup warning: {warm_up.__module__}: {exc}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="ACI606 Cloud Fleet Challenge",
    version="0.2.0",
    description="Cloud fleet service: persistent telemetry ingestion, ML inference, metrics and dashboard.",
)
app.add_middleware(metrics.MetricsMiddleware)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    # NaN/Infinity are rejected with 422 instead of reaching the database/model.
    model_config = ConfigDict(allow_inf_nan=False)

    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass



@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # The default handler echoes the invalid input; a NaN there cannot be
    # serialised to JSON and would turn a 422 into a 500. Invalid data must
    # always be a 4xx, so we return only location, message and type.
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "database": db.backend_name(),
        "database_ok": db.ping(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    row = sample.model_dump()
    # Classify at ingestion so the dashboard and history show the condition
    # without extra calls. Ingestion must never fail because of the model.
    try:
        row["predicted_status"], row["confidence"] = model.predict(row)
    except Exception:
        row["predicted_status"], row["confidence"] = None, None
    db.insert_telemetry(row, received_at=time.time())
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    return db.latest_per_robot()


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str, limit: int | None = Query(None, ge=1, le=100_000)) -> list[dict]:
    rows = db.robot_history(robot_id, limit)
    if rows is None:
        raise HTTPException(status_code=404, detail="Robot not found")
    return rows


@app.post("/predict")
def predict(sample: PredictionRequest) -> dict:
    if not model.available():
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python ml/train_model.py")
    start = time.perf_counter()
    status, confidence = model.predict(sample.model_dump())
    inference_ms = (time.perf_counter() - start) * 1000
    metrics.record_prediction(status, inference_ms)
    # Exact response contract required by the evaluator (do not rename fields).
    return {
        "robot_id": sample.robot_id,
        "predicted_status": status,
        "confidence": round(confidence, 4),
        "model_version": model.version(),
        "inference_ms": round(inference_ms, 3),
    }


@app.get("/metrics")
def get_metrics() -> dict:
    data = metrics.snapshot()
    data["persistence"] = db.backend_name()
    try:
        data.update(db.counts())
    except Exception:
        data["database_ok"] = False
    data["prediction_service"] = "READY" if model.available() else "MODEL_NOT_LOADED"
    if model.available():
        meta = model.meta()
        data["model"] = {key: meta.get(key) for key in ("model_version", "model", "features", "test_f1_macro",
                                                         "baseline_f1_macro", "trained_at")}
    return data


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
