from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from app import backups, coverage, db, metrics, model


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
    stop_backups = backups.start_scheduler()
    yield
    if stop_backups:
        stop_backups.set()


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
def robot_history(
    robot_id: str,
    limit: int | None = Query(None, ge=1, le=100_000),
    details: bool = Query(False, description="include predicted_status, confidence and received_at"),
) -> list[dict]:
    rows = db.robot_history(robot_id, limit, with_prediction=details)
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


@app.get("/fleet")
def fleet(connected_s: float = Query(30, gt=0, le=3_600)) -> dict:
    """Snapshot for the dashboard: robots with their predicted condition,
    fleet status counts and operational metrics, in a single call."""
    now = time.time()
    robots = db.latest_per_robot(with_prediction=True)
    status_counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
    for robot in robots:
        robot["age_s"] = round(now - robot["received_at"], 1)
        robot["connected"] = robot["age_s"] <= connected_s
        if robot["connected"] and robot["predicted_status"] in status_counts:
            status_counts[robot["predicted_status"]] += 1
    return {
        "server_time": now,
        "robots": robots,
        "connected": sum(r["connected"] for r in robots),
        "status_counts": status_counts,
        "metrics": metrics.snapshot(),
        "storage": {"backend": db.backend_name(), **db.counts()},
    }


@app.get("/fleet/summary")
def fleet_summary() -> dict:
    """Slower-changing dashboard data: coverage, 15-minute trend, backups, model."""
    backup_info = backups.status()
    backup_info.pop("history", None)
    return {
        "coverage": coverage.compute(window_s=60),
        "trend": db.trend(15),
        "backups": backup_info,
        "model": model.meta() if model.available() else None,
    }


@app.get("/coverage")
def get_coverage(
    window_s: float = Query(300, gt=0, le=86_400),
    connected_s: float = Query(30, gt=0, le=3_600),
) -> dict:
    """Area covered by the fleet and optimal target point for each robot."""
    return coverage.compute(window_s, connected_s)


@app.get("/backups")
def backup_status() -> dict:
    """Backup policy (how often each backup runs), last runs and next due time."""
    return backups.status()


@app.post("/backups/run")
def run_backup(
    kind: Literal["logical", "physical"] = Query("logical"),
    x_backup_token: str | None = Header(None),
) -> dict:
    # On a public deployment only someone holding BACKUP_TOKEN may trigger it.
    token = os.environ.get("BACKUP_TOKEN")
    if token and x_backup_token != token:
        raise HTTPException(status_code=403, detail="Invalid backup token")
    if not token and not db.IS_SQLITE:
        raise HTTPException(status_code=403, detail="Set BACKUP_TOKEN to enable remote backups")
    return backups.run(kind)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    # Opening the bare public URL lands on the operator dashboard.
    return RedirectResponse("/dashboard")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # Browsers request it automatically; answering 204 keeps it out of the 4xx count.
    return Response(status_code=204)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
