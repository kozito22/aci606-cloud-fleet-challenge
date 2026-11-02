from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import db

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.2.0",
    description="Cloud fleet service: persistent telemetry ingestion, ML inference, metrics and dashboard.",
)

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


REQUEST_COUNT = 0


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
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    db.insert_telemetry(sample.model_dump(), received_at=time.time())
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
def predict(_: PredictionRequest) -> dict:
    # TODO E4: load the trained model and return a real prediction.
    # Required contract is documented in README.md and the student guide.
    raise HTTPException(
        status_code=501,
        detail="/predict is part of the challenge. Train and integrate your model.",
    )


@app.get("/metrics")
def metrics() -> dict:
    # TODO E4: extend with latency, errors, throughput and model metrics.
    return {
        "requests_received": REQUEST_COUNT,
        **db.counts(),
        "persistence": db.backend_name(),
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
