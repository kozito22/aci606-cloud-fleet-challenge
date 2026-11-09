"""Operational metrics collected by a lightweight ASGI middleware.

Counters live in the memory of each process (like a Prometheus exporter) and
each process saves them in the database after its requests. On a serverless
platform several instances answer in parallel, so /metrics adds up the
counters of every instance: the totals describe the whole service, not just
the instance that happened to answer. Uptime and latency percentiles are per
instance (percentiles cannot be added up).
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import Counter, deque

from starlette.concurrency import run_in_threadpool

STARTED_AT = time.time()
INSTANCE_ID = os.environ.get("RENDER_INSTANCE_ID") or uuid.uuid4().hex[:8]
ACTIVE_WINDOW_S = 300     # an instance seen in the last 5 minutes counts as active

_lock = threading.Lock()
requests_by_route: Counter[str] = Counter()
errors_by_route: Counter[str] = Counter()
status_classes: Counter[str] = Counter()
predictions_by_status: Counter[str] = Counter()
# Recent request latencies (ms) of /predict measured inside the server and
# pure model time reported as inference_ms.
predict_latency_ms: deque[float] = deque(maxlen=2000)
inference_ms: deque[float] = deque(maxlen=2000)
predict_latency_sum = 0.0
predict_ok = 0
telemetry_times: deque[float] = deque(maxlen=20000)
_flushing = False   # a save is running in this instance
_dirty = False      # counters changed since the last save started


def record_prediction(status: str, model_ms: float) -> None:
    with _lock:
        predictions_by_status[status] += 1
        inference_ms.append(model_ms)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _counters() -> dict:
    """Additive counters of this instance (what gets saved and summed)."""
    with _lock:
        return {
            "requests_total": sum(requests_by_route.values()),
            "telemetry_requests": requests_by_route["POST /telemetry"],
            "prediction_requests": requests_by_route["POST /predict"],
            "errors_4xx": status_classes["4xx"],
            "errors_5xx": status_classes["5xx"],
            "telemetry_errors": errors_by_route["POST /telemetry"],
            "prediction_errors": errors_by_route["POST /predict"],
            "predict_latency_sum_ms": predict_latency_sum,
            "predict_ok": predict_ok,
            "predictions_by_status": dict(predictions_by_status),
        }


def flush(force: bool = False) -> None:
    """Save this instance's counters (never raises: metrics must not break requests)."""
    try:
        from app import db
        db.save_metrics_snapshot(INSTANCE_ID, STARTED_AT, _counters())
    except Exception as exc:  # database asleep, etc.: try again on the next flush
        print(f"metrics flush failed: {exc}")


def _service_totals(own: dict) -> dict:
    """Add up the latest counters of every instance (this one with live values)."""
    try:
        from app import db
        rows = [r for r in db.load_metrics_snapshots() if r["instance_id"] != INSTANCE_ID]
    except Exception:
        rows = []
    now = time.time()
    totals = {k: 0 for k in own if k != "predictions_by_status"}
    by_status: Counter[str] = Counter(own["predictions_by_status"])
    for k in totals:
        totals[k] += own[k]
    for r in rows:
        for k in totals:
            totals[k] += r["counters"].get(k, 0)
        by_status.update(r["counters"].get("predictions_by_status", {}))
    totals["predictions_by_status"] = dict(by_status)
    totals["instances_total"] = len(rows) + 1
    totals["instances_active"] = 1 + sum(1 for r in rows if now - r["updated_at"] <= ACTIVE_WINDOW_S)
    totals["since"] = min([STARTED_AT] + [r["started_at"] for r in rows])
    return totals


def snapshot() -> dict:
    now = time.time()
    own = _counters()
    with _lock:
        latencies = list(predict_latency_ms)
        model_times = list(inference_ms)
        telemetry_last_min = sum(1 for t in telemetry_times if now - t <= 60)
        by_route = dict(requests_by_route)
    total = _service_totals(own)
    return {
        "scope": "service",  # counters below add up every instance of the deployment
        "uptime_seconds": round(now - STARTED_AT, 1),
        "started_at": STARTED_AT,
        "instance_id": INSTANCE_ID,
        "instances_active": total["instances_active"],
        "requests_total": total["requests_total"],
        "telemetry_requests": total["telemetry_requests"],
        "prediction_requests": total["prediction_requests"],
        "errors_total": total["errors_4xx"] + total["errors_5xx"],
        "errors_4xx": total["errors_4xx"],
        "errors_5xx": total["errors_5xx"],
        "telemetry_errors": total["telemetry_errors"],
        "prediction_errors": total["prediction_errors"],
        "avg_prediction_latency_ms": round(total["predict_latency_sum_ms"] / total["predict_ok"], 3) if total["predict_ok"] else None,
        "p50_prediction_latency_ms": _percentile(latencies, 50),   # this instance
        "p95_prediction_latency_ms": _percentile(latencies, 95),   # this instance
        "avg_inference_ms": round(sum(model_times) / len(model_times), 3) if model_times else None,
        "telemetry_per_minute": telemetry_last_min,                # this instance
        "predictions_by_status": total["predictions_by_status"],
        "counting_since": total["since"],
        "instance": {
            "requests_total": own["requests_total"],
            "telemetry_requests": own["telemetry_requests"],
            "prediction_requests": own["prediction_requests"],
            "errors_4xx": own["errors_4xx"],
            "errors_5xx": own["errors_5xx"],
            "requests_by_route": by_route,
        },
    }


async def _save_after_request() -> None:
    """Save the counters after every request without piling up writes: if a
    save is already running, mark the counters dirty and that save loops once
    more, so no request is ever left out of the service totals."""
    global _flushing, _dirty
    _dirty = True
    if _flushing:
        return
    _flushing = True
    try:
        while _dirty:
            _dirty = False
            await run_in_threadpool(flush)
    finally:
        _flushing = False


class MetricsMiddleware:
    """Pure ASGI middleware (cheaper than BaseHTTPMiddleware)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            key = f"{scope['method']} {path}"
            code = status["code"]
            global predict_latency_sum, predict_ok
            with _lock:
                requests_by_route[key] += 1
                if code >= 500:
                    status_classes["5xx"] += 1
                    errors_by_route[key] += 1
                elif code >= 400:
                    status_classes["4xx"] += 1
                    errors_by_route[key] += 1
                if key == "POST /predict" and code < 400:
                    predict_latency_ms.append(elapsed_ms)
                    predict_latency_sum += elapsed_ms
                    predict_ok += 1
                elif key == "POST /telemetry" and code < 400:
                    telemetry_times.append(time.time())
            # The response has already been sent when this runs.
            await _save_after_request()
