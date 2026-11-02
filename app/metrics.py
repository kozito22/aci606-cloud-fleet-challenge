"""Operational metrics collected by a lightweight ASGI middleware.

Counters live in the memory of each process (like a Prometheus exporter):
they restart with the process, while the data itself lives in the database.
`instance_id` identifies the process that answered, so a restart or a second
instance is visible in /metrics.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import Counter, deque

STARTED_AT = time.time()
INSTANCE_ID = os.environ.get("RENDER_INSTANCE_ID") or uuid.uuid4().hex[:8]

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


def snapshot() -> dict:
    now = time.time()
    with _lock:
        predict_count = requests_by_route["POST /predict"]
        latencies = list(predict_latency_ms)
        model_times = list(inference_ms)
        telemetry_last_min = sum(1 for t in telemetry_times if now - t <= 60)
        data = {
            "uptime_seconds": round(now - STARTED_AT, 1),
            "started_at": STARTED_AT,
            "instance_id": INSTANCE_ID,
            "requests_total": sum(requests_by_route.values()),
            "telemetry_requests": requests_by_route["POST /telemetry"],
            "prediction_requests": predict_count,
            "errors_total": status_classes["4xx"] + status_classes["5xx"],
            "errors_4xx": status_classes["4xx"],
            "errors_5xx": status_classes["5xx"],
            "telemetry_errors": errors_by_route["POST /telemetry"],
            "prediction_errors": errors_by_route["POST /predict"],
            "avg_prediction_latency_ms": round(predict_latency_sum / predict_ok, 3) if predict_ok else None,
            "p50_prediction_latency_ms": _percentile(latencies, 50),
            "p95_prediction_latency_ms": _percentile(latencies, 95),
            "avg_inference_ms": round(sum(model_times) / len(model_times), 3) if model_times else None,
            "telemetry_per_minute": telemetry_last_min,
            "predictions_by_status": dict(predictions_by_status),
            "requests_by_route": dict(requests_by_route),
        }
    return data


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
