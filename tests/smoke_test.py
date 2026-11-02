from __future__ import annotations

import argparse
import time

import requests

SAMPLE = {
    "robot_id": "TEST-01",
    "timestamp": time.time(),
    "x": 1.2,
    "y": 3.4,
    "speed": 0.72,
    "battery": 81.0,
    "motor_temp": 49.0,
    "vibration": 0.42,
    "motor_current": 3.8,
    "obstacle_distance": 1.3,
}

PREDICT_CONTRACT = {
    "robot_id": str,
    "predicted_status": str,
    "confidence": (int, float),
    "model_version": str,
    "inference_ms": (int, float),
}


def check_predict_contract(response: requests.Response) -> str | None:
    if not response.ok:
        return f"HTTP {response.status_code}"
    body = response.json()
    if set(body) != set(PREDICT_CONTRACT):
        return f"fields {sorted(body)} != {sorted(PREDICT_CONTRACT)}"
    for field, kind in PREDICT_CONTRACT.items():
        if not isinstance(body[field], kind):
            return f"{field} has type {type(body[field]).__name__}"
    if body["predicted_status"] not in {"NORMAL", "WARNING", "CRITICAL"}:
        return f"invalid predicted_status {body['predicted_status']}"
    if not 0 <= body["confidence"] <= 1:
        return f"confidence out of [0, 1]: {body['confidence']}"
    if body["robot_id"] != SAMPLE["robot_id"]:
        return "robot_id not echoed"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--final", action="store_true",
                   help="also require /predict (exact contract), history, dashboard and 4xx on invalid data")
    args = p.parse_args()
    base = args.api.rstrip("/")

    checks = [
        ("GET /health", requests.get(base + "/health", timeout=60)),
        ("POST /telemetry", requests.post(base + "/telemetry", json=SAMPLE, timeout=10)),
        ("GET /robots", requests.get(base + "/robots", timeout=10)),
        ("GET /metrics", requests.get(base + "/metrics", timeout=10)),
    ]

    failed = False
    for name, response in checks:
        print(f"{name:<20} -> {response.status_code}")
        if not response.ok:
            failed = True

    if args.final:
        history = requests.get(f"{base}/robots/{SAMPLE['robot_id']}/history", timeout=10)
        print(f"{'GET /history':<20} -> {history.status_code}")
        failed |= not history.ok

        dashboard = requests.get(base + "/dashboard", timeout=10)
        print(f"{'GET /dashboard':<20} -> {dashboard.status_code}")
        failed |= not dashboard.ok

        problem = check_predict_contract(requests.post(base + "/predict", json=SAMPLE, timeout=10))
        print(f"{'POST /predict':<20} -> {'contract OK' if problem is None else 'FAIL: ' + problem}")
        failed |= problem is not None

        invalid = {
            "missing field": {k: v for k, v in SAMPLE.items() if k != "battery"},
            "out of range": {**SAMPLE, "battery": 150},
            "NaN": '{"robot_id": "TEST-01", "timestamp": 1, "x": 1, "y": 1, "speed": NaN, "battery": 50,'
                   ' "motor_temp": 40, "vibration": 0.3, "motor_current": 3, "obstacle_distance": 1}',
        }
        for label, payload in invalid.items():
            for route in ("/telemetry", "/predict"):
                if isinstance(payload, str):
                    r = requests.post(base + route, data=payload, timeout=10,
                                      headers={"content-type": "application/json"})
                else:
                    r = requests.post(base + route, json=payload, timeout=10)
                ok = 400 <= r.status_code < 500
                print(f"{'invalid ' + label:<20} {route:<11} -> {r.status_code}{'' if ok else '  FAIL (expected 4xx)'}")
                failed |= not ok

        alive = requests.get(base + "/health", timeout=10)
        print(f"{'alive after invalid':<20} -> {alive.status_code}")
        failed |= not alive.ok

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
