"""Client-side load test of POST /predict (what the evaluation measures).

Sends N requests with C concurrent workers using realistic samples from the
official simulator and reports p50/p95/max latency, throughput and errors.

    python tests/load_test.py --api http://127.0.0.1:8000 --requests 500 --concurrency 20
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
from core import make_robot, step_robot  # noqa: E402  (official simulator, unchanged)

_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    url = args.api.rstrip("/") + "/predict"

    rng = random.Random(args.seed)
    robots = [make_robot(i + 1, rng) for i in range(10)]
    samples = [step_robot(robots[i % len(robots)], rng)[0] for i in range(args.requests)]
    for s in samples:
        s["robot_id"] = "L" + s["robot_id"]  # unseen IDs

    requests.get(args.api.rstrip("/") + "/health", timeout=60)  # wake up a sleeping service

    def call(sample: dict) -> tuple[float, int]:
        t0 = time.perf_counter()
        try:
            r = session().post(url, json=sample, timeout=15)
            code = r.status_code
        except requests.RequestException:
            code = 0
        return (time.perf_counter() - t0) * 1000, code

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, samples))
    elapsed = time.perf_counter() - start

    latencies = sorted(ms for ms, code in results if code == 200)
    errors = sum(1 for _, code in results if code != 200)
    q = statistics.quantiles(latencies, n=100) if len(latencies) > 1 else [latencies[0]] * 99
    print(f"requests={len(results)} concurrency={args.concurrency} errors={errors}")
    print(f"p50={q[49]:.1f} ms  p95={q[94]:.1f} ms  max={latencies[-1]:.1f} ms  "
          f"throughput={len(results) / elapsed:.0f} req/s")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
