"""Area coverage of the fleet and optimal target points.

Arena: 10 x 10 m. The simulator wraps positions (x %= 10), so the arena is a
torus: leaving through one edge means entering through the opposite one.

1. Real coverage: share of 1 x 1 m cells visited by any robot in a time window.
2. Optimal points: with n robots, the positions that maximise coverage are the
   centroids of a centroidal Voronoi tessellation (CVT): every robot is
   responsible for an equal-area zone and sits at its centre, minimising the
   distance from any point of the arena to the nearest robot. We approximate
   the CVT with k-means over a dense uniform grid of the arena.
3. Optimal assignment: which robot should go to which point so that the total
   travelled distance is minimal (Hungarian algorithm, toroidal distance).
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from app import db

ARENA = 10.0
CELLS = 10  # 1 m cells, aligned with the dashboard floor tiles


@lru_cache(maxsize=32)
def optimal_points(n: int) -> tuple[np.ndarray, float]:
    """CVT centroids for n robots and the coverage radius (worst distance from
    a point of the arena to its assigned centroid)."""
    axis = (np.arange(60) + 0.5) * ARENA / 60
    grid = np.array([(x, y) for x in axis for y in axis])
    km = KMeans(n_clusters=n, n_init=4, random_state=0).fit(grid)
    centers = km.cluster_centers_
    radius = float(np.max(np.linalg.norm(grid - centers[km.labels_], axis=1)))
    return centers, radius


def toroidal_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    d = np.minimum(d, ARENA - d)
    return np.sqrt((d ** 2).sum(axis=-1))


def compute(window_s: float = 300, connected_s: float = 30) -> dict:
    now = time.time()
    heat = np.zeros((CELLS, CELLS), dtype=int)
    samples = 0
    for col, row, n in db.visited_cells(now - window_s, ARENA / CELLS):
        heat[min(max(row, 0), CELLS - 1), min(max(col, 0), CELLS - 1)] += n
        samples += n
    covered = int((heat > 0).sum())

    latest = [r for r in db.latest_per_robot(with_prediction=True) if now - r["received_at"] <= connected_s]
    assignment = []
    radius = None
    if latest:
        centers, radius = optimal_points(len(latest))
        positions = np.array([[r["x"] % ARENA, r["y"] % ARENA] for r in latest])
        cost = toroidal_distance(positions[:, None, :], centers[None, :, :])
        robot_idx, target_idx = linear_sum_assignment(cost)
        for i, j in zip(robot_idx, target_idx):
            speed = max(latest[i]["speed"], 0.05)
            assignment.append({
                "robot_id": latest[i]["robot_id"],
                "x": round(float(positions[i, 0]), 3),
                "y": round(float(positions[i, 1]), 3),
                "target_x": round(float(centers[j, 0]), 3),
                "target_y": round(float(centers[j, 1]), 3),
                "distance_m": round(float(cost[i, j]), 3),
                "eta_s": round(float(cost[i, j]) / speed, 1),
            })

    return {
        "arena_m": ARENA,
        "cell_m": ARENA / CELLS,
        "window_s": window_s,
        "samples_in_window": samples,
        "coverage_pct": round(100 * covered / CELLS ** 2, 1),
        "cells_covered": covered,
        "cells_total": CELLS ** 2,
        "heatmap": heat.tolist(),  # heatmap[row=y][col=x]
        "connected_robots": len(latest),
        "coverage_radius_m": round(radius, 3) if radius is not None else None,
        "total_assignment_distance_m": round(sum(a["distance_m"] for a in assignment), 3),
        "assignment": sorted(assignment, key=lambda a: a["robot_id"]),
    }
