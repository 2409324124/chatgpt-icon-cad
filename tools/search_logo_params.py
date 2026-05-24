#!/usr/bin/env python3
"""Parallel search for ChatGPT-style logo path parameters.

This is a fast silhouette scorer, not a CAD exporter.  It evaluates candidate
2D logo footprints against reference/chatgpt_reference.png using all available
CPU cores, then prints the best parameter set to paste into chatgpt_icon.py.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

import compare_top_view as compare


PROJECT_ROOT = Path(__file__).absolute().parents[1]
REFERENCE = PROJECT_ROOT / "reference" / "chatgpt_reference.png"
REPORT = PROJECT_ROOT / "exports" / "param_search_report.json"

SIZE = 180
# Search grid viewport diameter in mm.  This controls the coordinate space for
# the fast silhouette scorer and is NOT the model base diameter (which is now
# computed dynamically from the logo bounding box + margin).
LOGO_VIEWPORT_DIAMETER = 80.0
MARGIN = 18

REF: np.ndarray
X: np.ndarray
Y: np.ndarray


SEEDS = [
    # Current best from manual iteration.
    [(3.5, -7.8), (9.5, -17.0), (21.0, -16.5), (27.5, -5.0), (25.0, 6.0), (14.0, 11.0), (5.0, 5.5)],
    # More official-like folded band: short inner segment, outside arc, return inward.
    [(4.0, -8.0), (8.0, -19.0), (19.0, -22.0), (29.0, -12.0), (27.0, 1.0), (17.0, 8.0), (6.0, 5.0)],
    [(4.0, -6.0), (12.0, -18.0), (24.0, -16.0), (29.0, -4.0), (23.0, 7.0), (12.0, 10.0), (4.5, 4.0)],
    [(5.0, -4.0), (16.0, -12.0), (27.0, -8.0), (28.0, 5.0), (17.0, 12.0), (7.0, 8.0), (4.0, 0.0)],
]


def _edge(mask: np.ndarray) -> np.ndarray:
    eroded = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(mask, (dy, dx), axis=(0, 1))
        if dy == 1:
            shifted[0, :] = False
        elif dy == -1:
            shifted[-1, :] = False
        elif dx == 1:
            shifted[:, 0] = False
        elif dx == -1:
            shifted[:, -1] = False
        eroded &= shifted
    return mask & ~eroded


def _morph_op(mask: np.ndarray, op: str) -> np.ndarray:
    """Morphological erode (op='&') or dilate (op='|') with 8-connected kernel."""
    result = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(mask, (dy, dx), axis=(0, 1))
            if dy == 1:
                shifted[0, :] = False
            elif dy == -1:
                shifted[-1, :] = False
            if dx == 1:
                shifted[:, 0] = False
            elif dx == -1:
                shifted[:, -1] = False
            if op == '&':
                result &= shifted
            else:
                result |= shifted
    return result


def _vector_quality_np(mask: np.ndarray) -> tuple[float, float]:
    """Numpy-based vector quality: straightness and curvature.

    Returns (straightness_score, curvature_score), matching the semantics of
    compare_top_view._vector_quality_scores:

    - straightness_score: perimeter ratio (smoothed / original).  Smooth edges
      change little (≈ 1.0); jagged pixel stair-steps get smoothed away,
      reducing the perimeter (< 1.0).
    - curvature_score: IoU between original and smoothed mask.  High IoU means
      edges were already smooth; low IoU means smoothing significantly altered
      the silhouette.

    Uses 8-connected morphological close-then-open as a proxy for vector cleanup.
    """
    # Close (dilate then erode): fills gaps, smooths stair-steps.
    closed = _morph_op(_morph_op(mask, '|'), '&')
    # Open (erode then dilate): removes thin protrusions.
    smoothed = _morph_op(_morph_op(closed, '&'), '|')

    # straightness_score: perimeter ratio (smoothed / original).
    edge_orig = _edge(mask)
    edge_smooth = _edge(smoothed)
    orig_perim = int(edge_orig.sum())
    smooth_perim = int(edge_smooth.sum())
    straightness = float(min(1.0, smooth_perim / orig_perim)) if orig_perim > 0 else 1.0

    # curvature_score: IoU between original and smoothed mask.
    inter = np.logical_and(mask, smoothed).sum()
    union = np.logical_or(mask, smoothed).sum()
    curvature = float(inter / union) if union else 0.0

    return straightness, curvature


def _segment_mask(lx: np.ndarray, ly: np.ndarray, start: tuple[float, float], end: tuple[float, float], width: float) -> np.ndarray:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return np.zeros_like(lx, dtype=bool)
    t = np.clip(((lx - sx) * dx + (ly - sy) * dy) / length_sq, 0.0, 1.0)
    cx = sx + t * dx
    cy = sy + t * dy
    return (lx - cx) ** 2 + (ly - cy) ** 2 <= (width / 2) ** 2


def _inside_hex(x: np.ndarray, y: np.ndarray, flat_diameter: float) -> np.ndarray:
    r = flat_diameter / 2
    return (np.abs(y) <= r) & (math.sqrt(3) * np.abs(x) + np.abs(y) <= 2 * r)


def _candidate_mask(params: dict[str, Any]) -> np.ndarray:
    points = params["points"]
    width = params["width"]
    start_angle = params["angle"]
    void = params["void"]

    mask = np.zeros_like(X, dtype=bool)
    for i in range(6):
        theta = math.radians(start_angle + i * 60.0)
        co = math.cos(theta)
        si = math.sin(theta)
        lx = X * co + Y * si
        ly = -X * si + Y * co
        lobe = np.zeros_like(mask)
        for start, end in zip(points, points[1:]):
            lobe |= _segment_mask(lx, ly, start, end, width)
        mask |= lobe
    mask &= ~_inside_hex(X, Y, void)
    return mask


def _score_mask(mask: np.ndarray) -> dict[str, float | int]:
    inter = np.logical_and(mask, REF).sum()
    union = np.logical_or(mask, REF).sum()
    iou = inter / union if union else 0.0
    area = int(mask.sum())
    ref_area = int(REF.sum())
    area_ratio = min(area, ref_area) / max(area, ref_area) if max(area, ref_area) else 0.0
    edge = _edge(mask)
    ref_edge = _edge(REF)
    edge_inter = np.logical_and(edge, ref_edge).sum()
    edge_union = np.logical_or(edge, ref_edge).sum()
    edge_iou = edge_inter / edge_union if edge_union else 0.0

    # Shape sub-score (weights sum to 1.0 within the shape group).
    shape_score = 0.62 * iou + 0.23 * area_ratio + 0.15 * edge_iou

    # Vector quality: straightness + curvature.
    straightness, curvature = _vector_quality_np(mask)
    vector_quality_score = 0.5 * straightness + 0.5 * curvature

    # Final score: shape (82 %) + vector quality (18 %) = 1.0.
    score = 0.82 * shape_score + 0.18 * vector_quality_score
    return {
        "score": float(score),
        "shape_score": float(shape_score),
        "vector_quality_score": float(vector_quality_score),
        "iou": float(iou),
        "area_ratio": float(area_ratio),
        "edge_iou": float(edge_iou),
        "straightness_score": float(straightness),
        "curvature_score": float(curvature),
        "area": area,
    }


def _valid(points: list[tuple[float, float]], width: float) -> bool:
    if width < 0.8:
        return False
    # Keep the logo within the search grid viewport.
    max_allowed_radius = LOGO_VIEWPORT_DIAMETER / 2 - 5.5
    if max(math.hypot(x, y) for x, y in points) + width / 2 > max_allowed_radius:
        return False
    for a, b in zip(points, points[1:]):
        if math.dist(a, b) <= width * 1.15:
            return False
    return True


def _mutate(seed: list[tuple[float, float]], rng: random.Random) -> dict[str, Any]:
    # Smooth-ish perturbation: keep point order but vary radius/angle and local jitter.
    scale_x = rng.uniform(0.82, 1.12)
    scale_y = rng.uniform(0.82, 1.12)
    shear = rng.uniform(-0.12, 0.12)
    points = []
    for x, y in seed:
        nx = x * scale_x + y * shear + rng.uniform(-2.4, 2.4)
        ny = y * scale_y + rng.uniform(-2.4, 2.4)
        points.append((nx, ny))
    width = rng.uniform(3.2, 5.8)
    return {
        "points": points,
        "width": width,
        "angle": rng.uniform(-8, 28),
        "void": rng.uniform(5.5, 10.0),
    }


def _worker(args: tuple[int, int, int]) -> list[dict[str, Any]]:
    worker_id, samples, keep = args
    rng = random.Random(10_000 + worker_id)
    best: list[dict[str, Any]] = []
    for _ in range(samples):
        seed = rng.choice(SEEDS)
        params = _mutate(seed, rng)
        if not _valid(params["points"], params["width"]):
            continue
        metrics = _score_mask(_candidate_mask(params))
        result = {**metrics, **params}
        best.append(result)
        if len(best) > keep * 4:
            best = sorted(best, key=lambda item: item["score"], reverse=True)[:keep]
    return sorted(best, key=lambda item: item["score"], reverse=True)[:keep]


def _init_globals() -> None:
    global REF, X, Y
    modules = compare._load_pillow()
    REF = np.array(compare._reference_logo_mask(REFERENCE, SIZE, modules)) > 0
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    scale = (SIZE - 2 * MARGIN) / LOGO_VIEWPORT_DIAMETER
    half = SIZE / 2
    X = ((xx + 0.5) - half) / scale
    Y = (half - (yy + 0.5)) / scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel search for better logo silhouette parameters.")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--samples", type=int, default=24000, help="Total random candidates to score")
    parser.add_argument("--keep", type=int, default=10, help="Best candidates to keep per worker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not REFERENCE.exists():
        raise SystemExit(f"Reference missing: {REFERENCE}")
    _init_globals()
    workers = max(1, args.workers)
    per_worker = max(1, math.ceil(args.samples / workers))
    jobs = [(i, per_worker, args.keep) for i in range(workers)]
    print(f"Searching {per_worker * workers} candidates with {workers} workers...")
    with mp.Pool(processes=workers, initializer=_init_globals) as pool:
        chunks = pool.map(_worker, jobs)
    all_best = [item for chunk in chunks for item in chunk]
    all_best.sort(key=lambda item: item["score"], reverse=True)
    top = all_best[:20]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8")
    for idx, item in enumerate(top[:8], 1):
        pts = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in item["points"])
        print(
            f"#{idx} score={item['score']:.4f} shape={item['shape_score']:.4f} "
            f"vq={item['vector_quality_score']:.4f} "
            f"iou={item['iou']:.4f} area={item['area_ratio']:.4f} edge={item['edge_iou']:.4f} "
            f"straight={item['straightness_score']:.4f} curve={item['curvature_score']:.4f} "
            f"width={item['width']:.2f} angle={item['angle']:.2f} void={item['void']:.2f}"
        )
        print(f"   points=({pts})")
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
