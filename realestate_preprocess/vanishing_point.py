"""Estimación robusta de puntos de fuga (vanishing points) con RANSAC.

Un conjunto de rectas paralelas en el mundo (p.ej. todas las verticales de una
fachada) se cruza, en la imagen, en un único punto de fuga. Ese punto nos dice
hacia dónde "convergen" las verticales y es la base para enderezarlas.

Algoritmo:
  1. RANSAC: se toman 2 rectas al azar, se cruzan (producto vectorial) para
     obtener un candidato de punto de fuga, y se cuentan cuántas rectas apuntan
     a él (inliers).
  2. Con los inliers del mejor modelo, se refina por mínimos cuadrados (SVD):
     el punto de fuga es el vector que minimiza |L · vp| (dirección singular
     más chica de la matriz de rectas).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from .line_detection import LineSegment


def estimate_vanishing_point(
    segments: List[LineSegment],
    iterations: int = 500,
    inlier_thresh_deg: float = 2.0,
    min_lines: int = 6,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """Devuelve (vp_homogéneo[3], índices_inliers) o None si no hay datos."""
    if len(segments) < 2:
        return None

    lines = [s.homogeneous_line() for s in segments]
    mids = [s.midpoint for s in segments]
    dirs = [s.unit_direction() for s in segments]

    rng = random.Random(seed)
    thresh = np.deg2rad(inlier_thresh_deg)
    n = len(segments)
    best_inliers: List[int] = []

    for _ in range(iterations):
        i, j = rng.sample(range(n), 2)
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-9:
            continue
        inliers = _consensus(vp, mids, dirs, thresh)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Si el consenso quedó pobre, usamos todas las rectas para el refinamiento.
    if len(best_inliers) < min_lines and len(best_inliers) < max(2, n // 2):
        best_inliers = list(range(n))

    vp = _refine(lines, best_inliers)
    return vp, best_inliers


def _consensus(
    vp: np.ndarray, mids: List[np.ndarray], dirs: List[np.ndarray], thresh: float
) -> List[int]:
    """Índices de rectas cuyo ángulo respecto a la dirección al vp es < thresh."""
    inliers: List[int] = []
    finite = abs(vp[2]) >= 1e-9
    vp_xy = vp[:2] / vp[2] if finite else None
    for k, (mid, direction) in enumerate(zip(mids, dirs)):
        if finite:
            to_vp = vp_xy - mid
        else:
            to_vp = vp[:2]  # punto de fuga en el infinito -> dirección constante
        norm = np.linalg.norm(to_vp)
        if norm < 1e-9:
            inliers.append(k)
            continue
        to_vp = to_vp / norm
        cos_angle = min(1.0, abs(float(np.dot(to_vp, direction))))
        if np.arccos(cos_angle) < thresh:
            inliers.append(k)
    return inliers


def _refine(lines: List[np.ndarray], inliers: List[int]) -> np.ndarray:
    """Punto de fuga por mínimos cuadrados: vector singular más chico de L."""
    subset = [lines[i] for i in inliers] if len(inliers) >= 2 else lines
    matrix = np.asarray(subset, dtype=np.float64)
    _, _, vt = np.linalg.svd(matrix)
    return vt[-1]
