"""Modelo de cámara: matriz de intrínsecos K y estimación de la focal.

La focal en píxeles es necesaria para pasar del plano imagen al espacio 3D de
la cámara (K^-1) y para razonar sobre direcciones (verticales del mundo). Se
resuelve, en orden de prioridad:
  1. focal_px forzada en la config.
  2. EXIF (FocalLengthIn35mmFilm).
  3. Heurística: focal_fallback_ratio * lado_mayor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import PreprocessConfig
from .image_io import read_focal_px_from_exif


def build_intrinsics(width: int, height: int, focal_px: float) -> np.ndarray:
    """Matriz K asumiendo píxeles cuadrados y centro óptico en el medio."""
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resolve_focal_px(
    image_path: Optional[str], width: int, height: int, config: PreprocessConfig
) -> float:
    if config.focal_px:
        return float(config.focal_px)
    if image_path:
        exif_focal = read_focal_px_from_exif(image_path, width)
        if exif_focal:
            return exif_focal
    return config.focal_fallback_ratio * float(max(width, height))
