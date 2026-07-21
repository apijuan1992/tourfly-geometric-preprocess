"""Configuración del pipeline de preproceso.

Todo se controla desde un único dataclass `PreprocessConfig`, para que la
integración futura en una API sea directa (se construye desde el body del
request, variables de entorno, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class PreprocessConfig:
    # --- Detección de líneas ---
    canny_low: int = 50
    canny_high: int = 150
    # Longitud mínima de un segmento para ser tenido en cuenta (fracción del
    # lado mayor de la imagen). Filtra ruido y bordes cortos. En fotos reales el
    # detector parte las líneas largas en tramos; 0.02 conserva las estructurales
    # sin quedarse con puro ruido.
    min_line_length_ratio: float = 0.02
    # Tolerancia angular para clasificar un segmento como vertical / horizontal.
    vertical_angle_tol_deg: float = 20.0
    horizontal_angle_tol_deg: float = 20.0

    # --- Cámara / focal ---
    # Si no se puede leer de EXIF: focal_px = focal_fallback_ratio * max(W, H).
    focal_fallback_ratio: float = 1.15
    # Fuerza una focal específica en píxeles (tiene prioridad sobre EXIF).
    focal_px: Optional[float] = None

    # --- Corrección de perspectiva / nivelado ---
    enable_leveling: bool = True
    enable_perspective: bool = True
    # Límite de rotación aplicada (seguridad anti-deformación). Una foto no
    # suele estar más torcida que esto; recortar acá evita warps agresivos y
    # limita el daño si la detección se equivoca (gran angular, líneas curvas).
    max_correction_deg: float = 30.0

    # --- Distorsión de lente ---
    # (k1, k2, p1, p2, k3). Si es None NO se corrige (adivinarlo es riesgoso).
    dist_coeffs: Optional[Sequence[float]] = None

    # --- Recorte mínimo ---
    enable_min_crop: bool = True
    # Techo de resolución para buscar el rectángulo interior (velocidad).
    crop_search_max_dim: int = 1200

    # --- RANSAC del punto de fuga ---
    ransac_iterations: int = 500
    ransac_inlier_thresh_deg: float = 2.0
    min_lines_for_vp: int = 6
    ransac_seed: int = 0

    # --- Salida ---
    jpeg_quality: int = 95
