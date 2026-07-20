"""Orquestador del preproceso. Punto de entrada para la futura API.

Orden de las etapas (cada una es opcional y aislada):
  1. Focal / intrínsecos de cámara.
  2. Corrección de distorsión de lente (sólo si hay coeficientes).
  3. Detección y clasificación de líneas.
  4. Enderezado: nivelado + verticales paralelas (una sola rotación válida).
     - Si no hay verticales pero sí horizontes, se nivela por roll (fallback).
  5. Recorte mínimo de los bordes vacíos.

`preprocess_image` trabaja sobre arrays (ideal para la API); `process_file`
envuelve la lectura/escritura desde disco y guarda "antes" y "después".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import camera, cropping, image_io, lens, line_detection, rectify, vanishing_point
from .config import PreprocessConfig


@dataclass
class PreprocessResult:
    image: np.ndarray                       # imagen final ("después")
    original: np.ndarray                    # copia sin tocar ("antes")
    homography: Optional[np.ndarray]        # H aplicada (None si no hubo)
    correction_angle_deg: float             # magnitud del enderezado
    crop_rect: Optional[Tuple[int, int, int, int]]
    num_vertical_lines: int
    num_horizontal_lines: int
    lens_corrected: bool
    notes: List[str] = field(default_factory=list)


def preprocess_image(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
    image_path: Optional[str] = None,
) -> PreprocessResult:
    """Preprocesa una imagen BGR (numpy) y devuelve el resultado con metadatos."""
    config = config or PreprocessConfig()
    original = image.copy()
    notes: List[str] = []
    height, width = image.shape[:2]

    # 1) Cámara
    focal = camera.resolve_focal_px(image_path, width, height, config)
    K = camera.build_intrinsics(width, height, focal)

    # 2) Distorsión de lente (opcional)
    lens_corrected = False
    if config.dist_coeffs is not None:
        image, K = lens.undistort(image, K, config.dist_coeffs)
        lens_corrected = True
        notes.append("Distorsión de lente corregida con los coeficientes provistos.")
    else:
        notes.append("Sin coeficientes de lente: se omite la corrección de distorsión.")

    # 3) Líneas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = line_detection.detect_segments(gray, config)
    verticals, horizontals = line_detection.classify(segments, config)

    # 4) Enderezado
    H: Optional[np.ndarray] = None
    angle = 0.0
    mask: Optional[np.ndarray] = None

    if config.enable_perspective and len(verticals) >= 2:
        vp_result = vanishing_point.estimate_vanishing_point(
            verticals,
            iterations=config.ransac_iterations,
            inlier_thresh_deg=config.ransac_inlier_thresh_deg,
            min_lines=config.min_lines_for_vp,
            seed=config.ransac_seed,
        )
        if vp_result is not None:
            vp_vertical, _ = vp_result
            H, angle = rectify.upright_homography(vp_vertical, K, config.max_correction_deg)
            # Nivelar la línea de la losa / horizontales dominantes (foto de frente):
            # tras enderezar las verticales, si el borde del techo o el piso quedan
            # inclinados, se aplica un roll extra para dejarlos horizontales. Se
            # auto-cancela si las horizontales no son consistentes (no es de frente).
            if config.enable_leveling:
                extra_roll = rectify.residual_roll_from_horizontals(H, horizontals)
                extra_roll = max(-config.max_correction_deg, min(config.max_correction_deg, extra_roll))
                if abs(extra_roll) > 0.1:
                    H = rectify.compose_roll(H, (height, width), extra_roll)
                    angle += abs(extra_roll)
                    notes.append(f"Horizontales (losa/piso) niveladas (roll extra {extra_roll:.2f}°).")
            image, mask = rectify.warp_to_fit(image, H)
            notes.append(
                f"Horizonte nivelado y verticales corregidas (rotación {angle:.2f}°)."
            )
    elif config.enable_leveling and len(horizontals) >= 2:
        roll = rectify.roll_from_horizontals(horizontals)
        if abs(roll) > 0.1:
            H = rectify.roll_homography(width, height, roll, K)
            image, mask = rectify.warp_to_fit(image, H)
            angle = abs(roll)
            notes.append(f"Horizonte nivelado por roll ({roll:.2f}°). Sin datos para perspectiva.")

    if H is None:
        notes.append("No se detectaron líneas suficientes; la imagen no se transformó.")

    # 5) Recorte mínimo
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    if config.enable_min_crop and mask is not None:
        rect = cropping.largest_interior_rectangle(mask, config.crop_search_max_dim)
        x, y, w, h = rect
        if w > 0 and h > 0 and (w < width or h < height):
            image = cropping.crop_to_rect(image, rect)
            crop_rect = rect
            notes.append(f"Recorte mínimo aplicado: {w}x{h} px (origen {width}x{height}).")

    return PreprocessResult(
        image=image,
        original=original,
        homography=H,
        correction_angle_deg=angle,
        crop_rect=crop_rect,
        num_vertical_lines=len(verticals),
        num_horizontal_lines=len(horizontals),
        lens_corrected=lens_corrected,
        notes=notes,
    )


def process_file(
    input_path: str,
    before_path: Optional[str] = None,
    after_path: Optional[str] = None,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Lee de disco, preprocesa y guarda 'antes'/'después' para comparar."""
    config = config or PreprocessConfig()
    image = image_io.load_image(input_path)
    result = preprocess_image(image, config, image_path=input_path)
    if before_path:
        image_io.save_image(before_path, result.original, config.jpeg_quality)
    if after_path:
        image_io.save_image(after_path, result.image, config.jpeg_quality)
    return result
