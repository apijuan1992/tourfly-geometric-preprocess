"""Enderezado geométrico: nivelado + verticales paralelas.

Idea central (y por qué NO deforma la escena):

Una foto torcida o "cabeceada" equivale a haber rotado la cámara. Si estimamos
esa rotación R (a partir del punto de fuga vertical) y aplicamos la homografía
de rotación pura:

        H = K · R · K⁻¹

estamos re-renderizando la MISMA escena como si la cámara hubiese estado
perfectamente a nivel. Una homografía de rotación:
  * mantiene las rectas rectas (no curva nada),
  * corresponde a un movimiento físico real de la cámara (no inventa geometría),
  * no agranda ni achica el contenido (no hay zoom): sólo lo re-proyecta.

Con una sola rotación que lleve la dirección "vertical del mundo" a la vertical
de la imagen conseguimos, a la vez, nivelar el horizonte y dejar las verticales
paralelas. Ese es exactamente el objetivo.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .line_detection import LineSegment


def upright_homography(
    vp_vertical: np.ndarray, K: np.ndarray, max_correction_deg: float = 12.0
) -> Tuple[np.ndarray, float]:
    """Homografía que envía el punto de fuga vertical al infinito (arriba).

    Devuelve (H, ángulo_de_corrección_en_grados). El ángulo se recorta a
    max_correction_deg como salvaguarda contra warps agresivos.
    """
    K_inv = np.linalg.inv(K)

    # Dirección de la vertical del mundo en coordenadas de cámara.
    u = K_inv @ vp_vertical
    u = u / np.linalg.norm(u)

    # Objetivo: que esa dirección apunte "hacia arriba" en la imagen -> [0,-1,0].
    target = np.array([0.0, -1.0, 0.0])
    if np.dot(u, target) < 0:
        u = -u

    R_full = _rotation_between(u, target)

    # Recorta la magnitud de la rotación (en eje-ángulo) por seguridad.
    rvec, _ = cv2.Rodrigues(R_full)
    angle_deg = math.degrees(float(np.linalg.norm(rvec)))
    if angle_deg > max_correction_deg and angle_deg > 1e-9:
        rvec = rvec * (math.radians(max_correction_deg) / math.radians(angle_deg))
        R, _ = cv2.Rodrigues(rvec)
    else:
        R = R_full

    H = K @ R @ K_inv
    applied_deg = min(angle_deg, max_correction_deg)
    return H, applied_deg


def residual_roll_from_horizontals(
    H: np.ndarray,
    horizontals: list,
    min_lines: int = 4,
    max_spread_deg: float = 6.0,
) -> float:
    """Inclinación (grados) que le queda a las horizontales dominantes DESPUÉS de H.

    Tras enderezar las verticales, la línea de la losa / el piso (horizontales
    fuertes) deberían quedar horizontales en una foto de frente. Esta función
    mide su ángulo residual para nivelarlas. Devuelve 0 si hay pocas o si están
    inconsistentes (mucho spread = la foto NO es de frente: las horizontales
    convergen por profundidad/yaw) — así NO se fuerza ni se deforma.
    """
    if len(horizontals) < min_lines:
        return 0.0
    angles = []
    weights = []
    for s in horizontals:
        p1 = H @ np.array([s.x1, s.y1, 1.0])
        p2 = H @ np.array([s.x2, s.y2, 1.0])
        if abs(p1[2]) < 1e-9 or abs(p2[2]) < 1e-9:
            continue
        p1 = p1[:2] / p1[2]
        p2 = p2[:2] / p2[2]
        a = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        while a > 90.0:
            a -= 180.0
        while a < -90.0:
            a += 180.0
        angles.append(a)
        weights.append(s.length)
    if len(angles) < min_lines:
        return 0.0
    angles_arr = np.array(angles)
    # ángulo dominante ponderado por longitud (líneas largas = más confiables)
    order = np.argsort(np.array(weights))[::-1]
    top = angles_arr[order[: max(min_lines, len(order) // 2)]]
    dominant = float(np.median(top))
    spread = float(np.median(np.abs(angles_arr - dominant)))  # MAD
    if spread > max_spread_deg:
        return 0.0  # horizontales inconsistentes → no forzar (no es foto de frente)
    return -dominant  # rotar para llevarlas a 0°


def compose_roll(H: np.ndarray, image_shape, roll_deg: float) -> np.ndarray:
    """Compone un roll (rotación en el plano, alrededor del centro) sobre H."""
    if abs(roll_deg) < 1e-3:
        return H
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(roll_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    M = np.array(
        [
            [cos_t, -sin_t, cx - cos_t * cx + sin_t * cy],
            [sin_t, cos_t, cy - sin_t * cx - cos_t * cy],
            [0.0, 0.0, 1.0],
        ]
    )
    return M @ H


def roll_from_horizontals(horizontals: list) -> float:
    """Ángulo de inclinación (roll) a partir de la mediana de las horizontales.

    Fallback para nivelar el horizonte cuando no hay verticales suficientes.
    """
    angles = []
    for s in horizontals:
        a = math.degrees(math.atan2(s.y2 - s.y1, s.x2 - s.x1))
        # Normaliza a [-90, 90]: una horizontal y su opuesta son lo mismo.
        while a > 90.0:
            a -= 180.0
        while a < -90.0:
            a += 180.0
        angles.append(a)
    if not angles:
        return 0.0
    return float(np.median(angles))


def roll_homography(width: int, height: int, roll_deg: float, K: np.ndarray) -> np.ndarray:
    """Rotación pura alrededor del eje óptico (roll) — nivela sin deformar."""
    theta = math.radians(-roll_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K @ R @ np.linalg.inv(K)


def warp_to_fit(
    image: np.ndarray, H: np.ndarray, max_canvas_scale: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica H expandiendo el lienzo para NO perder contenido (sin zoom).

    Calcula a dónde van las 4 esquinas, traslada el resultado a (0,0) y usa un
    lienzo del tamaño del contenido transformado. Así una corrección grande NO
    empuja la foto fuera del cuadro (esa era la causa de que se viera "a la
    mitad"). Los bordes vacíos se recortan después con el recorte mínimo.
    `max_canvas_scale` acota el lienzo para ángulos extremos (memoria).

    Devuelve (imagen_transformada, máscara de píxeles válidos).
    """
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    min_xy = warped_corners.min(axis=0)
    max_xy = warped_corners.max(axis=0)

    out_w = int(np.ceil(max_xy[0] - min_xy[0]))
    out_h = int(np.ceil(max_xy[1] - min_xy[1]))
    out_w = max(1, min(out_w, int(w * max_canvas_scale)))
    out_h = max(1, min(out_h, int(h * max_canvas_scale)))

    # Traslación que lleva la esquina superior-izquierda del contenido a (0,0).
    T = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]])
    H_shifted = T @ H

    warped = cv2.warpPerspective(
        image, H_shifted, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    white = np.full((h, w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(white, H_shifted, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return warped, mask


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotación mínima que lleva el vector unitario a -> b (fórmula de Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        # Vectores casi paralelos (c~1) o casi opuestos (c~-1).
        if c > 0:
            return np.eye(3)
        # Caso opuesto: rota 180° alrededor de cualquier eje perpendicular a `a`.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis = axis / np.linalg.norm(axis)
        rvec = axis * math.pi
        R, _ = cv2.Rodrigues(rvec)
        return R
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
