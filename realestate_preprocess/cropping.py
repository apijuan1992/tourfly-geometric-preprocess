"""Recorte mínimo: el mayor rectángulo válido dentro de la imagen transformada.

Tras enderezar, pueden quedar cuñas o bordes vacíos (negros). Buscamos el
rectángulo axis-aligned de MAYOR ÁREA que caiga completamente dentro de la zona
válida (la máscara). Ese es el recorte mínimo necesario: elimina sólo los bordes
vacíos, sin agrandar el contenido (no hay zoom).

Se usa el algoritmo clásico de "mayor rectángulo en un histograma" aplicado fila
por fila. Para no pagar O(W·H) en Python sobre imágenes de 12+ MP, la búsqueda
corre sobre una versión reducida de la máscara y el rectángulo se escala de
vuelta con un pequeño margen hacia adentro (evita reintroducir borde negro).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def largest_interior_rectangle(
    mask: np.ndarray, max_dim: int = 1200
) -> Tuple[int, int, int, int]:
    """Devuelve (x, y, w, h) del mayor rectángulo válido dentro de la máscara."""
    h, w = mask.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(
            mask, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        small = mask

    x, y, rw, rh = _largest_rect(small)

    if scale < 1.0:
        inv = 1.0 / scale
        margin = 2  # margen de seguridad hacia adentro (px del tamaño real)
        x = int(round(x * inv)) + margin
        y = int(round(y * inv)) + margin
        rw = int(round(rw * inv)) - 2 * margin
        rh = int(round(rh * inv)) - 2 * margin

    x = max(0, x)
    y = max(0, y)
    rw = max(0, min(rw, w - x))
    rh = max(0, min(rh, h - y))
    return x, y, rw, rh


def crop_to_rect(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return image
    return image[y:y + h, x:x + w]


def _largest_rect(mask: np.ndarray) -> Tuple[int, int, int, int]:
    binary = (mask > 0).astype(np.int32)
    rows, cols = binary.shape
    heights = np.zeros(cols, dtype=np.int32)
    best = (0, 0, 0, 0)
    best_area = 0
    for row in range(rows):
        heights = np.where(binary[row] > 0, heights + 1, 0)
        x, width, height = _largest_in_histogram(heights)
        area = width * height
        if area > best_area:
            best_area = area
            best = (x, row - height + 1, width, height)
    return best


def _largest_in_histogram(heights: np.ndarray) -> Tuple[int, int, int]:
    """Mayor rectángulo bajo un histograma. Devuelve (x, ancho, alto)."""
    stack = []  # (índice_inicio, altura)
    best = (0, 0, 0)
    best_area = 0
    n = len(heights)
    for i in range(n + 1):
        cur = int(heights[i]) if i < n else 0
        start = i
        while stack and stack[-1][1] > cur:
            s, hh = stack.pop()
            area = hh * (i - s)
            if area > best_area:
                best_area = area
                best = (s, i - s, hh)
            start = s
        stack.append((start, cur))
    return best
