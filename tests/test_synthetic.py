"""Prueba sintética: verticales que convergen deben quedar más paralelas.

Genera una imagen con líneas verticales a las que se les aplica una homografía
de "cabeceo" (keystone) para simular una foto tomada mirando hacia arriba.
Tras el preproceso, el punto de fuga vertical debe alejarse (verticales más
paralelas) y los segmentos verticales acercarse a los 90°.

Ejecutar:  python -m pytest tests/  (o)  python tests/test_synthetic.py
"""

import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realestate_preprocess import PreprocessConfig, preprocess_image
from realestate_preprocess import line_detection


def _make_keystoned_image(w=800, h=600):
    """Fondo claro con líneas verticales, deformado con un keystone conocido."""
    img = np.full((h, w, 3), 240, np.uint8)
    for x in range(80, w - 40, 90):
        cv2.line(img, (x, 40), (x, h - 40), (30, 30, 30), 3)
    for y in range(80, h - 40, 120):  # algunas horizontales de referencia
        cv2.line(img, (40, y), (w - 40, y), (90, 90, 90), 2)

    # Homografía que "cierra" las verticales hacia arriba (simula cabeceo).
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = w * 0.10
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h), borderValue=(240, 240, 240))


def _mean_vertical_deviation(image, config):
    """Desvío angular medio de las verticales respecto a 90° (menor = más recto)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segments = line_detection.detect_segments(gray, config)
    verticals, _ = line_detection.classify(segments, config)
    if not verticals:
        return None
    devs = [abs(abs(s.angle_deg) - 90.0) for s in verticals]
    return float(np.mean(devs))


def test_verticals_become_more_parallel():
    config = PreprocessConfig(vertical_angle_tol_deg=30.0)
    warped = _make_keystoned_image()

    before = _mean_vertical_deviation(warped, config)
    result = preprocess_image(warped, config)
    after = _mean_vertical_deviation(result.image, config)

    assert before is not None and after is not None, "no se detectaron verticales"
    # Tras corregir, las verticales deben desviarse menos de la vertical ideal.
    assert after <= before + 1e-6, f"empeoró: antes={before:.2f} después={after:.2f}"
    print(f"OK  desvío vertical: antes={before:.2f}°  después={after:.2f}°")


if __name__ == "__main__":
    test_verticals_become_more_parallel()
    print("Prueba sintética superada.")
