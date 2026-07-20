"""Corrección de distorsión de lente (opcional, basada en calibración).

IMPORTANTE: corregir bien la distorsión requiere los coeficientes de la lente
(k1, k2, p1, p2, k3), que se obtienen calibrando la cámara con un patrón de
ajedrez o de un perfil de lente conocido. Estimarlos desde una sola foto es
poco confiable y puede introducir curvaturas falsas, así que NO se adivina:
si no hay coeficientes, esta etapa se saltea sin tocar la imagen.

Con `alpha=1` en getOptimalNewCameraMatrix conservamos todo el campo visual
(sin zoom); los bordes vacíos que aparezcan los limpia el recorte mínimo.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def undistort(
    image: np.ndarray, K: np.ndarray, dist_coeffs: Optional[Sequence[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (imagen_corregida, K_nueva). Si no hay coeficientes, no cambia nada."""
    if dist_coeffs is None:
        return image, K
    d = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    h, w = image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, d, (w, h), alpha=1)
    corrected = cv2.undistort(image, K, d, None, new_K)
    return corrected, new_K
