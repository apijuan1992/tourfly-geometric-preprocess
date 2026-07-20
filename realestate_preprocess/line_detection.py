"""Detección y clasificación de segmentos de recta.

Detecta segmentos con LSD (Line Segment Detector) y, si no está disponible en
el build de OpenCV, cae a HoughLinesP sobre bordes de Canny. Después clasifica
cada segmento en vertical / horizontal según su ángulo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import PreprocessConfig


@dataclass
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def angle_deg(self) -> float:
        """Ángulo del segmento en grados, en (-180, 180]."""
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def length(self) -> float:
        return float(math.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def midpoint(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0])

    def homogeneous_line(self) -> np.ndarray:
        """Recta que contiene al segmento en coordenadas homogéneas (l = p1 × p2)."""
        p1 = np.array([self.x1, self.y1, 1.0])
        p2 = np.array([self.x2, self.y2, 1.0])
        return np.cross(p1, p2)

    def unit_direction(self) -> np.ndarray:
        d = np.array([self.x2 - self.x1, self.y2 - self.y1])
        n = np.linalg.norm(d)
        return d / n if n > 0 else d


def detect_segments(gray: np.ndarray, config: PreprocessConfig) -> List[LineSegment]:
    """Devuelve segmentos con longitud >= min_line_length_ratio * lado_mayor."""
    min_len = config.min_line_length_ratio * max(gray.shape[:2])
    segments = _detect_with_lsd(gray)
    if not segments:
        segments = _detect_with_hough(gray, config, min_len)
    return [s for s in segments if s.length >= min_len]


def _to_segment(line) -> LineSegment:
    """Aplana cualquier forma que devuelva OpenCV ((1,4), (4,), (N,1,4)...) a 4 números."""
    vals = np.asarray(line, dtype=np.float64).reshape(-1)
    return LineSegment(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _detect_with_lsd(gray: np.ndarray) -> List[LineSegment]:
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gray)[0]
    except Exception:
        return []
    if lines is None:
        return []
    return [_to_segment(line) for line in lines]


def _detect_with_hough(
    gray: np.ndarray, config: PreprocessConfig, min_len: float
) -> List[LineSegment]:
    edges = cv2.Canny(gray, config.canny_low, config.canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=int(min_len),
        maxLineGap=10,
    )
    if lines is None:
        return []
    return [_to_segment(line) for line in lines]


def classify(
    segments: List[LineSegment], config: PreprocessConfig
) -> Tuple[List[LineSegment], List[LineSegment]]:
    """Separa en (verticales, horizontales) según el ángulo del segmento."""
    verticals: List[LineSegment] = []
    horizontals: List[LineSegment] = []
    for s in segments:
        angle = abs(s.angle_deg) % 180.0  # [0, 180)
        if abs(angle - 90.0) <= config.vertical_angle_tol_deg:
            verticals.append(s)
        elif angle <= config.horizontal_angle_tol_deg or angle >= 180.0 - config.horizontal_angle_tol_deg:
            horizontals.append(s)
    return verticals, horizontals
