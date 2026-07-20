"""Entrada/salida de imágenes y lectura de focal desde EXIF.

Aislado en su propio módulo para que el resto del pipeline trabaje siempre
sobre arrays de numpy (BGR de OpenCV) sin conocer el origen (disco, request
HTTP, bytes en memoria, etc.).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def _pil_to_bgr(pil_image) -> np.ndarray:
    """Aplica la orientación EXIF y convierte a BGR (formato de OpenCV).

    Las cámaras/celulares guardan la foto sin rotar + un flag EXIF de
    orientación. OpenCV ignora ese flag, así que sin esto una foto vertical se
    procesaría de costado. Pillow (ImageOps.exif_transpose) lo resuelve.
    """
    from PIL import ImageOps

    pil_image = ImageOps.exif_transpose(pil_image)
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image(path: str) -> np.ndarray:
    """Lee una imagen de disco como BGR, respetando la orientación EXIF."""
    try:
        from PIL import Image

        with Image.open(path) as pil_image:
            return _pil_to_bgr(pil_image)
    except Exception:
        # Fallback sin Pillow (o formato que PIL no abre): OpenCV directo.
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        return image


def decode_image(data: bytes) -> np.ndarray:
    """Decodifica una imagen desde bytes (API), respetando la orientación EXIF."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as pil_image:
            return _pil_to_bgr(pil_image)
    except Exception:
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("No se pudieron decodificar los bytes de la imagen")
        return image


def save_image(path: str, image: np.ndarray, jpeg_quality: int = 95) -> None:
    """Guarda una imagen creando la carpeta destino si hace falta."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    if not cv2.imwrite(path, image, params):
        raise IOError(f"No se pudo escribir la imagen: {path}")


def read_focal_px_from_exif(path: str, image_width: int) -> Optional[float]:
    """Estima la focal en píxeles desde EXIF.

    Usa `FocalLengthIn35mmFilm` cuando está disponible:
        f_px ≈ (f35 / 36 mm) * ancho_imagen_px
    Requiere Pillow (opcional). Devuelve None si no se puede determinar.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except Exception:
        return None

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            data = {TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}
            f35 = data.get("FocalLengthIn35mmFilm")
            if f35:
                return float(f35) / 36.0 * float(image_width)
    except Exception:
        return None
    return None
