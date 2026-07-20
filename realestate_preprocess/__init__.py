"""Preproceso geométrico de fotografías inmobiliarias con OpenCV.

API pública del paquete:
    from realestate_preprocess import PreprocessConfig, preprocess_image, process_file
"""

from .config import PreprocessConfig
from .pipeline import PreprocessResult, preprocess_image, process_file

__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_image",
    "process_file",
]

__version__ = "0.1.0"
