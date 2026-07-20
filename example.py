"""Ejemplo de uso del módulo.

    python example.py mi_foto.jpg

Genera `mi_foto_antes.jpg` y `mi_foto_despues.jpg` al lado del original.
También muestra cómo usar el pipeline desde memoria (como haría una API).
"""

import os
import sys

from realestate_preprocess import PreprocessConfig, process_file


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python example.py <ruta_imagen>")
        return 1

    input_path = sys.argv[1]
    stem, ext = os.path.splitext(input_path)
    before_path = f"{stem}_antes{ext}"
    after_path = f"{stem}_despues{ext}"

    config = PreprocessConfig(
        max_correction_deg=12.0,   # límite de enderezado
        enable_min_crop=True,      # recorte mínimo de bordes vacíos
        # dist_coeffs=(-0.12, 0.03, 0, 0, 0),  # descomentar si tenés calibración
    )

    result = process_file(input_path, before_path, after_path, config)

    print(f"Corrección: {result.correction_angle_deg:.2f}°  |  recorte: {result.crop_rect}")
    for note in result.notes:
        print(" -", note)
    print(f"Antes:   {before_path}")
    print(f"Después: {after_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
