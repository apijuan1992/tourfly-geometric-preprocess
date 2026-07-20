"""Interfaz de línea de comandos.

Uso:
    python -m realestate_preprocess entrada.jpg --after despues.jpg --before antes.jpg
    python -m realestate_preprocess entrada.jpg -o salida/   # guarda *_antes / *_despues
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from .config import PreprocessConfig
from .pipeline import process_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realestate_preprocess",
        description="Nivela, endereza verticales y recorta fotos inmobiliarias.",
    )
    parser.add_argument("input", help="Ruta de la imagen de entrada")
    parser.add_argument("--after", help="Ruta de salida del 'después'")
    parser.add_argument("--before", help="Ruta de salida del 'antes'")
    parser.add_argument(
        "-o", "--outdir",
        help="Carpeta de salida (genera <nombre>_antes.jpg y <nombre>_despues.jpg)",
    )
    parser.add_argument("--max-correction", type=float, default=None,
                        help="Rotación máxima aplicada, en grados (def. 12)")
    parser.add_argument("--focal-px", type=float, default=None,
                        help="Focal en píxeles (fuerza el valor, ignora EXIF)")
    parser.add_argument("--no-crop", action="store_true", help="No recortar bordes vacíos")
    parser.add_argument("--no-perspective", action="store_true",
                        help="Sólo nivelar, sin corregir perspectiva")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    config = PreprocessConfig()
    if args.max_correction is not None:
        config.max_correction_deg = args.max_correction
    if args.focal_px is not None:
        config.focal_px = args.focal_px
    if args.no_crop:
        config.enable_min_crop = False
    if args.no_perspective:
        config.enable_perspective = False

    before_path = args.before
    after_path = args.after
    if args.outdir:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        before_path = before_path or os.path.join(args.outdir, f"{stem}_antes.jpg")
        after_path = after_path or os.path.join(args.outdir, f"{stem}_despues.jpg")
    if not after_path:
        stem, ext = os.path.splitext(args.input)
        after_path = f"{stem}_despues{ext or '.jpg'}"

    result = process_file(args.input, before_path, after_path, config)

    print(f"Verticales detectadas:   {result.num_vertical_lines}")
    print(f"Horizontales detectadas: {result.num_horizontal_lines}")
    print(f"Corrección aplicada:     {result.correction_angle_deg:.2f}°")
    print(f"Recorte:                 {result.crop_rect}")
    for note in result.notes:
        print(f"  - {note}")
    if after_path:
        print(f"Guardado 'después': {after_path}")
    if before_path:
        print(f"Guardado 'antes':   {before_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
