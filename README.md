# realestate-preprocess

Preproceso geométrico de fotografías inmobiliarias con **OpenCV**, pensado para
correr **antes** de mandar la foto a un modelo de IA. Modular y listo para
integrarse en una API.

## Qué hace

1. **Nivela el horizonte** y **deja las verticales paralelas** con una única
   rotación de cámara físicamente válida (`H = K · R · K⁻¹`). Una homografía de
   rotación mantiene las rectas rectas, no inventa geometría y **no hace zoom**.
2. **Detecta las verticales** (paredes, columnas, marcos) y estima su punto de
   fuga con RANSAC para calcular esa rotación.
3. **Corrige la distorsión de lente** *si* le pasás los coeficientes de
   calibración (ver más abajo). Sin ellos, se saltea (no se adivina).
4. **Recorta lo mínimo**: busca el mayor rectángulo válido dentro de la imagen
   transformada y elimina sólo los bordes vacíos. No reescala (sin zoom).
5. Guarda **"antes"** y **"después"** para comparar.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido (CLI)

```bash
python -m realestate_preprocess entrada.jpg --before antes.jpg --after despues.jpg
# o dejando que genere los nombres:
python -m realestate_preprocess entrada.jpg -o salida/
```

## Uso como librería (para la API)

```python
from realestate_preprocess import PreprocessConfig, preprocess_image

# `image` es un array BGR de numpy (p.ej. cv2.imdecode de los bytes del request)
result = preprocess_image(image, PreprocessConfig())
salida = result.image          # imagen corregida (numpy BGR)
result.correction_angle_deg    # cuánto se enderezó
result.crop_rect               # recorte aplicado (x, y, w, h) o None
result.notes                   # explicación legible de cada etapa
```

`preprocess_image` trabaja en memoria (ideal para un endpoint); `process_file`
envuelve lectura/escritura desde disco.

## Configuración (`PreprocessConfig`)

| Campo | Default | Qué controla |
|-------|---------|--------------|
| `max_correction_deg` | 12.0 | Tope de enderezado (anti-deformación) |
| `enable_perspective` | True | Corregir convergencia de verticales |
| `enable_leveling` | True | Nivelar (fallback por roll si no hay verticales) |
| `enable_min_crop` | True | Recorte mínimo de bordes vacíos |
| `dist_coeffs` | None | `(k1,k2,p1,p2,k3)` de la lente; None = no corrige |
| `focal_px` | None | Focal en px (si no, EXIF o heurística) |

## Distorsión de lente — nota importante

Corregir bien la distorsión necesita los **coeficientes de calibración** de la
cámara/lente (se obtienen con un patrón de ajedrez, o de un perfil de lente).
Estimarlos desde una sola foto es poco confiable, así que el módulo **no los
adivina**: si los tenés, pasalos en `dist_coeffs` y se aplican; si no, esa etapa
se omite sin tocar la imagen.

## Estructura (modular)

```
realestate_preprocess/
  config.py           # PreprocessConfig
  image_io.py         # cargar/guardar + focal desde EXIF
  line_detection.py   # LSD / Hough + clasificación vertical/horizontal
  vanishing_point.py  # punto de fuga por RANSAC
  camera.py           # intrínsecos K y focal
  rectify.py          # homografía de enderezado + nivelado (núcleo)
  lens.py             # undistort (opcional, con coeficientes)
  cropping.py         # mayor rectángulo interior (recorte mínimo)
  pipeline.py         # orquestador + PreprocessResult  <-- entrada para la API
  cli.py              # línea de comandos
tests/test_synthetic.py
example.py
```

## Micro-servicio (API HTTP)

Para que Tourfly lo llame como un paso más del pipeline, el módulo se expone con
FastAPI (`service/app.py`). Corre en un contenedor (OpenCV **no** funciona en el
serverless de Node/Vercel): se despliega en Cloud Run / Fly / Railway / una VM.

Local:
```bash
pip install -r service/requirements.txt
uvicorn service.app:app --host 0.0.0.0 --port 8000
curl -F "file=@foto.jpg" http://localhost:8000/preprocess -o corregida.jpg
```

Con Docker:
```bash
docker build -f service/Dockerfile -t tourfly-preprocess .
docker run -p 8000:8000 tourfly-preprocess
```

Interfaz visual para probar: abrí **`http://localhost:8000/`** en el navegador
→ subís (o arrastrás) una foto y ves el **antes/después** lado a lado con los
datos de la corrección. (La API cruda también está en `/docs`.)

Endpoints:
- `GET /` → página HTML de prueba (antes/después)
- `GET /health` → `{"status":"ok"}`
- `POST /preprocess` (multipart `file=<imagen>`) → JPEG corregido.
  Params opcionales: `max_correction_deg`, `focal_px`, `enable_crop`,
  `enable_perspective`, `response=image|json`. Metadatos en headers
  (`X-Correction-Angle`, `X-Crop-Rect`, ...) o en el JSON si `response=json`.

## Prueba

```bash
python tests/test_synthetic.py      # verticales convergentes -> más paralelas
python tests/test_service.py        # endpoint /preprocess responde una imagen
```

## Límites conocidos

- El enderezado se **acota** a `max_correction_deg` para no deformar escenas
  raras; subilo si tenés fotos muy cabeceadas.
- Sin EXIF, la focal se **estima** con una heurística; para máxima precisión,
  pasá `focal_px` o los datos reales de la cámara.
- La distorsión de lente requiere calibración (ver arriba).
