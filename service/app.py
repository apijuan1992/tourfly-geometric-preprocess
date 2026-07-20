"""Micro-servicio FastAPI que expone el preproceso geométrico.

Recibe una imagen, la endereza/recorta con el módulo `realestate_preprocess`
y devuelve la imagen corregida. Pensado para que el pipeline de Tourfly lo
llame como un paso más (HTTP), sin acoplar Python al backend TypeScript.

Ejecutar en local:
    pip install -r service/requirements.txt
    uvicorn service.app:app --host 0.0.0.0 --port 8000

Probar:
    curl -F "file=@foto.jpg" http://localhost:8000/preprocess -o corregida.jpg
"""

from __future__ import annotations

import base64
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

# Permite ejecutar tanto como paquete (uvicorn service.app:app desde la raíz)
# como en contenedor con el paquete al lado.
try:
    from realestate_preprocess import PreprocessConfig, preprocess_image
    from realestate_preprocess.image_io import decode_image
except ModuleNotFoundError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from realestate_preprocess import PreprocessConfig, preprocess_image
    from realestate_preprocess.image_io import decode_image


app = FastAPI(title="Tourfly · Preproceso geométrico", version="0.1.0")

# Tope de resolución de trabajo. Una foto muy grande (6000px+) no entra en la
# RAM de una instancia chica y el proceso muere. Se achica al lado mayor = MAX_DIM
# antes de procesar (para un paso previo a la IA, esta resolución sobra). Subir
# GEOMETRIC_MAX_DIM cuando la instancia tenga más memoria.
MAX_DIM = int(os.environ.get("GEOMETRIC_MAX_DIM", "3000"))


def _fit_working_size(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIM:
        return image
    scale = MAX_DIM / float(longest)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Página simple para probar: subir foto y ver antes/después."""
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    """Chequeo de vida para el orquestador / load balancer."""
    return {"status": "ok"}


@app.post("/preprocess")
async def preprocess(
    file: UploadFile = File(..., description="Imagen a preprocesar"),
    max_correction_deg: Optional[float] = Form(None),
    focal_px: Optional[float] = Form(None),
    enable_crop: bool = Form(True),
    enable_perspective: bool = Form(True),
    response: str = Form("image", description="'image' (JPEG) o 'json' (base64 + metadatos)"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    try:
        image = decode_image(raw)  # respeta orientación EXIF
    except ValueError:
        raise HTTPException(status_code=415, detail="Formato de imagen no soportado")

    image = _fit_working_size(image)  # achica si es muy grande (memoria)

    config = PreprocessConfig(
        enable_min_crop=enable_crop,
        enable_perspective=enable_perspective,
    )
    if max_correction_deg is not None:
        config.max_correction_deg = max_correction_deg
    if focal_px is not None:
        config.focal_px = focal_px

    result = preprocess_image(image, config)

    ok, encoded = cv2.imencode(
        ".jpg", result.image, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    )
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la salida")
    out_bytes = encoded.tobytes()

    metadata = {
        "correction_angle_deg": round(result.correction_angle_deg, 3),
        "crop_rect": result.crop_rect,
        "num_vertical_lines": result.num_vertical_lines,
        "num_horizontal_lines": result.num_horizontal_lines,
        "lens_corrected": result.lens_corrected,
        "notes": result.notes,
    }

    if response == "json":
        return JSONResponse(
            {"image_base64": base64.b64encode(out_bytes).decode("ascii"), "metadata": metadata}
        )

    headers = {
        "X-Correction-Angle": str(metadata["correction_angle_deg"]),
        "X-Crop-Rect": ",".join(map(str, result.crop_rect)) if result.crop_rect else "none",
        "X-Vertical-Lines": str(result.num_vertical_lines),
        "X-Lens-Corrected": str(result.lens_corrected).lower(),
    }
    return Response(content=out_bytes, media_type="image/jpeg", headers=headers)


# --- Página de prueba (HTML autocontenido, sin dependencias externas) ---------
INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tourfly · Preproceso de fotos</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #0f0a1f; color: #eee; padding: 24px;
  }
  .wrap { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h1 span { color: #8b7cf0; }
  .sub { color: #9a92b5; margin: 0 0 24px; font-size: 14px; }
  .card { background: #1a1330; border: 1px solid #2a2148; border-radius: 14px; padding: 20px; }
  .drop {
    border: 2px dashed #3a2f66; border-radius: 12px; padding: 34px; text-align: center;
    cursor: pointer; transition: all .15s; color: #b9b2d6;
  }
  .drop:hover, .drop.over { border-color: #5b6cf0; background: #201842; color: #fff; }
  .controls { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; margin: 18px 0; font-size: 14px; }
  .controls label { display: flex; align-items: center; gap: 8px; color: #c9c2e4; }
  input[type=range] { accent-color: #5b6cf0; }
  input[type=checkbox] { accent-color: #5b6cf0; width: 16px; height: 16px; }
  button {
    background: #5b6cf0; color: #fff; border: 0; border-radius: 999px;
    padding: 11px 26px; font-weight: 600; font-size: 14px; cursor: pointer;
  }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 22px; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
  .pane { background: #120d26; border: 1px solid #2a2148; border-radius: 12px; overflow: hidden; }
  .pane h3 { margin: 0; padding: 10px 14px; font-size: 13px; color: #9a92b5; border-bottom: 1px solid #2a2148; }
  .pane img { width: 100%; display: block; }
  .meta { margin-top: 18px; font-size: 13px; color: #c9c2e4; line-height: 1.7; }
  .meta b { color: #fff; }
  .notes { margin: 8px 0 0; padding-left: 18px; color: #9a92b5; }
  .err { color: #ff8080; margin-top: 14px; }
  .muted { color: #6f688c; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Tourfly <span>· Preproceso</span></h1>
  <p class="sub">Subí una foto y compará el <b>antes</b> y el <b>después</b> (nivelado, verticales derechas y recorte mínimo).</p>

  <div class="card">
    <div id="drop" class="drop">
      <div id="dropText">Arrastrá una foto acá, o hacé click para elegirla</div>
      <input id="file" type="file" accept="image/*" style="display:none">
    </div>

    <div class="controls">
      <label>Corrección máx.: <input id="maxcorr" type="range" min="2" max="45" step="1" value="15"><span id="maxcorrVal">15°</span></label>
      <label><input id="crop" type="checkbox" checked> Recortar bordes vacíos</label>
      <label><input id="persp" type="checkbox" checked> Corregir perspectiva</label>
      <button id="go" disabled>Procesar</button>
      <span id="status" class="muted"></span>
    </div>

    <div id="err" class="err"></div>

    <div class="grid" id="results" style="display:none">
      <div class="pane"><h3>ANTES</h3><img id="before"></div>
      <div class="pane"><h3>DESPUÉS</h3><img id="after"></div>
    </div>

    <div id="meta" class="meta"></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let currentFile = null;

$("maxcorr").addEventListener("input", () => $("maxcorrVal").textContent = $("maxcorr").value + "\\u00b0");

const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]);
});
$("file").addEventListener("change", (e) => { if (e.target.files.length) pick(e.target.files[0]); });

function pick(file) {
  currentFile = file;
  $("dropText").textContent = file.name;
  $("go").disabled = false;
  $("before").src = URL.createObjectURL(file);
  $("after").removeAttribute("src");
  $("results").style.display = "grid";
  $("meta").innerHTML = "";
  $("err").textContent = "";
}

$("go").addEventListener("click", async () => {
  if (!currentFile) return;
  $("go").disabled = true;
  $("status").textContent = "Procesando...";
  $("err").textContent = "";
  try {
    const fd = new FormData();
    fd.append("file", currentFile);
    fd.append("max_correction_deg", $("maxcorr").value);
    fd.append("enable_crop", $("crop").checked ? "true" : "false");
    fd.append("enable_perspective", $("persp").checked ? "true" : "false");
    fd.append("response", "json");
    const resp = await fetch("/preprocess", { method: "POST", body: fd });
    if (!resp.ok) throw new Error("HTTP " + resp.status + ": " + (await resp.text()));
    const data = await resp.json();
    $("after").src = "data:image/jpeg;base64," + data.image_base64;
    renderMeta(data.metadata);
    $("status").textContent = "Listo";
  } catch (e) {
    $("err").textContent = "Error: " + e.message;
    $("status").textContent = "";
  } finally {
    $("go").disabled = false;
  }
});

function renderMeta(m) {
  const crop = m.crop_rect ? m.crop_rect.join(", ") : "sin recorte";
  const notes = (m.notes || []).map(n => "<li>" + n + "</li>").join("");
  $("meta").innerHTML =
    "<b>Corrección aplicada:</b> " + m.correction_angle_deg + "\\u00b0 &nbsp;·&nbsp; " +
    "<b>Recorte:</b> " + crop + " &nbsp;·&nbsp; " +
    "<b>Verticales:</b> " + m.num_vertical_lines + " &nbsp;·&nbsp; " +
    "<b>Lente:</b> " + (m.lens_corrected ? "corregida" : "sin corregir") +
    "<ul class='notes'>" + notes + "</ul>";
}
</script>
</body>
</html>
"""
