"""Prueba del endpoint /preprocess con FastAPI TestClient.

Sube una imagen sintética con verticales convergentes y verifica que el
servicio responde una imagen JPEG con los headers de metadatos.

Requiere: pip install -r service/requirements.txt httpx
Ejecutar:  python tests/test_service.py   (o)  python -m pytest tests/
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from service.app import app
from tests.test_synthetic import _make_keystoned_image


def test_preprocess_endpoint_returns_image():
    client = TestClient(app)

    image = _make_keystoned_image()
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    files = {"file": ("foto.jpg", encoded.tobytes(), "image/jpeg")}
    resp = client.post("/preprocess", files=files)

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert "X-Correction-Angle" in resp.headers

    # La respuesta debe ser una imagen válida y decodificable.
    out = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    assert out is not None and out.size > 0
    print("OK  /preprocess ->", len(resp.content), "bytes,",
          "ángulo", resp.headers.get("X-Correction-Angle"))


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.json()["status"] == "ok"


if __name__ == "__main__":
    test_health()
    test_preprocess_endpoint_returns_image()
    print("Pruebas del servicio superadas.")
