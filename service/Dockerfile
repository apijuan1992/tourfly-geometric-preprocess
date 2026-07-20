# Imagen del micro-servicio de preproceso. Contenedor porque OpenCV no corre
# en serverless de Node (Vercel); se despliega en Cloud Run / Fly / Railway / VM.
FROM python:3.11-slim

# Librerías del sistema que necesita OpenCV (headless).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps primero (mejor cache).
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# Código.
COPY realestate_preprocess ./realestate_preprocess
COPY service ./service

EXPOSE 8000

# $PORT lo inyectan Cloud Run / Railway; por defecto 8000.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT}"]
