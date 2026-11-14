# The whole app in one container: the 3D viewer (static files) + the upload service (FastAPI, ONNX on CPU).
#   docker build -t brain-seg .
#   docker run -p 7860:7860 --memory 16g --cpus 2 brain-seg      -> http://localhost:7860
# Port 7860 and the 2-core / 16 GB limits match a free Hugging Face Space.

# ---- stage 1: build the viewer (TypeScript + three.js -> static files)
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- stage 2: the service
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/pipeline \
    BRAIN_THREADS=2 \
    BRAIN_WEB_DIR=/app/web/dist \
    BRAIN_JOBS_DIR=/tmp/jobs \
    HF_HOME=/tmp/hf
WORKDIR /app

# libraries first (this layer is reused as long as the requirements don't change)
COPY requirements/api.txt requirements/api.txt
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements/api.txt

# code + small data that live in git
COPY api/ api/
COPY ml/__init__.py ml/inference.py ml/fingerprint.py ml/metrics.py ml/
COPY pipeline/ pipeline/
COPY splits/ splits/
COPY models/*.json models/
COPY scripts/hub.py scripts/hub.py
COPY --from=web /web/dist web/dist

# large files from the Hugging Face Hub: models, sample brains (downloads) and gallery cases (viewer)
ARG HF_REPO=MohammadSakibHossain/brain-tumor-segmentation-mlops
RUN test -n "$HF_REPO" || (echo "set --build-arg HF_REPO=USER/brain-tumor-segmentation-mlops" && exit 1) \
 && python scripts/hub.py download --repo "$HF_REPO" --to /app \
        --samples-dir /app/web/dist/samples --cases-dir /app/web/dist/cases \
 && rm -rf /tmp/hf

# run as a normal user (Hugging Face Spaces use uid 1000)
RUN useradd -m -u 1000 app && mkdir -p /tmp/jobs && chown app /tmp/jobs
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
