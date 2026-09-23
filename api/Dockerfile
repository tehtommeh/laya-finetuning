# The base image carries CUDA + a matching torch build. Overriding it via
# API_BASE_IMAGE lets the same Dockerfile target older drivers or ROCm hosts.
ARG BASE_IMAGE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Newer pytorch images install torch into a PEP 668 system Python; a venv
    # would hide it, so allow pip to add packages alongside (older pips ignore this).
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Dependencies first so edits to app.py reuse the cached layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

EXPOSE 8000
# One worker only: each worker would load its own full copy of the weights.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
