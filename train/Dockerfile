# Training image: same CUDA/torch base as the API so train and serve match.
ARG BASE_IMAGE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    USE_TF=0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./

ENTRYPOINT ["python3"]
CMD ["train.py", "--help"]
