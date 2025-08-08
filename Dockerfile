FROM python:3.10.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OUTPUT_DIR=/data \
    MODEL_ID=SimianLuo/LCM_Dreamshaper_v7 \
    LCM_REVISION=fb9c5d \
    LCM_CUSTOM_PIPELINE=latent_consistency_txt2img \
    LCM_CUSTOM_REVISION=main \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers \
    DIFFUSERS_CACHE=/home/appuser/.cache/huggingface/diffusers

WORKDIR /app

# Install minimal OS deps (for healthcheck) and create non-root user
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 appuser \
    && useradd -m -u 1000 -g 1000 appuser

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt ./

# Allows CPU/GPU specific Torch wheels at build time
ARG TORCH_VERSION=2.3.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output and cache dirs with proper permissions
RUN mkdir -p ${OUTPUT_DIR} ${HF_HOME} ${TRANSFORMERS_CACHE} ${DIFFUSERS_CACHE} \
    && chown -R appuser:appuser ${OUTPUT_DIR} /app ${HF_HOME}

USER appuser

# Optional: pre-download model weights to warm cache (increases image size)
ARG PRELOAD_MODEL=0
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
    python - <<'PY' \
import os
import torch
from diffusers import DiffusionPipeline

model_id = os.getenv('MODEL_ID')
revision = os.getenv('LCM_REVISION')
custom_pipeline = os.getenv('LCM_CUSTOM_PIPELINE')
custom_revision = os.getenv('LCM_CUSTOM_REVISION')

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
_ = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=custom_pipeline, custom_revision=custom_revision, revision=revision, torch_dtype=dtype, safety_checker=None)
print('Model cached')
PY
    ; fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD wget -qO- http://127.0.0.1:8000/healthz || exit 1

VOLUME ["/data"]

ENV LOG_LEVEL=INFO
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
