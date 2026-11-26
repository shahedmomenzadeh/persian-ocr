FROM python:3.11-slim-bookworm

ARG PADDLE_FLAVOR=gpu
ARG INSTALL_EXTRAS=nemotron,kenlm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/app/models/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/models/cache/huggingface/hub \
    PADDLE_PDX_CACHE_HOME=/app/models/cache/paddlex \
    PADDLEX_HOME=/app/models/cache/paddlex \
    PADDLE_PDX_MODEL_SOURCE=huggingface \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    libbz2-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    liblzma-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN set -eux; \
    if [ "${PADDLE_FLAVOR}" = "gpu" ]; then \
        uv pip install --system --no-cache "paddlepaddle-gpu>=3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/; \
    else \
        uv pip install --system --no-cache "paddlepaddle>=3.3.0"; \
    fi; \
    EXTRA_FLAGS=""; \
    CPU_TORCH=0; \
    for extra in $(printf '%s' "${INSTALL_EXTRAS}" | tr ',' ' '); do \
        if [ -n "$extra" ]; then \
            EXTRA_FLAGS="$EXTRA_FLAGS --extra $extra"; \
            if [ "${PADDLE_FLAVOR}" != "gpu" ] && [ "$extra" = "nemotron" ]; then \
                CPU_TORCH=1; \
            fi; \
        fi; \
    done; \
    if [ "$CPU_TORCH" = "1" ]; then \
        uv pip install --system --no-cache "torch>=2.7.0,<2.8" "torchvision>=0.22.0,<0.23" --index-url https://download.pytorch.org/whl/cpu; \
    fi; \
    uv pip install --system --no-cache -r pyproject.toml $EXTRA_FLAGS "numpy>=1.26.0,<2.3"

COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && uv pip install --system --no-cache --no-deps -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
