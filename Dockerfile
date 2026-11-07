# syntax=docker/dockerfile:1.7
# K-12.07: replication container. Reproduces the validation report end to end.
#
#   docker build -t kairos-replicate .
#   docker run --rm --gpus all -v "$PWD/out:/out" kairos-replicate
#
# Without a GPU, add `-e KAIROS_CPU_ONLY=1` (much slower: LPPLS grid search
# falls back to the float64 CPU reference).

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG UV_VERSION=0.8

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    KAIROS_CPU_ONLY=0

# CmdStan (used by prophet) needs a C++ toolchain at build time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN useradd --create-home --uid 1000 kairos \
 && mkdir -p /opt/venv /opt/python /out /work \
 && chown -R kairos:kairos /opt/venv /opt/python /out /work
USER kairos
WORKDIR /work

RUN uv python install 3.12

# Dependencies first, from the lock file only, so this layer caches.
COPY --chown=kairos:kairos pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra gpu --extra dev

COPY --chown=kairos:kairos . .
RUN uv sync --frozen --extra gpu --extra dev

COPY --chown=kairos:kairos docker/reproduce.sh /usr/local/bin/kairos-reproduce

VOLUME ["/out"]
ENTRYPOINT ["bash", "/usr/local/bin/kairos-reproduce"]
