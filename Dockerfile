# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS dependencies

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home rag

COPY requirements-linux-${TARGETARCH}.lock ./requirements-linux.lock
RUN python -m pip install --no-cache-dir uv==0.12.1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --torch-backend cpu -r requirements-linux.lock


FROM dependencies AS test

COPY Dockerfile .dockerignore compose.yaml requirements.in requirements-linux-*.lock ./
COPY .github ./.github
COPY src ./src
COPY scripts ./scripts
COPY eval ./eval
COPY tests ./tests
COPY data ./data

RUN python -m unittest discover -s tests -v
RUN python -m pip check


FROM dependencies AS runtime

COPY --chown=rag:rag src ./src
COPY --chown=rag:rag scripts ./scripts
COPY --chown=rag:rag eval ./eval
COPY --chown=rag:rag data ./data
COPY --chown=rag:rag .env.example ./.env.example

RUN mkdir -p /app/data/chroma /app/data/processed /app/artifacts /home/rag/.cache \
    && chown -R rag:rag /app/data /app/artifacts /home/rag

USER rag

ENTRYPOINT ["python", "scripts/container.py"]
CMD ["--help"]
