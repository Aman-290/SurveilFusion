FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[vision,agents,integrations]"

COPY surveilfusion ./surveilfusion
COPY web ./web
COPY config ./config

EXPOSE 8080
CMD ["python", "-m", "surveilfusion"]
