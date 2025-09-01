# Dockerfile (repo root)
FROM python:3.11.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app


# Copy and install deps (force wheels if available)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary --only-binary=:all: -r /tmp/requirements.txt

# Copy app 
COPY . /app

# Defaults (can be overridden by compose)
ENV API_URL=http://localhost:8000
ENV CLEAN_CLABSI_CSV=/app/demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023_clean.csv
ENV DOCS_DIR=/app/docs

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
