# ------------------------------
# Stage 1 — Build dependencies
# ------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Install system deps
RUN apt-get update && apt-get install -y \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------
# Stage 2 — Final runtime image
# ------------------------------
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app ./app
COPY ui ./ui
COPY supervisord.conf .

# Create empty models folder
RUN mkdir -p /app/models

# Expose backend + frontend ports
EXPOSE 8000
EXPOSE 8501

# Install supervisor
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Start both apps
CMD ["supervisord", "-c", "supervisord.conf"]

