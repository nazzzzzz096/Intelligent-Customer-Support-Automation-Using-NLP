# ============================================================
# STAGE 1 — Build Dependencies
# ============================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# System dependencies for building some wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into /wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels


# ============================================================
# STAGE 2 — Final Runtime Image
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# Copy built wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy only the minimal FastAPI app
COPY app ./app

# Create models directory (empty → downloaded at runtime)
RUN mkdir -p /app/models

# Environment fixes
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


