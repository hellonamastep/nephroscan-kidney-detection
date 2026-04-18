# ──────────────────────────────────────────────
# NephroScan — Dockerfile
# Hugging Face Spaces · Docker SDK
# Port: 7860 (HF default)
# ──────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="Atharva Barde"
LABEL description="NephroScan — Kidney Stone Detection using MobileNetV2"

# System deps for TensorFlow + Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/

# Create necessary directories
RUN mkdir -p model uploads

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose HF Spaces default port
EXPOSE 7860

# Environment
ENV FLASK_ENV=production
ENV PORT=7860

# Start with gunicorn (production WSGI server)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "120", \
     "--workers", "1", \
     "--threads", "2", \
     "--log-level", "info", \
     "app:app"]
