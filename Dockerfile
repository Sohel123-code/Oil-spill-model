# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps (CPU-only torch to stay lean) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart Pillow pandas timm

# ── Copy app files ────────────────────────────────────────────────────────────
COPY main.py model.py geo_db.py ./
COPY static/ ./static/
COPY external/ ./external/
COPY cnn_swin_best.pth .

# ── HuggingFace Spaces runs as non-root user 1000 ─────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── Expose port 7860 (required by HuggingFace Spaces) ────────────────────────
EXPOSE 7860

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
