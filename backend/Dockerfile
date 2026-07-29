FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    EMBEDDING_BACKEND=fastembed \
    FASTEMBED_MODEL=BAAI/bge-small-en-v1.5 \
    FASTEMBED_CACHE_DIR=/app/.cache/fastembed

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and validate the hosted embedding model while building the image.
# Render can then start from the image cache instead of downloading it during
# the first document upload.
RUN python -c "from fastembed import TextEmbedding; model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed'); next(model.embed(['warmup']))"

COPY app ./app

# Hosts (Render, HF Spaces, Fly) inject $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
