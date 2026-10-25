# FastAPI backend (api/) as a Hugging Face Docker Space. Spaces send traffic to port 7860.
# Local: docker build -t clinical-api . && docker run -p 7860:7860 -e SUPABASE_URL=... -e SUPABASE_KEY=... clinical-api
# Report worker (queue mode, needs REDIS_URL): same image, command python -m worker.report_worker; see docker-compose.yml
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860

# OpenCV (headless) needs glib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch first, so the unpinned torch in requirements.txt doesn't pull the CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY --chown=user . .
# traces/ gets one JSONL file per day (lost on restart; Supabase keeps the real copy)
RUN mkdir -p /app/traces && chown user /app /app/traces
USER user

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn api.main:create_app --factory --host 0.0.0.0 --port ${PORT}"]
