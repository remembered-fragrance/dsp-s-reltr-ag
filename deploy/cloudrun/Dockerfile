# Image cho Google Cloud Run (CPU). Build context = thư mục do CI dựng:
#   space/build_space.py --out build/web  +  web.py  +  weights/epoch4_weights.pth (từ Release weights-a1)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt fastapi "uvicorn[standard]" python-multipart

COPY . .
RUN useradd --create-home app && chown -R app /app
USER app

ENV DSP_CKPT=/app/weights/epoch4_weights.pth PORT=8080
CMD ["sh", "-c", "exec uvicorn web:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 5"]
