# CPU-only PyTorch keeps the image ~1GB instead of ~5GB. The weights file
# (mnist.pth) is baked in, so the container is self-contained at runtime.
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching. requirements.txt pins the CPU
# torch wheels via the PyTorch index.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py inference.py app.py mnist.pth ./

EXPOSE 8000

# Single worker is plenty: inference is sub-millisecond and the model is held
# in memory per process. Scale with more workers only if you need throughput.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
