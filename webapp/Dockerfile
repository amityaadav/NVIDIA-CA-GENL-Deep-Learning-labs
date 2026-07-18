# Single-container build for a public "Watch it think" deploy (e.g. Hugging Face
# Spaces): builds the frontend, then serves it AND the inference API from one
# FastAPI process on one origin (no CORS). Training is disabled here.

# --- Stage 1: build the React frontend ---
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty API URL -> the app calls /inference on the same origin.
RUN VITE_API_URL= npm run build

# --- Stage 2: backend + built frontend ---
# CPU-only PyTorch keeps the image small; mnist.pth is baked in.
FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/model.py backend/inference.py backend/training.py backend/app.py backend/mnist.pth ./
COPY --from=frontend /fe/dist ./static

# Public demo: inference only, and lock CORS down (same-origin anyway).
ENV ENABLE_TRAINING=0
ENV ALLOWED_ORIGINS=""

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
