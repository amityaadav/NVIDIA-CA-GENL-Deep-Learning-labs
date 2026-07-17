"""FastAPI service exposing POST /inference.

Loads the trained MNIST net once at startup and returns a full activation
trace per drawing. Inference is sub-millisecond; the frontend paces it into
the animation, so no streaming is needed.
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import Predictor

EXPECTED_PIXELS = 28 * 28

app = FastAPI(title="MNIST Forward-Pass Visualizer", version="1.0.0")

# Public demo: lock this to your deployed frontend origin(s) via the
# ALLOWED_ORIGINS env var (comma-separated). Falls back to permissive for
# local dev only.
_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Loaded once at import/startup and reused for every request.
predictor = Predictor()


class InferenceRequest(BaseModel):
    pixels: list[float] = Field(..., description="784 grayscale values in [0, 1], row-major 28x28")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inference")
def inference(req: InferenceRequest):
    if len(req.pixels) != EXPECTED_PIXELS:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {EXPECTED_PIXELS} pixels, got {len(req.pixels)}",
        )
    return predictor.predict(req.pixels)
