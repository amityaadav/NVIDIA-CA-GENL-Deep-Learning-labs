"""FastAPI service exposing POST /inference.

Loads the trained MNIST net once at startup and returns a full activation
trace per drawing. Inference is sub-millisecond; the frontend paces it into
the animation, so no streaming is needed.
"""
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import Predictor, LAYER_SIZE

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


class ExplainRequest(BaseModel):
    pixels: list[float] = Field(..., description="784 grayscale values in [0, 1], row-major 28x28")
    target: int = Field(..., ge=0, le=9, description="Which output digit to explain (0-9)")


class NeuronRequest(BaseModel):
    pixels: list[float] = Field(..., description="784 grayscale values in [0, 1], row-major 28x28")
    layer: Literal["hidden_1", "hidden_2", "output"]
    index: int = Field(..., ge=0, description="Neuron index within the layer")


def _require_784(pixels: list[float]) -> None:
    if len(pixels) != EXPECTED_PIXELS:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {EXPECTED_PIXELS} pixels, got {len(pixels)}",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inference")
def inference(req: InferenceRequest):
    _require_784(req.pixels)
    return predictor.predict(req.pixels)


@app.post("/explain")
def explain(req: ExplainRequest):
    """"Why this digit?" -- the sub-network that drove output `target`."""
    _require_784(req.pixels)
    return predictor.explain(req.pixels, req.target)


@app.post("/neuron")
def neuron(req: NeuronRequest):
    """One neuron's weighted-sum breakdown (+ weight image for hidden_1)."""
    _require_784(req.pixels)
    if req.index >= LAYER_SIZE[req.layer]:
        raise HTTPException(
            status_code=422,
            detail=f"index {req.index} out of range for {req.layer} (size {LAYER_SIZE[req.layer]})",
        )
    return predictor.neuron(req.pixels, req.layer, req.index)
