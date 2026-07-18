"""FastAPI service exposing POST /inference.

Loads the trained MNIST net once at startup and returns a full activation
trace per drawing. Inference is sub-millisecond; the frontend paces it into
the animation, so no streaming is needed.
"""
import asyncio
import json
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import training
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


BATCH_SIZES = (8, 16, 32, 64, 128)

# Training is CPU-heavy and can be abused on a public host, so it's opt-in.
# The public "Watch it think" deploy runs with ENABLE_TRAINING=0.
ENABLE_TRAINING = os.environ.get("ENABLE_TRAINING", "1") == "1"


@app.get("/train")
async def train(request: Request, lr: float = 0.1, batch_size: int = 32, epochs: int = 3):
    """Stream live training metrics (Server-Sent Events) for the Train tab.

    Trains a fresh model on a small MNIST subset; the canonical mnist.pth is
    untouched. Emits one `data:` event per batch, then a done/diverged event.
    """
    if not ENABLE_TRAINING:
        raise HTTPException(404, "training is disabled on this server")
    if not (1e-4 <= lr <= 10):
        raise HTTPException(422, "lr must be in [1e-4, 10]")
    if batch_size not in BATCH_SIZES:
        raise HTTPException(422, f"batch_size must be one of {BATCH_SIZES}")
    if not (1 <= epochs <= 10):
        raise HTTPException(422, "epochs must be in [1, 10]")

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def stream():
        yield sse({"status": "starting", "trainImages": training.TRAIN_SUBSET,
                   "validImages": training.VALID_SUBSET})
        # Each next() runs one batch off the event loop so it stays responsive
        # and we can notice the client disconnecting (Stop).
        it = training.run(lr, batch_size, epochs)
        while True:
            if await request.is_disconnected():
                break
            metric = await run_in_threadpool(next, it, None)
            if metric is None:
                break
            yield sse(metric)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Single-container deploy: if a built frontend is present, serve it at "/" (must
# come AFTER the API routes so they take precedence). Absent in local dev.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="frontend")
