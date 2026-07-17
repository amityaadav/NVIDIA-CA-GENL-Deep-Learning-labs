# Tech steering — Watch a Neural Network Think

## Stack
| Area | Choice | Version (pinned) |
|------|--------|------------------|
| Inference API | FastAPI + Uvicorn | fastapi 0.115.6, uvicorn[standard] 0.34.0 |
| Validation | Pydantic v2 | 2.10.4 |
| ML | PyTorch + torchvision (CPU wheels) | torch 2.5.1, torchvision 0.20.1 |
| Frontend | React 18 + Vite 6 | react 18.3.1, vite 6.0.7, @vitejs/plugin-react 4.3.4 |
| Rendering | HTML5 Canvas 2D (no viz library) | — |
| Packaging | Docker + docker-compose (nginx serves the built frontend) | — |

Dependencies are **pinned**. Keep them pinned; bump deliberately, not incidentally.

## Runtime shape
- Backend is **CPU-only by design** (`torch.device("cpu")` in `inference.py`).
  No GPU assumptions; the model is tiny and inference is sub-millisecond.
- The model loads **once at startup** (`predictor = Predictor()` in `app.py`) and
  is reused for every request. Do not load weights per-request.
- Inference returns the **entire trace in one response** — no streaming, no
  websockets. The frontend owns pacing/animation.

## Deliberate decisions (don't "fix" these without cause)
- **state_dict, not the whole model.** `train.py` saves weights only; `model.py`
  owns the architecture. Keeps `mnist.pth` portable across PyTorch versions.
- **No mean/std normalization in training.** `train.py` uses only
  `ToImage` + `ToDtype(float32, scale=True)` → pixels in `[0,1]`. This is so the
  browser can send plain `[0,1]` pixels and match. If you add normalization to
  one side, you must add it to both (`train.py` and `frontend/lib/preprocess.js`).
- **Post-ReLU activations are what we visualize** (`inference.py`), because that's
  what neurons actually fire — not raw pre-activations.
- **Top-40 edges per transition** (`TOP_EDGES_PER_TRANSITION`) ranked by
  `|weight × activation|`. Enough to read the signal path without a hairball.
- **Named Linear layers, not nn.Sequential** (`model.py`), so inference can hook
  each layer and read its weight matrix by name.
- **Preprocessing lives in the browser** (`lib/preprocess.js`), not the backend,
  so the backend contract is a clean 784-float array and the normalization is
  visible/tweakable next to the canvas.

## Conventions
### Python (backend)
- Module docstrings explain *why* the module exists; comments explain non-obvious
  decisions, not mechanics. Match this density — it's high and intentional.
- Type hints on public functions/attrs; Pydantic models for request bodies.
- Constants in UPPER_SNAKE at module top (`EXPECTED_PIXELS`, `WEIGHTS_PATH`,
  `TOP_EDGES_PER_TRANSITION`, `INPUT_SIZE`…). Reuse them; don't inline magic numbers.
- Paths resolved relative to the file (`HERE = os.path.dirname(...)`), never CWD.
- Validate inputs at the API boundary and raise `HTTPException` with a clear detail
  (see the pixel-count check in `app.py`).

### JavaScript / React (frontend)
- Functional components + hooks only. ES modules (`"type": "module"`).
- Animation state lives in the `useAnimation` hook (the "Animation Controller");
  components read from it and render — they don't own timing logic.
- Canvas drawing is imperative inside `useEffect`/helper functions; keep the pure
  layout math (e.g. `computePositions`) separate and memoized.
- Config via Vite env: `VITE_API_URL` (baked at build time). Default to
  `http://localhost:8000` for local dev.
- JSDoc blocks on non-trivial functions, matching the existing style.
- No CSS framework; plain CSS in `src/styles.css`. Colors/spacing are hand-tuned.

## Security / deploy config
- **CORS**: `ALLOWED_ORIGINS` env var (comma-separated) on the backend. Defaults
  to `*` for local dev only — set it to the real frontend origin in production.
- **API URL**: build the frontend with `--build-arg VITE_API_URL=...` for deploys.
- Cold starts: PyTorch import takes a few seconds; keep one instance warm for demos.

## Testing
- **Backend: pytest** (`backend/tests/`, config in `backend/pytest.ini`). Run:
  `cd backend && .venv/bin/python -m pytest`. Test deps in `requirements-dev.txt`
  (pytest, httpx). Coverage centers on the shared contract: network shape/layer
  names (`test_model.py`), the trace payload invariants (`test_inference.py`), and
  the HTTP boundary/validation (`test_app.py`). Uses FastAPI `TestClient`; no
  network (the committed `mnist.pth` loads locally).
- **Frontend: Vitest + jsdom** (`*.test.js` beside sources). Run:
  `cd frontend && npm test` (watch: `npm run test:watch`). Config lives in
  `vite.config.js` (`test` block) + `src/test/setup.js`. The `canvas` package gives
  jsdom a real 2D context so `preprocess.test.js` exercises the actual MNIST
  normalization; `useAnimation.test.js` pins the animation controller's discrete
  logic; `api.test.js` pins the fetch contract.
- **When you change the shared contract, update the tests in the same change** —
  they exist to make the "keep both sides in sync" rule enforceable.

## Not yet present (add here if you introduce them)
- No linter/formatter, no type-checker, no CI. If you add any, document the
  command and wire it into the run instructions in `../CLAUDE.md`.
- The frontend animation's continuous rAF loop is not covered (only the discrete
  step/play/pause logic); NetworkView canvas rendering is not asserted pixel-wise.
