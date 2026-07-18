# Structure steering — Watch a Neural Network Think

## Directory map
```
webapp/
├── docker-compose.yml         full-stack local run (frontend :8080, backend :8000)
├── README.md                  public-facing docs + API contract
├── CLAUDE.md                  project guide (imports the steering trio)
├── .claude/steering/          product.md · tech.md · structure.md
├── backend/                   FastAPI + PyTorch inference + live-training service
│   ├── app.py                 HTTP: POST /inference · /explain · /neuron, GET /health · /train (SSE), CORS
│   ├── inference.py           Predictor: forward hooks, top paths, class-trace, neuron breakdown
│   ├── training.py            live training: fresh model on an MNIST subset, yields per-batch metrics
│   ├── model.py               MNISTNet (784→512→512→10) + LAYER_NAMES
│   ├── train.py               one-shot training → mnist.pth
│   ├── mnist.pth              committed trained weights (~97.9% val acc)
│   ├── requirements.txt       pinned deps (CPU torch)
│   ├── requirements-dev.txt   test deps (pytest, httpx)
│   ├── pytest.ini
│   ├── tests/                 test_model · test_inference · test_app · test_explain · test_neuron · test_training
│   └── Dockerfile
└── frontend/                  React + Vite + Canvas
    ├── index.html · vite.config.js · package.json · nginx.conf · Dockerfile
    └── src/
        ├── main.jsx           React entry
        ├── App.jsx            orchestrator: state machine, keyboard, wiring, focus
        ├── styles.css
        ├── test/setup.js      Vitest + jest-dom setup
        ├── components/
        │   ├── DrawCanvas.jsx        280×280 drawing surface (imperative ref API)
        │   ├── NetworkView.jsx       canvas renderer: forward pass, hover, focus, glyphs,
        │   │                         activation-fn info, preprocess morph, softmax moment
        │   ├── Controls.jsx          play/pause/step/speed (pure view of the anim hook)
        │   ├── ContributorPanel.jsx  plain-language "why this digit?" summary
        │   ├── NeuronInspector.jsx   weight image (hidden_1) + weighted-sum math for a neuron
        │   ├── LegendItem.jsx        legend key + clickable info popover
        │   └── TrainingPanel.jsx     "Watch it learn" tab: controls + live loss chart (SSE)
        ├── hooks/
        │   └── useAnimation.js  the Animation Controller (progress 0→4, + snapToEnd for Live)
        └── lib/
            ├── preprocess.js    normalize() → { input, geometry, inkCanvas }; canvasToInput wrapper
            ├── inspect.js       pure helpers: positions, hitTest, describeNeuron, LAYER_INFO, region/runner-up
            ├── model.js         in-browser inference (predict/explain/neuron) — port of inference.py
            └── api.js           runInference/runExplain/runNeuron → model.js; openTrainStream → backend
```

## Deployment: static, client-side inference
The public "Watch it think" app runs **entirely in the browser** — no backend.
`frontend/src/lib/model.js` is a faithful JS port of `inference.py` that loads the
weights (`backend/export_weights.py` → `frontend/public/model/weights.f32`, a flat
float32 blob) and produces the same trace shapes. `model.test.js` proves JS↔Python
parity against a backend-generated fixture (`parity.fixture.json`). This makes the
app a pure static site (Vite `base` from `VITE_BASE`), deployed to GitHub Pages by
`.github/workflows/deploy.yml`; `.github/workflows/ci.yml` runs both test suites.
The FastAPI backend is still used for the (hidden) training tab and local dev.
**If you change `inference.py` or the model, re-run `export_weights.py`, keep
`model.js` in sync, and regenerate the parity fixture.**

## The shared contract (the spine of the system)
Backend and frontend are decoupled in code but coupled by two things. Change
either and you must update both sides + the README in the same change.

**1. Layer names** — `input`, `hidden_1`, `hidden_2`, `output`.
- Defined in `model.py` (`LAYER_NAMES = ["hidden_1", "hidden_2", "output"]`).
- Consumed by hooks in `inference.py` and by the `COL` map in `NetworkView.jsx`.

**2. The `POST /inference` JSON trace** (authoritative example in `README.md`):
```jsonc
{
  "prediction": 7,
  "probs": [ ...10 softmax floats... ],
  "layers": [
    { "name": "input",    "size": 784, "shape": [28,28], "activations": [...], "peak": ... },
    { "name": "hidden_1", "size": 512, "activations": [...], "peak": ... },
    { "name": "hidden_2", "size": 512, "activations": [...], "peak": ... },
    { "name": "output",   "size": 10,  "activations": [...], "peak": ... }
  ],
  "transitions": [
    { "from": "input", "to": "hidden_1",
      "links": [ { "src": 296, "dst": 6, "strength": 1.0, "sign": -1 }, ... ] },
    ...
  ]
}
```
- `activations` are normalized-for-brightness raw values (post-ReLU for hidden,
  softmax probs for output). `strength` is 0–1 (edge opacity/width); `sign` is
  +1 excitatory / −1 inhibitory (edge color).
- Hidden layers also carry `preacts` (raw z before ReLU) and the output carries
  `logits` (before softmax). These power the activation-function view: the ReLU
  curve + hovered neuron's z→a point, and the "dead neuron" (z ≤ 0) markers.
  Invariant: `relu(preacts) == activations`; `softmax(logits) == probs`.

**3. `POST /explain` ("why this digit?", additive)** — `{ pixels, target }` →
the backward sub-network that drove output `target`:
```jsonc
{
  "target": 7, "prediction": 7, "targetProb": 0.98,
  "nodes": { "hidden_2": [...], "hidden_1": [...], "input": [...] },  // selected indices
  "edges": [ { "from": "hidden_2", "to": "output", "src": 41, "dst": 7,
              "strength": 1.0, "sign": 1 }, ... ]                     // same edge shape as above
}
```
- Additive: it does NOT change `/inference`. Reuses the layer names + edge
  shape, so it's bound by the same contract. Beam widths (`TRACE_*` in
  `inference.py`) are tuned for legibility, not completeness.
- Frontend: `App` stores the last `pixels`; clicking an output digit calls
  `runExplain` → `focus`, which `NetworkView` renders (dim all but the
  sub-network) and `ContributorPanel` summarizes.

**4. `POST /neuron` (single-neuron inspector, additive)** — `{ pixels, layer, index }`
→ `{ bias, z, a, activation, sourceLayer, topTerms[], weightImage? }`. `z` is the
pre-activation (logit for output), `a` post (softmax prob for output); `topTerms`
are the strongest `weight × value` inputs; `weightImage` (784) is present only for
`hidden_1`, whose inputs are pixels. Frontend: clicking a hidden neuron →
`runNeuron` → `NeuronInspector` (weight template + the arithmetic).

## Interaction & animation flows (frontend)
- **Live mode** (`App.live`): each `DrawCanvas` stroke end re-runs inference and
  `useAnimation.snapToEnd()` — updates the view with no sweep. Off by default.
- **Preprocess morph** (`App.prep`): on Run, `normalize()` yields geometry + an
  ink canvas; a short rAF drives `prep.t` 0→1 and `NetworkView.drawPreprocess`
  animates crop→scale→center in the input area, then the forward sweep plays.
- **Softmax moment**: within the output phase, `drawOutput` grows bars to raw
  `logits`, then morphs them into probabilities — using the `logits` already in
  the `/inference` trace.
- **Neuron inspector vs. class trace** are mutually exclusive panels (clicking one
  clears the other).

## Training mode ("Watch it learn" tab)
- `App.mode` toggles between the inference view and `TrainingPanel`.
- **`GET /train` (Server-Sent Events)** — query params `lr`, `batch_size`, `epochs`.
  Streams one `data:` event per batch: `{ step, epoch, totalSteps, loss, trainAcc,
  validAcc|null }`, then a `{done}` or `{diverged}` event. `training.py` trains a
  **fresh** MNISTNet with plain SGD on a small MNIST subset (`TRAIN_SUBSET`/
  `VALID_SUBSET`) — the committed `mnist.pth` is never touched.
- `training.iter_metrics(train_set, valid_set, ...)` takes datasets as args so it's
  testable with a synthetic dataset (no download); `training.run()` wires in the
  cached real subset. Non-finite loss → a `diverged` event (the high-LR lesson).
- At snapshot steps the stream also carries: `templates` (12 hidden_1 weight
  images), `sampleTrace` (a fixed digit's forward pass through the CURRENT model —
  same shape as `/inference`, via `inference.forward_trace`), and `learning`
  (per-neuron gradient norms + top weight-update edges).
- Frontend: `openTrainStream()` opens the `EventSource`; `TrainingPanel` draws the
  live loss curve, the weight-template grid, and `TrainNetworkView` — a live
  network with a toggle: **Activations** (the sample forward pass, teal) and
  **Learning** (gradient magnitudes, amber). `TrainNetworkView` reuses
  `computePositions` but is its own focused renderer (no inference interactivity).
- Still planned: the single-step backprop teardown (forward → loss → reverse
  gradient sweep → update).

## Backend layering (strict, keep it this way)
```
app.py  (HTTP: validation, CORS, error mapping — thin, no ML logic)
   └── inference.py  (Predictor: owns the model instance, hooks, trace assembly)
          └── model.py  (MNISTNet + shape constants — no I/O, no serialization)
```
- Add ML/trace logic to `inference.py`, not `app.py`.
- `model.py` stays pure structure: no file I/O, no HTTP, no numpy/json shaping.
- New endpoints go in `app.py` and delegate to a method on `Predictor`.

## Frontend data flow
```
DrawCanvas (ref) ──canvas──▶ App.handleRun
                                 │ canvasToInput()  [lib/preprocess.js]
                                 │ runInference()   [lib/api.js]  → trace
                                 ▼
                    App state: trace + status ("idle|loading|ready|error")
                                 │
                    useAnimation(PHASES=4) drives `progress` 0→4
                                 ▼
                    NetworkView(trace, progress)   Controls(anim)
```
- `App.jsx` is the only stateful orchestrator: owns `trace`/`status`/`error`,
  the keyboard shortcuts (Enter=run, Esc=clear), and wires the pieces.
- `useAnimation` owns *all* timing: `progress` runs 0→`phases` (one phase per
  layer). Layer L is fully lit when `progress ≥ L`; the transition into L animates
  while `progress ∈ [L-1, L]`. Nothing else should own a timer.
- `NetworkView` is a pure function of `(trace, progress)` → canvas pixels. Neuron
  screen positions are precomputed once (`computePositions`, memoized). If you add
  a layer or change counts, update `computePositions` and the `COL`/column headers.
- `Controls` is a dumb view over the anim hook — no logic of its own.
- Components communicate with the canvas via imperative refs (`DrawCanvas`
  exposes `canvas()` and `clear()` through `useImperativeHandle`).

## Where things go (extension guide)
- **New visualization** of the same forward pass → `NetworkView.jsx` (rendering)
  and/or extra fields in the trace (both `inference.py` and the README).
- **New playback behavior** → `useAnimation.js`, surfaced through `Controls.jsx`.
- **Change how drawings are normalized** → `lib/preprocess.js` *and* `train.py`
  must agree (see the no-normalization decision in tech.md).
- **New network / different weights** → `model.py` (+ retrain via `train.py`);
  then reconcile layer names, counts, and the trace on the frontend.
- **New API call** → add to `lib/api.js`; keep fetch/error handling there, not in
  components.
