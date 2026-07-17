# Watch a Neural Network Think

An interactive visualizer for the Lab 1 MNIST network. Draw a digit; it's
normalized the way MNIST expects and pushed through a `784 → 512 → 512 → 10`
feed-forward net. The forward pass is then replayed as an animation: neurons
light up by activation, layer by layer, and the strongest **weighted paths**
(the actual `weight × activation` contributions on your drawing) trace how the
answer forms.

```
webapp/
├── backend/          FastAPI + PyTorch inference service
│   ├── model.py      the 784→512→512→10 network (shared with training)
│   ├── train.py      train once → mnist.pth
│   ├── inference.py  forward hooks + top-weighted-path extraction
│   ├── app.py        POST /inference
│   └── mnist.pth     trained weights (committed; 97.9% val accuracy)
└── frontend/         React + Vite + Canvas
    └── src/
        ├── components/   DrawCanvas · NetworkView · Controls
        ├── hooks/        useAnimation (the Animation Controller)
        └── lib/          preprocess (MNIST normalization) · api
```

## Run it locally

**Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The committed `mnist.pth` is loaded at startup, so the API works immediately.
To retrain from scratch (a couple of minutes on CPU): `python train.py`.

**Frontend**

```bash
cd frontend
npm install
cp .env.example .env      # points at http://localhost:8000
npm run dev               # http://localhost:5173
```

## Run the whole stack with Docker

```bash
cd webapp
docker compose up --build
```

Frontend on `http://localhost:8080`, backend on `http://localhost:8000`.

## How it works

1. **Draw** — you sketch a digit on a 280×280 canvas (`DrawCanvas`).
2. **Preprocess** — in the browser (`lib/preprocess.js`): invert to white-on-black,
   crop to the ink, scale the longest side to 20px, and center by center-of-mass
   inside a 28×28 field — the same normalization MNIST digits went through. This
   step is what makes predictions accurate; skip it and the model guesses garbage.
3. **Infer** — `POST /inference` with 784 floats. The backend runs one forward
   pass with hooks on each layer, softmaxes the logits, and computes the top
   `weight × activation` edges per layer transition.
4. **Animate** — the response is the full activation trace. `useAnimation` paces
   it into a multi-second sweep; `NetworkView` renders every neuron plus the
   travelling signal along the strongest paths.

Inference is sub-millisecond, so the entire trace comes back in one response —
no streaming needed. The animation, not the compute, is what takes time.

## API contract

`POST /inference`

```jsonc
// request
{ "pixels": [0.0, 0.0, 0.98, ...] }   // 784 floats in [0,1], row-major 28×28

// response
{
  "prediction": 7,
  "probs": [0.001, ..., 0.98, ...],    // 10 softmax probabilities
  "layers": [
    { "name": "input",    "size": 784, "shape": [28,28], "activations": [...] },
    { "name": "hidden_1", "size": 512, "activations": [...], "peak": 8.3 },
    { "name": "hidden_2", "size": 512, "activations": [...], "peak": 6.1 },
    { "name": "output",   "size": 10,  "activations": [...] }
  ],
  "transitions": [
    { "from": "input", "to": "hidden_1",
      "links": [ { "src": 296, "dst": 6, "strength": 1.0, "sign": -1 }, ... ] },
    ...
  ]
}
```

`strength` is 0–1 (drives edge opacity/width); `sign` is +1 excitatory / −1
inhibitory (drives edge color).

## Deploying the public demo

- **CORS** — set `ALLOWED_ORIGINS` on the backend to your frontend's real origin
  (comma-separated for several). It defaults to permissive for local dev only.
- **API URL** — build the frontend with `VITE_API_URL` pointing at your public
  backend domain (`docker build --build-arg VITE_API_URL=...`).
- **Cold starts** — PyTorch takes a few seconds to import. On hosts that sleep
  idle containers (free tiers), the first request after idle will lag. Keep one
  instance warm (min-instances ≥ 1, or a periodic `/health` ping) so the demo
  feels instant. Steady-state inference is milliseconds.
