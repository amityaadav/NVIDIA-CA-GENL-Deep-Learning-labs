# Watch a Neural Network Think — project guide

This is the `webapp/` sub-project of the NVIDIA Deep Learning Labs repo: an
interactive visualizer of the Lab 1 MNIST feed-forward network. Draw a digit,
watch the forward pass replay as an animation.

The steering docs below are the source of truth for how this project is built.
Read them before making non-trivial changes, and update them when a decision
they record changes.

- @.claude/steering/product.md — what this is, who it's for, what's in/out of scope
- @.claude/steering/tech.md — stack, versions, conventions, deliberate decisions
- @.claude/steering/structure.md — architecture, layering, and the shared JSON contract

## The one rule that keeps this project working

Backend and frontend are coupled by a **shared contract**: layer names
(`input`, `hidden_1`, `hidden_2`, `output`) and the JSON trace shape returned by
`POST /inference`. Training, the inference hooks, and the animation all depend on
it. If you change the network shape, a layer name, or the response schema, update
*every* side in the same change:

- `backend/model.py` (`LAYER_NAMES`, `MNISTNet`)
- `backend/inference.py` (hooks, `predict()` payload)
- `frontend/src/components/NetworkView.jsx` (`COL` map, column layout)
- `README.md` API contract section

## Working here

- Backend: `cd backend && uvicorn app:app --reload --port 8000`
- Frontend: `cd frontend && npm run dev` (http://localhost:5173)
- Full stack: `cd webapp && docker compose up --build` (frontend :8080, backend :8000)
- Retrain weights: `cd backend && python train.py` → rewrites `mnist.pth`

## Tests

- Backend: `cd backend && .venv/bin/python -m pytest` (pytest; deps in
  `requirements-dev.txt`).
- Frontend: `cd frontend && npm test` (Vitest + jsdom).

The suite exists to guard the shared contract above — when you change the network
shape, a layer name, or the response schema, update the tests in the same change.
No linter/formatter/CI yet; if you add one, record it in `tech.md`.
