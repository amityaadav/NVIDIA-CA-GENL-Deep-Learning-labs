/**
 * App API. Inference runs entirely in the browser (see model.js) so the app is
 * fully static — no backend needed. Training (the hidden "Watch it learn" tab)
 * still streams from the backend when enabled.
 */
import { getModel } from "./model.js";

export async function runInference(pixels) {
  const model = await getModel();
  return model.predict(pixels);
}

/** "Why this digit?" — the backward sub-network that drove output `target`. */
export async function runExplain(pixels, target) {
  const model = await getModel();
  return model.explain(pixels, target);
}

/** One neuron's weighted-sum breakdown (+ weight image for hidden_1). */
export async function runNeuron(pixels, layer, index) {
  const model = await getModel();
  return model.neuron(pixels, layer, index);
}

// --- training (backend, only used by the hidden Train tab) ---
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export function openTrainStream({ lr, batchSize, epochs }) {
  const url = new URL(`${API_URL}/train`);
  url.searchParams.set("lr", lr);
  url.searchParams.set("batch_size", batchSize);
  url.searchParams.set("epochs", epochs);
  return new EventSource(url);
}

// Start fetching the weights as soon as the app loads, so the first prediction
// is instant.
getModel().catch(() => {});
