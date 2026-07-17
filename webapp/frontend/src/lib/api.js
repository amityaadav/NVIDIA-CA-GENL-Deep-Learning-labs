const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Send a 784-length pixel array (row-major 28x28, values 0..1) to the backend
 * and get back the full activation trace to animate.
 */
export async function runInference(pixels) {
  const res = await fetch(`${API_URL}/inference`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Inference failed (${res.status}). ${detail}`);
  }
  return res.json();
}

/**
 * Ask "why this digit?": returns the backward sub-network (selected nodes per
 * layer + connecting edges) that drove output `target` on this drawing.
 */
export async function runExplain(pixels, target) {
  const res = await fetch(`${API_URL}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels, target }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Explain failed (${res.status}). ${detail}`);
  }
  return res.json();
}

/**
 * Inspect one neuron: its weighted-sum breakdown (bias, z, a, top terms) and,
 * for hidden_1, its weights as a 28x28 image ("what is it looking for?").
 */
export async function runNeuron(pixels, layer, index) {
  const res = await fetch(`${API_URL}/neuron`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels, layer, index }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Neuron inspect failed (${res.status}). ${detail}`);
  }
  return res.json();
}
