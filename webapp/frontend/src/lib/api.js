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
