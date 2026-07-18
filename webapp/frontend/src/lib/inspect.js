/**
 * Pure geometry + inspection helpers for the network view. Kept free of React
 * and canvas so the layout math and hit-testing are unit-testable on their own.
 */

// Column order matches the trace's layer order. COL maps a layer NAME to its
// column index; LAYER_KEYS maps a column index back to its name.
export const COL = { input: 0, hidden_1: 1, hidden_2: 2, output: 3 };
export const LAYER_KEYS = ["input", "hidden_1", "hidden_2", "output"];

// Logical drawing space (must match NetworkView's W/H backing store).
export const W = 960;
export const H = 560;

// Per-layer hit radius in logical px: input pixels are tiny and dense, hidden
// nodes are small, output nodes are large. Tuned so hover feels precise.
const HIT_RADIUS = [5, 6.5, 6.5, 20];

/**
 * Precompute a screen position for every neuron in every layer. Returns a
 * 4-element array of position arrays; input entries also carry their `cell`
 * size. This is the single source of truth for where neurons live on screen.
 */
export function computePositions() {
  const pos = [[], [], [], []];

  // Input: 28x28 grid drawn like the actual image.
  const inSide = 190, inLeft = 26, inTop = 120;
  const cell = inSide / 28;
  for (let i = 0; i < 784; i++) {
    const c = i % 28, r = (i / 28) | 0;
    pos[COL.input].push({ x: inLeft + (c + 0.5) * cell, y: inTop + (r + 0.5) * cell, cell });
  }

  // Hidden layers: 16 columns x 32 rows.
  const cols = 16, rows = 32;
  const hid = (left) => {
    const arr = [];
    const areaW = 118, areaH = 430, top = 70;
    for (let i = 0; i < 512; i++) {
      const c = i % cols, r = (i / cols) | 0;
      arr.push({ x: left + (c / (cols - 1)) * areaW, y: top + (r / (rows - 1)) * areaH });
    }
    return arr;
  };
  pos[COL.hidden_1] = hid(348);
  pos[COL.hidden_2] = hid(600);

  // Output: 10 nodes stacked vertically. Kept left enough that each node's
  // probability bar + percent label to its right fit inside the 960px canvas.
  const outX = 815, outTop = 90, outGap = 42;
  for (let i = 0; i < 10; i++) pos[COL.output].push({ x: outX, y: outTop + i * outGap });

  return pos;
}

/**
 * Find the neuron nearest to logical point (lx, ly), if any lands within its
 * layer's hit radius. Returns { layer, index, x, y } or null. When several
 * layers are in range, the closest relative to its own radius wins.
 */
export function hitTest(positions, lx, ly) {
  let best = null;
  for (let L = 0; L < positions.length; L++) {
    const pts = positions[L];
    let bi = -1, bd = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i].x - lx, dy = pts[i].y - ly;
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; bi = i; }
    }
    const radius = HIT_RADIUS[L];
    if (bi >= 0 && bd <= radius * radius) {
      const norm = Math.sqrt(bd) / radius; // 0 (dead center) .. 1 (edge)
      if (!best || norm < best.norm) {
        best = { layer: L, index: bi, x: pts[bi].x, y: pts[bi].y, norm };
      }
    }
  }
  return best;
}

/**
 * Human-readable label + value for a hovered neuron, using the trace's real
 * (unnormalized) activation values. Returns { label, value }.
 */
export function describeNeuron(trace, hovered) {
  const { layer, index } = hovered;
  const name = LAYER_KEYS[layer];
  const value = trace.layers[layer].activations[index];

  if (name === "input") {
    const r = (index / 28) | 0, c = index % 28;
    return {
      label: `INPUT · px (${r}, ${c})`,
      value: `intensity ${value.toFixed(2)}`,
      sub: "no activation (raw pixel)",
    };
  }
  if (name === "output") {
    const logit = trace.layers[layer].logits?.[index];
    const sub = logit == null
      ? null
      : `logit ${logit.toFixed(2)} → softmax → ${(value * 100).toFixed(1)}%`;
    return { label: `OUTPUT · digit ${index}`, value: `${(value * 100).toFixed(1)}% confidence`, sub };
  }
  // Hidden neuron: show its pre-activation z and how ReLU mapped it.
  const z = trace.layers[layer].preacts?.[index];
  const dead = z != null && z <= 0;
  const sub = z == null
    ? null
    : `z ${z.toFixed(2)} → ReLU → ${value.toFixed(2)}${dead ? " · dead" : ""}`;
  return { label: `${name.toUpperCase()} · #${index}`, value: `activation ${value.toFixed(2)}`, sub };
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/**
 * Map a hidden neuron's pre-activation z to a normalized point { x, y } in
 * [0,1] on a ReLU curve glyph: x spans z ∈ [-zMax, zMax], y spans a = relu(z)
 * ∈ [0, zMax] (0 at the bottom). Used to plot the neuron's spot on the curve.
 */
export function reluNormalizedPoint(z, zMax = 6) {
  const x = clamp01((z + zMax) / (2 * zMax));
  const a = z > 0 ? z : 0;
  const y = clamp01(a / zMax);
  return { x, y };
}

/**
 * Explanatory copy shown when hovering a layer's header/activation glyph: what
 * the layer does, what its size means, and what its activation function is and
 * why it's used here. Keyed by layer name.
 */
export const LAYER_INFO = {
  input: {
    title: "Input layer · the drawing",
    body:
      "Your drawing turned into numbers: a 28×28 grid of 784 pixels, each a " +
      "brightness from 0 (blank) to 1 (full ink). This layer has no activation " +
      "function — the pixel values are fed straight into Hidden 1 through weighted " +
      "connections. A brighter pixel simply sends a stronger signal.",
  },
  hidden_1: {
    title: "Hidden layer 1 · ReLU",
    body:
      "512 neurons that each hunt for a simple visual feature (an edge, a stroke, " +
      "a curve). A neuron multiplies every one of the 784 pixels by a learned " +
      "weight, adds them up (a 'weighted sum'), then runs the result through its " +
      "activation function, ReLU.\n\n" +
      "ReLU (Rectified Linear Unit) is just max(0, z): if that sum is negative the " +
      "neuron outputs 0 and stays silent; if positive, it passes the value through " +
      "unchanged. That simple bend at zero is what lets the network learn curved, " +
      "complex shapes — without a non-linear step like this, stacking layers would " +
      "be no more powerful than a single straight-line equation.",
  },
  hidden_2: {
    title: "Hidden layer 2 · ReLU",
    body:
      "Another 512 neurons, working exactly like Hidden 1 (weighted sum → ReLU), " +
      "except their inputs are Hidden 1's outputs instead of raw pixels. So this " +
      "layer combines simple features into higher-level ones — strokes and edges " +
      "becoming loops, curves, and whole parts of digits. Activation function is " +
      "again ReLU: max(0, z), keep positives, zero out negatives.",
  },
  output: {
    title: "Output layer · softmax",
    body:
      "10 neurons, one per digit 0–9. Each first produces a raw score called a " +
      "'logit' (a weighted sum of Hidden 2's outputs) — these can be any value, " +
      "positive or negative, and don't add up to anything meaningful on their own.\n\n" +
      "Softmax turns those 10 scores into probabilities: it exponentiates each one " +
      "and divides by their total, so every output becomes positive and all 10 sum " +
      "to 100%. That's what lets you read them as confidence and compare them — the " +
      "highest is the network's prediction.",
  },
};


/**
 * Describe where a set of input-pixel indices (row-major 28x28) is concentrated,
 * e.g. "top left", "center", "bottom". Used by the contributor summary.
 */
export function describeInputRegion(indices) {
  if (!indices.length) return "no clear region";
  let r = 0, c = 0;
  for (const i of indices) { r += (i / 28) | 0; c += i % 28; }
  r /= indices.length;
  c /= indices.length;
  const vert = r < 28 / 3 ? "top" : r > (2 * 28) / 3 ? "bottom" : "middle";
  const horiz = c < 28 / 3 ? "left" : c > (2 * 28) / 3 ? "right" : "center";
  if (vert === "middle" && horiz === "center") return "center";
  if (vert === "middle") return horiz;
  if (horiz === "center") return vert;
  return `${vert} ${horiz}`;
}

/**
 * The next-most-likely class besides `exclude`, as { digit, prob }. Lets the
 * summary say "next guess: 1 at 3%".
 */
export function runnerUp(probs, exclude) {
  let bestProb = -1, bestDigit = -1;
  for (let i = 0; i < probs.length; i++) {
    if (i === exclude) continue;
    if (probs[i] > bestProb) { bestProb = probs[i]; bestDigit = i; }
  }
  return { digit: bestDigit, prob: bestProb };
}
