/**
 * In-browser MNIST inference — a faithful port of the Python backend
 * (inference.py) so the app runs fully client-side with no server. Produces the
 * exact same trace shapes the UI already consumes: predict / explain / neuron.
 *
 * Weights are loaded once from a flat float32 blob (see backend/export_weights.py).
 */

const IN = 784, H = 512, OUT = 10;

// Must match inference.py.
const TOP_EDGES_PER_TRANSITION = 40;
const TRACE_TOP_HIDDEN2 = 6;
const TRACE_FANOUT_HIDDEN1 = 3;
const TRACE_FANOUT_INPUT = 3;
const NEURON_TOP_TERMS = 8;

const round5 = (v) => Math.round(v * 1e5) / 1e5;
const round4 = (v) => Math.round(v * 1e4) / 1e4;

let _modelPromise = null;

/** Load (once) and return the model with predict/explain/neuron methods. */
export function getModel() {
  if (!_modelPromise) {
    const url = `${import.meta.env.BASE_URL}model/weights.f32`;
    _modelPromise = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load model weights (${r.status})`);
        return r.arrayBuffer();
      })
      .then((buf) => new Model(new Float32Array(buf)));
  }
  return _modelPromise;
}

export class Model {
  constructor(f) {
    let o = 0;
    const take = (n) => { const a = f.subarray(o, o + n); o += n; return a; };
    this.W1 = take(H * IN); this.b1 = take(H);   // hidden_1: [512,784], [512]
    this.W2 = take(H * H);  this.b2 = take(H);    // hidden_2: [512,512], [512]
    this.W3 = take(OUT * H); this.b3 = take(OUT); // output:   [10,512],  [10]
  }

  /** Forward pass → { input, z1, h1, z2, h2, logits, probs } (Float32-ish). */
  _forward(pixels) {
    const input = Float32Array.from(pixels);
    const z1 = dense(this.W1, this.b1, input, H, IN);
    const h1 = relu(z1);
    const z2 = dense(this.W2, this.b2, h1, H, H);
    const h2 = relu(z2);
    const logits = dense(this.W3, this.b3, h2, OUT, H);
    const probs = softmax(logits);
    return { input, z1, h1, z2, h2, logits, probs };
  }

  predict(pixels) {
    const s = this._forward(pixels);
    return {
      prediction: argmax(s.probs),
      probs: Array.from(s.probs, round5),
      layers: [
        layerPayload("input", s.input, { shape: [28, 28] }),
        layerPayload("hidden_1", s.h1, { preacts: s.z1 }),
        layerPayload("hidden_2", s.h2, { preacts: s.z2 }),
        layerPayload("output", s.probs, { logits: s.logits }),
      ],
      transitions: [
        topEdges("input", "hidden_1", s.input, this.W1, IN, H),
        topEdges("hidden_1", "hidden_2", s.h1, this.W2, H, H),
        topEdges("hidden_2", "output", s.h2, this.W3, H, OUT),
      ],
    };
  }

  explain(pixels, target) {
    const s = this._forward(pixels);

    // 1) output[target] <- hidden_2
    const h2Contrib = new Float32Array(H);
    for (let i = 0; i < H; i++) h2Contrib[i] = s.h2[i] * this.W3[target * H + i];
    const [h2Sel, edgesOut] = beamFromSingle(h2Contrib, target, "hidden_2", "output", TRACE_TOP_HIDDEN2);

    // 2) hidden_2[sel] <- hidden_1
    const [h1Sel, edgesH2] = beamFanout(s.h1, this.W2, H, h2Sel, "hidden_1", "hidden_2", TRACE_FANOUT_HIDDEN1);

    // 3) hidden_1[sel] <- input
    const [inSel, edgesIn] = beamFanout(s.input, this.W1, IN, h1Sel, "input", "hidden_1", TRACE_FANOUT_INPUT);

    return {
      target,
      prediction: argmax(s.probs),
      targetProb: round5(s.probs[target]),
      nodes: { hidden_2: h2Sel, hidden_1: h1Sel, input: inSel },
      edges: [...edgesOut, ...edgesH2, ...edgesIn],
    };
  }

  neuron(pixels, layer, index) {
    const s = this._forward(pixels);
    const cfg = {
      hidden_1: { src: s.input, w: this.W1, nSrc: IN, z: s.z1, a: s.h1, sourceLayer: "input" },
      hidden_2: { src: s.h1, w: this.W2, nSrc: H, z: s.z2, a: s.h2, sourceLayer: "hidden_1" },
      output: { src: s.h2, w: this.W3, nSrc: H, z: s.logits, a: s.probs, sourceLayer: "hidden_2" },
    }[layer];
    const bias = layer === "hidden_1" ? this.b1[index] : layer === "hidden_2" ? this.b2[index] : this.b3[index];

    const base = index * cfg.nSrc;
    const contrib = new Float32Array(cfg.nSrc);
    for (let i = 0; i < cfg.nSrc; i++) contrib[i] = cfg.w[base + i] * cfg.src[i];
    const idx = topKIndices(contrib, NEURON_TOP_TERMS);
    const topTerms = idx.map((i) => ({
      src: i,
      weight: round5(cfg.w[base + i]),
      value: round5(cfg.src[i]),
      contribution: round5(contrib[i]),
    }));

    const result = {
      layer,
      index,
      sourceLayer: cfg.sourceLayer,
      activation: layer === "output" ? "softmax" : "relu",
      bias: round5(bias),
      z: round5(cfg.z[index]),
      a: round5(cfg.a[index]),
      topTerms,
    };
    if (layer === "hidden_1") {
      result.weightImage = Array.from(cfg.w.subarray(base, base + cfg.nSrc), round5);
    }
    return result;
  }
}

// --- math ---

function dense(W, b, x, nOut, nIn) {
  const out = new Float32Array(nOut);
  for (let j = 0; j < nOut; j++) {
    let sum = b[j];
    const row = j * nIn;
    for (let i = 0; i < nIn; i++) sum += W[row + i] * x[i];
    out[j] = sum;
  }
  return out;
}

function relu(v) {
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] > 0 ? v[i] : 0;
  return out;
}

function softmax(v) {
  let m = -Infinity;
  for (const x of v) if (x > m) m = x;
  const out = new Float32Array(v.length);
  let sum = 0;
  for (let i = 0; i < v.length; i++) { out[i] = Math.exp(v[i] - m); sum += out[i]; }
  for (let i = 0; i < v.length; i++) out[i] /= sum;
  return out;
}

function argmax(v) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
  return bi;
}

// --- trace builders (mirror inference.py) ---

function layerPayload(name, values, { shape, preacts, logits } = {}) {
  let peak = values.length ? -Infinity : 1;
  for (const v of values) if (v > peak) peak = v;
  const payload = {
    name,
    size: values.length,
    activations: Array.from(values, round5),
    peak: round5(peak),
  };
  if (shape) payload.shape = shape;
  if (preacts) payload.preacts = Array.from(preacts, round5);
  if (logits) payload.logits = Array.from(logits, round5);
  return payload;
}

/** Top edges src_i -> dst_j by |weight[j,i] * srcAct[i]|. weight is flat [dst*nSrc + src]. */
function topEdges(from, to, srcAct, weight, nSrc, nDst) {
  const cands = [];
  for (let i = 0; i < nSrc; i++) {
    const a = srcAct[i];
    if (a === 0) continue; // zeros never crack the top-k for a real input
    for (let j = 0; j < nDst; j++) {
      const signed = weight[j * nSrc + i] * a;
      cands.push({ src: i, dst: j, signed, mag: Math.abs(signed) });
    }
  }
  cands.sort((x, y) => y.mag - x.mag);
  const top = cands.slice(0, TOP_EDGES_PER_TRANSITION);
  const maxV = top.length && top[0].mag > 0 ? top[0].mag : 1;
  return {
    from,
    to,
    links: top.map((c) => ({
      src: c.src,
      dst: c.dst,
      strength: round4(c.mag / maxV),
      sign: c.signed >= 0 ? 1 : -1,
    })),
  };
}

function beamFromSingle(contrib, dstId, from, to, k) {
  const sel = topKIndices(contrib, k);
  const maxV = sel.length && Math.abs(contrib[sel[0]]) > 0 ? Math.abs(contrib[sel[0]]) : 1;
  const edges = sel.map((i) => edge(from, to, i, dstId, contrib[i], maxV));
  return [sel, edges];
}

function beamFanout(srcAct, weight, nSrc, dsts, from, to, k) {
  const raw = [];
  for (const dst of dsts) {
    const base = dst * nSrc;
    const contrib = new Float32Array(nSrc);
    for (let i = 0; i < nSrc; i++) contrib[i] = srcAct[i] * weight[base + i];
    for (const i of topKIndices(contrib, k)) raw.push({ src: i, dst, signed: contrib[i] });
  }
  if (!raw.length) return [[], []];
  let maxV = 0;
  for (const r of raw) maxV = Math.max(maxV, Math.abs(r.signed));
  maxV = maxV || 1;
  const sel = [...new Set(raw.map((r) => r.src))].sort((a, b) => a - b);
  const edges = raw.map((r) => edge(from, to, r.src, r.dst, r.signed, maxV));
  return [sel, edges];
}

function edge(from, to, src, dst, signed, maxV) {
  return { from, to, src, dst, strength: round4(Math.abs(signed) / maxV), sign: signed >= 0 ? 1 : -1 };
}

/** Indices of the k largest-|value| entries, largest first. */
function topKIndices(v, k) {
  const idx = Array.from(v, (_, i) => i);
  idx.sort((a, b) => Math.abs(v[b]) - Math.abs(v[a]));
  return idx.slice(0, Math.min(k, v.length));
}
