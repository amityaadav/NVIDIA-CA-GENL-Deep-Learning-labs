import { useEffect, useMemo, useRef } from "react";

// Logical drawing space; the canvas element scales responsively via CSS while
// the backing store is sized for the device pixel ratio for crisp rendering.
const W = 960;
const H = 560;

const COL = { input: 0, hidden_1: 1, hidden_2: 2, output: 3 };
const EXCITE = [79, 227, 238]; // cyan  -> positive contribution
const INHIBIT = [224, 119, 110]; // warm -> negative contribution
const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
const mix = ([r1, g1, b1], [r2, g2, b2], t) => [
  Math.round(lerp(r1, r2, t)),
  Math.round(lerp(g1, g2, t)),
  Math.round(lerp(b1, b2, t)),
];
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

/**
 * Precompute a screen position for every neuron in every layer.
 * Input is laid out as a 28x28 image; the 512-wide hidden layers as dense
 * 16x32 grids; the output as ten labelled nodes.
 */
function computePositions() {
  const pos = [[], [], [], []];

  // Input: 28x28 grid of cells drawn like the actual image.
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

  // Output: 10 nodes stacked vertically.
  const outX = 872, outTop = 90, outGap = 42;
  for (let i = 0; i < 10; i++) pos[COL.output].push({ x: outX, y: outTop + i * outGap });

  return pos;
}

export default function NetworkView({ trace, progress }) {
  const canvasRef = useRef(null);
  const positions = useMemo(computePositions, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, positions, trace, progress);
  }, [positions, trace, progress]);

  return <canvas ref={canvasRef} className="network-canvas" style={{ aspectRatio: `${W} / ${H}` }} />;
}

function draw(ctx, positions, trace, progress) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, W, H);

  // Column headers.
  ctx.textAlign = "center";
  ctx.font = "600 12px ui-monospace, monospace";
  ctx.fillStyle = "#5c6f7e";
  const heads = [
    ["INPUT · 28×28", 121],
    ["HIDDEN · 512", 407],
    ["HIDDEN · 512", 659],
    ["OUTPUT · 10", 872],
  ];
  for (const [label, x] of heads) ctx.fillText(label, x, 34);

  if (!trace) {
    ctx.fillStyle = "#3a4a57";
    ctx.font = "500 15px system-ui, sans-serif";
    ctx.fillText("Draw a digit, then run inference to watch the forward pass.", W / 2, H / 2);
    return;
  }

  const layerLit = (L) => clamp01(progress - L);
  const travel = (k) => clamp01(progress - (k + 1));

  // Edges (top weighted paths) drawn behind the neurons.
  trace.transitions.forEach((t, k) => {
    const tv = travel(k);
    if (tv <= 0) return;
    const src = positions[COL[t.from]];
    const dst = positions[COL[t.to]];
    for (const link of t.links) {
      const a = src[link.src];
      const b = dst[link.dst];
      if (!a || !b) continue;
      const color = link.sign > 0 ? EXCITE : INHIBIT;
      const s = link.strength;
      // Faint full connection.
      ctx.strokeStyle = rgba(color, 0.06 + s * 0.14);
      ctx.lineWidth = 0.4 + s * 1.2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      // Bright signal head travelling from source to destination.
      const hx = lerp(a.x, b.x, tv);
      const hy = lerp(a.y, b.y, tv);
      ctx.strokeStyle = rgba(color, (0.25 + s * 0.6) * (tv < 1 ? 1 : 0.7));
      ctx.lineWidth = 0.6 + s * 2.2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(hx, hy);
      ctx.stroke();
    }
  });

  // Input layer as a grayscale image.
  const input = trace.layers[0];
  const litIn = layerLit(0);
  for (let i = 0; i < input.size; i++) {
    const v = input.activations[i] * litIn;
    if (v < 0.02) continue;
    const p = positions[COL.input][i];
    const g = Math.round(v * 255);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillRect(p.x - p.cell / 2, p.y - p.cell / 2, p.cell + 0.5, p.cell + 0.5);
  }

  // Hidden layers as glowing nodes.
  for (const name of ["hidden_1", "hidden_2"]) {
    const L = COL[name];
    const layer = trace.layers[L];
    const lit = layerLit(L);
    if (lit <= 0) continue;
    const peak = layer.peak || 1;
    for (let i = 0; i < layer.size; i++) {
      const norm = clamp01(layer.activations[i] / peak) * lit;
      const p = positions[L][i];
      const col = mix([26, 37, 48], EXCITE, norm);
      ctx.fillStyle = rgba(col, 0.35 + norm * 0.65);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.7 + norm * 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawOutput(ctx, positions, trace, layerLit(3), progress);
}

function drawOutput(ctx, positions, trace, lit, progress) {
  if (lit <= 0) return;
  const out = trace.layers[3];
  const pred = trace.prediction;

  for (let i = 0; i < 10; i++) {
    const p = positions[COL.output][i];
    const prob = out.activations[i];
    const isPred = i === pred && progress > 3.5;

    // Node.
    const fill = mix([26, 37, 48], EXCITE, prob * lit);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 15, 0, Math.PI * 2);
    ctx.fillStyle = rgba(fill, 0.9);
    ctx.fill();
    if (isPred) {
      ctx.strokeStyle = rgba(EXCITE, 0.9);
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Digit label inside.
    ctx.fillStyle = prob * lit > 0.5 ? "#06131a" : "#8aa0ad";
    ctx.font = "700 14px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(DIGITS[i], p.x, p.y);
    ctx.textBaseline = "alphabetic";

    // Probability bar + percent to the right.
    const barX = p.x + 24, barW = 52;
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(barX, p.y - 5, barW, 10);
    ctx.fillStyle = rgba(EXCITE, 0.85);
    ctx.fillRect(barX, p.y - 5, barW * prob * lit, 10);
    ctx.fillStyle = "#7c93a1";
    ctx.font = "600 11px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${Math.round(prob * 100)}%`, barX + barW + 8, p.y + 4);
  }
}
