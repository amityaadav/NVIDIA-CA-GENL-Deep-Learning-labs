import { useEffect, useMemo, useRef } from "react";
import { COL, computePositions } from "../lib/inspect.js";

const W = 960, H = 560;
const EXCITE = [79, 227, 238];   // activation / weight increasing
const INHIBIT = [224, 119, 110]; // negative / weight decreasing
const LEARN = [255, 196, 84];    // amber: how hard a neuron is being trained
const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
const mix = ([r1, g1, b1], [r2, g2, b2], t) => [
  Math.round(lerp(r1, r2, t)), Math.round(lerp(g1, g2, t)), Math.round(lerp(b1, b2, t)),
];
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

/**
 * Live network for the Train tab. Two modes on the same layout:
 *  · "activations" — a fixed digit's forward pass through the current model
 *  · "learning" — how hard each neuron/connection is being trained (gradients)
 * Both update on every training snapshot.
 */
export default function TrainNetworkView({ trace, learning, mode }) {
  const canvasRef = useRef(null);
  const positions = useMemo(computePositions, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, positions, trace, learning, mode);
  }, [positions, trace, learning, mode]);

  return <canvas ref={canvasRef} className="network-canvas" style={{ aspectRatio: `${W} / ${H}` }} />;
}

function draw(ctx, positions, trace, learning, mode) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, W, H);

  ctx.textAlign = "center";
  ctx.font = "600 12px ui-monospace, monospace";
  ctx.fillStyle = "#5c6f7e";
  [["INPUT", 141], ["HIDDEN 1", 408], ["HIDDEN 2", 649], ["OUTPUT", 862]]
    .forEach(([t, x]) => ctx.fillText(t, x, 26));

  if (!trace) {
    ctx.fillStyle = "#3a4a57";
    ctx.font = "500 15px system-ui, sans-serif";
    ctx.fillText("Press Train — the network appears here as it learns.", W / 2, H / 2);
    return;
  }

  const learn = mode === "learning" && learning;

  // Input digit (context in both modes).
  const input = trace.layers[0];
  for (let i = 0; i < input.size; i++) {
    const v = input.activations[i];
    if (v < 0.03) continue;
    const p = positions[COL.input][i];
    const g = Math.round(v * 255);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillRect(p.x - p.cell / 2, p.y - p.cell / 2, p.cell + 0.5, p.cell + 0.5);
  }

  // Edges: contributions (activations) or biggest weight updates (learning).
  const transitions = learn ? learning.edges : trace.transitions;
  for (const t of transitions) {
    const src = positions[COL[t.from]], dst = positions[COL[t.to]];
    for (const link of t.links) {
      const a = src[link.src], b = dst[link.dst];
      if (!a || !b) continue;
      ctx.strokeStyle = rgba(link.sign > 0 ? EXCITE : INHIBIT, 0.08 + link.strength * 0.5);
      ctx.lineWidth = 0.4 + link.strength * 1.7;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  // Hidden nodes.
  const accent = learn ? LEARN : EXCITE;
  for (const name of ["hidden_1", "hidden_2"]) {
    const L = COL[name];
    if (learn) {
      const vals = learning.nodes[name];
      const peak = Math.max(1e-9, ...vals);
      for (let i = 0; i < vals.length; i++) {
        const norm = clamp01(vals[i] / peak);
        const p = positions[L][i];
        ctx.fillStyle = rgba(mix([26, 37, 48], LEARN, norm), 0.3 + norm * 0.7);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.6 + norm * 2.7, 0, Math.PI * 2);
        ctx.fill();
      }
    } else {
      const layer = trace.layers[L];
      const peak = layer.peak || 1;
      const preacts = layer.preacts;
      for (let i = 0; i < layer.size; i++) {
        const p = positions[L][i];
        if (preacts && preacts[i] <= 0) {
          ctx.fillStyle = "rgba(184,142,142,0.4)";
          ctx.beginPath(); ctx.arc(p.x, p.y, 1.35, 0, Math.PI * 2); ctx.fill();
        } else {
          const norm = clamp01(layer.activations[i] / peak);
          ctx.fillStyle = rgba(mix([26, 37, 48], EXCITE, norm), 0.4 + norm * 0.6);
          ctx.beginPath(); ctx.arc(p.x, p.y, 1.7 + norm * 2.6, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  }

  drawOutput(ctx, positions, trace, learn ? learning.nodes.output : null, accent);
}

function drawOutput(ctx, positions, trace, learnVals, accent) {
  const out = trace.layers[3];
  const pred = trace.prediction;
  const peak = learnVals ? Math.max(1e-9, ...learnVals) : 1;

  for (let i = 0; i < 10; i++) {
    const p = positions[COL.output][i];
    const val = learnVals ? clamp01(learnVals[i] / peak) : out.activations[i];

    ctx.beginPath();
    ctx.arc(p.x, p.y, 15, 0, Math.PI * 2);
    ctx.fillStyle = rgba(mix([26, 37, 48], accent, val), 0.9);
    ctx.fill();
    if (!learnVals && i === pred) {
      ctx.strokeStyle = rgba(accent, 0.9);
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    ctx.fillStyle = val > 0.5 ? "#06131a" : "#8aa0ad";
    ctx.font = "700 14px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(DIGITS[i], p.x, p.y);
    ctx.textBaseline = "alphabetic";

    const barX = p.x + 24, barW = 52;
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(barX, p.y - 5, barW, 10);
    ctx.fillStyle = rgba(accent, 0.85);
    ctx.fillRect(barX, p.y - 5, barW * val, 10);
    if (!learnVals) {
      ctx.fillStyle = "#7c93a1";
      ctx.font = "600 11px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText(`${Math.round(out.activations[i] * 100)}%`, barX + barW + 8, p.y + 4);
    }
  }
}
