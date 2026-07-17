import { useEffect, useMemo, useRef, useState } from "react";
import {
  COL,
  LAYER_KEYS,
  LAYER_INFO,
  W,
  H,
  computePositions,
  hitTest,
  hitInfoBand,
  describeNeuron,
  reluNormalizedPoint,
} from "../lib/inspect.js";

const EXCITE = [79, 227, 238]; // cyan  -> positive contribution
const INHIBIT = [224, 119, 110]; // warm -> negative contribution
const DEAD = [184, 142, 142]; // muted -> neuron clipped to 0 by ReLU
const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

// Activation-function glyphs drawn under each layer header. GLYPHS index aligns
// with COL so a hovered hidden neuron maps straight to its curve.
const GLYPHS = [
  { cx: 141, kind: "identity" }, // input: raw pixels, no activation
  { cx: 408, kind: "relu" },     // hidden_1
  { cx: 649, kind: "relu" },     // hidden_2
  { cx: 862, kind: "softmax" },  // output
];
const GW = 62, GH = 18, GY = 32; // glyph box: width, height, top y
const INFO_TIP_Y = 58; // logical y where a layer's info tooltip anchors (below glyph)
const INFO_TIP_W = 300; // fixed CSS px width of the info tooltip (matches .info-tip)

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
const mix = ([r1, g1, b1], [r2, g2, b2], t) => [
  Math.round(lerp(r1, r2, t)),
  Math.round(lerp(g1, g2, t)),
  Math.round(lerp(b1, b2, t)),
];
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

export default function NetworkView({ trace, progress, focus, prep, onPickOutput, onInspect }) {
  const canvasRef = useRef(null);
  const positions = useMemo(computePositions, []);
  const [hovered, setHovered] = useState(null);
  const [info, setInfo] = useState(null); // hovered layer header/glyph explanation

  // Precompute the focused sub-network as fast membership sets per layer.
  const focusSets = useMemo(() => {
    if (!focus) return null;
    return {
      sets: {
        input: new Set(focus.nodes.input),
        hidden_1: new Set(focus.nodes.hidden_1),
        hidden_2: new Set(focus.nodes.hidden_2),
        output: new Set([focus.target]),
      },
      edges: focus.edges,
      target: focus.target,
    };
  }, [focus]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, positions, trace, progress, hovered, focusSets, prep);
  }, [positions, trace, progress, hovered, focusSets, prep]);

  const toLogical = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      rect,
      lx: ((e.clientX - rect.left) / rect.width) * W,
      ly: ((e.clientY - rect.top) / rect.height) * H,
    };
  };

  const onMove = (e) => {
    const { rect, lx, ly } = toLogical(e);

    // Header/glyph band shows the layer explanation (works even before a draw).
    const band = hitInfoBand(lx, ly);
    if (band) {
      // Clamp the (fixed-width, centered) tooltip so an edge column's box stays
      // fully on-stage and wraps the same way a middle column's does.
      const half = INFO_TIP_W / 2, pad = 10;
      const minL = half + pad, maxL = rect.width - half - pad;
      const center = (band.cx / W) * rect.width;
      band.px = maxL > minL ? Math.max(minL, Math.min(maxL, center)) : rect.width / 2;
      band.py = (INFO_TIP_Y / H) * rect.height;
      setInfo(band);
      if (hovered) setHovered(null);
      return;
    }
    if (info) setInfo(null);

    if (!trace) return;
    const hit = hitTest(positions, lx, ly);
    if (!hit) {
      setHovered((h) => (h ? null : h));
      return;
    }
    // Store both logical hit and its on-screen pixel position for the tooltip.
    hit.px = (hit.x / W) * rect.width;
    hit.py = (hit.y / H) * rect.height;
    setHovered(hit);
  };

  const onLeave = () => {
    setHovered(null);
    setInfo(null);
  };

  // Click an output digit to trace it; click a hidden neuron to inspect it;
  // click empty space to clear both.
  const onClick = (e) => {
    if (!trace) return;
    const { lx, ly } = toLogical(e);
    const hit = hitTest(positions, lx, ly);
    if (hit && hit.layer === COL.output) {
      onPickOutput?.(hit.index);
    } else if (hit && (hit.layer === COL.hidden_1 || hit.layer === COL.hidden_2)) {
      onInspect?.({ layer: LAYER_KEYS[hit.layer], index: hit.index });
    } else {
      onPickOutput?.(null);
      onInspect?.(null);
    }
  };

  const tip = hovered && trace ? describeNeuron(trace, hovered) : null;
  const infoTip = info ? LAYER_INFO[LAYER_KEYS[info.layer]] : null;
  const clickable = hovered && (hovered.layer === COL.output || hovered.layer === COL.hidden_1 || hovered.layer === COL.hidden_2);
  const cursor = info ? "help" : clickable ? "pointer" : "crosshair";

  return (
    <div className="network-stage">
      <canvas
        ref={canvasRef}
        className="network-canvas"
        style={{ aspectRatio: `${W} / ${H}`, cursor }}
        onMouseMove={onMove}
        onMouseLeave={onLeave}
        onClick={onClick}
      />
      {tip && (
        <div className="neuron-tip" style={{ left: `${hovered.px}px`, top: `${hovered.py}px` }}>
          <span className="neuron-tip-label">{tip.label}</span>
          <span className="neuron-tip-value">{tip.value}</span>
          {tip.sub && <span className="neuron-tip-sub">{tip.sub}</span>}
        </div>
      )}
      {infoTip && (
        <div className="info-tip" style={{ left: `${info.px}px`, top: `${info.py}px` }}>
          <span className="info-tip-title">{infoTip.title}</span>
          <span className="info-tip-body">{infoTip.body}</span>
        </div>
      )}
    </div>
  );
}

function draw(ctx, positions, trace, progress, hovered, focus, prep) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, W, H);

  // Column headers.
  ctx.textAlign = "center";
  ctx.font = "600 12px ui-monospace, monospace";
  ctx.fillStyle = "#5c6f7e";
  // x's are the centers of each layer's compartment (between the separators at
  // 282/533/765, bounded by the canvas edges) so headers sit centered in-column.
  const heads = [
    ["INPUT · 28×28 = 784 px", 141],
    ["HIDDEN 1 · 512 nodes · ReLU", 408],
    ["HIDDEN 2 · 512 nodes · ReLU", 649],
    ["OUTPUT · 10 · softmax", 862],
  ];
  for (const [label, x] of heads) ctx.fillText(label, x, 26);

  // Structural scaffolding, always visible: dividers between the four layers,
  // the full 28x28 input grid, and each layer's activation-function glyph.
  drawSeparators(ctx);
  drawInputGrid(ctx, positions);
  drawActivationGlyphs(ctx, trace, hovered);

  if (!trace) {
    ctx.fillStyle = "#3a4a57";
    ctx.font = "500 15px system-ui, sans-serif";
    ctx.fillText("Draw a digit, then run inference to watch the forward pass.", W / 2, H / 2);
    return;
  }

  // Normalization morph plays in the input area before the forward pass.
  if (prep) {
    drawPreprocess(ctx, positions, prep);
    return;
  }

  const layerLit = (L) => clamp01(progress - L);
  const travel = (k) => clamp01(progress - (k + 1));

  // While a class is focused ("why this digit?"), everything outside the
  // responsible sub-network fades so the path that drove it stands out.
  const DIM = 0.12;
  const dimFor = (name, i) => (focus && !focus.sets[name].has(i) ? DIM : 1);

  // When hovering a neuron, edges touching it are emphasized and the rest are
  // dimmed so you can read that neuron's connections.
  const emphasis = (t, link) => {
    if (!hovered) return 1;
    const touches =
      (COL[t.from] === hovered.layer && link.src === hovered.index) ||
      (COL[t.to] === hovered.layer && link.dst === hovered.index);
    return touches ? 1.7 : 0.22;
  };

  if (focus) {
    // Focused: draw only the backward sub-network's edges, brightly.
    drawFocusEdges(ctx, positions, focus);
  } else {
    // Default: the forward top-weighted paths, animated by progress.
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
        const em = emphasis(t, link);
        // Faint full connection.
        ctx.strokeStyle = rgba(color, clamp01((0.06 + s * 0.14) * em));
        ctx.lineWidth = (0.4 + s * 1.2) * (em > 1 ? 1.5 : 1);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        // Bright signal head travelling from source to destination.
        const hx = lerp(a.x, b.x, tv);
        const hy = lerp(a.y, b.y, tv);
        ctx.strokeStyle = rgba(color, clamp01((0.25 + s * 0.6) * (tv < 1 ? 1 : 0.7) * em));
        ctx.lineWidth = (0.6 + s * 2.2) * (em > 1 ? 1.5 : 1);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(hx, hy);
        ctx.stroke();
      }
    });
  }

  // Input layer as a grayscale image.
  const input = trace.layers[0];
  const litIn = layerLit(0);
  for (let i = 0; i < input.size; i++) {
    const v = input.activations[i] * litIn * dimFor("input", i);
    if (v < 0.02) continue;
    const p = positions[COL.input][i];
    const g = Math.round(v * 255);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    // Fill inside the grid box so its faint border stays visible.
    ctx.fillRect(p.x - p.cell / 2, p.y - p.cell / 2, p.cell - 0.5, p.cell - 0.5);
  }

  // Hidden layers as glowing nodes. Neurons ReLU clipped to zero (pre-activation
  // <= 0) are drawn as muted "dead" markers so the layer's sparsity shows (C).
  for (const name of ["hidden_1", "hidden_2"]) {
    const L = COL[name];
    const layer = trace.layers[L];
    const lit = layerLit(L);
    if (lit <= 0) continue;
    const peak = layer.peak || 1;
    const preacts = layer.preacts;
    for (let i = 0; i < layer.size; i++) {
      const p = positions[L][i];
      const d = dimFor(name, i);
      const dead = preacts ? preacts[i] <= 0 : layer.activations[i] <= 0;
      if (dead) {
        ctx.fillStyle = rgba(DEAD, 0.42 * lit * d);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.35, 0, Math.PI * 2);
        ctx.fill();
      } else {
        const norm = clamp01(layer.activations[i] / peak) * lit;
        const col = mix([26, 37, 48], EXCITE, norm);
        ctx.fillStyle = rgba(col, (0.4 + norm * 0.6) * d);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.7 + norm * 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  drawOutput(ctx, positions, trace, layerLit(3), progress, focus);

  // Ring around the hovered neuron so it's clear what the tooltip refers to.
  if (hovered) {
    const p = positions[hovered.layer][hovered.index];
    if (p) {
      const r = hovered.layer === 0 ? 4 : hovered.layer === 3 ? 17 : 5;
      ctx.strokeStyle = "rgba(231,237,243,0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

function drawOutput(ctx, positions, trace, lit, progress, focus) {
  if (lit <= 0) return;
  const out = trace.layers[3];
  const pred = trace.prediction;
  const logits = out.logits;

  // The output phase (lit: 0->1) plays the "softmax moment": bars first rise to
  // the raw LOGIT heights, then reshape (squash + normalize) into probabilities.
  const op = lit;
  const grow = clamp01(op / 0.35);            // bars rise in
  const morph = clamp01((op - 0.5) / 0.5);    // logits -> probabilities
  let minL = Infinity, maxL = -Infinity;
  if (logits) for (const l of logits) { if (l < minL) minL = l; if (l > maxL) maxL = l; }
  const lrange = maxL - minL || 1;

  for (let i = 0; i < 10; i++) {
    const p = positions[COL.output][i];
    const prob = out.activations[i];
    const logitFrac = logits ? (logits[i] - minL) / lrange : prob;
    const frac = (logits ? lerp(logitFrac, prob, morph) : prob) * grow;
    const isPred = i === pred && progress > 3.5;
    // Under focus, the focused digit stays lit and the rest fade back.
    const focused = focus ? i === focus.target : null;
    const d = focus && !focused ? 0.18 : 1;

    // Node.
    const fill = mix([26, 37, 48], EXCITE, frac);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 15, 0, Math.PI * 2);
    ctx.fillStyle = rgba(fill, 0.9 * d);
    ctx.fill();
    if (isPred || focused) {
      ctx.strokeStyle = rgba(EXCITE, 0.9);
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Digit label inside.
    ctx.fillStyle = frac > 0.5 ? "#06131a" : `rgba(138,160,173,${d})`;
    ctx.font = "700 14px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(DIGITS[i], p.x, p.y);
    ctx.textBaseline = "alphabetic";

    // Bar, and label: the raw logit while in logit form, the % once softmaxed.
    const barX = p.x + 24, barW = 52;
    ctx.fillStyle = `rgba(255,255,255,${0.06 * d})`;
    ctx.fillRect(barX, p.y - 5, barW, 10);
    ctx.fillStyle = rgba(EXCITE, 0.85 * d);
    ctx.fillRect(barX, p.y - 5, barW * frac, 10);
    const label = !logits
      ? `${(prob * 100).toFixed(2)}%`
      : morph < 0.5 ? logits[i].toFixed(1) : `${(prob * 100).toFixed(2)}%`;
    ctx.fillStyle = `rgba(124,147,161,${d})`;
    ctx.font = "600 11px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText(label, barX + barW + 8, p.y + 4);
  }

  // Caption naming what the bars represent right now.
  if (logits && op > 0.02) {
    ctx.fillStyle = `rgba(140,170,190,${clamp01(op * 3) * 0.85})`;
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText(
      morph < 0.5 ? "raw scores (logits)" : "softmax → probabilities · sum 100%",
      862, 500,
    );
  }
}

/** Vertical dividers separating the four layer columns. */
function drawSeparators(ctx) {
  const xs = [282, 533, 765]; // midpoints between input|h1, h1|h2, h2|output
  ctx.strokeStyle = "rgba(140,170,190,0.28)";
  ctx.lineWidth = 1.5;
  for (const x of xs) {
    ctx.beginPath();
    ctx.moveTo(x, 44);
    ctx.lineTo(x, H - 16);
    ctx.stroke();
  }
}

/** Activation-function glyphs (A) under each header: ReLU curves for the hidden
 *  layers, a softmax distribution for the output, an identity line for the raw
 *  input. When a hidden neuron is hovered, its z→a point is plotted on the curve (B). */
function drawActivationGlyphs(ctx, trace, hovered) {
  for (const g of GLYPHS) {
    const x0 = g.cx - GW / 2, base = GY + GH;
    if (g.kind === "relu") {
      ctx.strokeStyle = "rgba(140,170,190,0.18)";
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x0, base); ctx.lineTo(x0 + GW, base); ctx.stroke(); // baseline
      ctx.strokeStyle = "rgba(120,200,210,0.75)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x0, base);        // flat at 0 for z < 0
      ctx.lineTo(g.cx, base);      // kink at z = 0
      ctx.lineTo(x0 + GW, GY);     // linear for z > 0
      ctx.stroke();
    } else if (g.kind === "softmax") {
      const n = 7, gap = 2, bw = (GW - (n - 1) * gap) / n;
      const hs = [0.18, 0.32, 0.5, 1.0, 0.55, 0.28, 0.14];
      ctx.fillStyle = "rgba(120,200,210,0.65)";
      for (let i = 0; i < n; i++) {
        ctx.fillRect(x0 + i * (bw + gap), base - hs[i] * GH, bw, hs[i] * GH);
      }
    } else {
      ctx.strokeStyle = "rgba(140,170,190,0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, base); ctx.lineTo(x0 + GW, GY); ctx.stroke(); // identity
    }
  }

  // (B) Plot the hovered hidden neuron's pre-activation on its ReLU curve.
  if (trace && hovered && (hovered.layer === COL.hidden_1 || hovered.layer === COL.hidden_2)) {
    const g = GLYPHS[hovered.layer];
    const preacts = trace.layers[hovered.layer].preacts;
    if (preacts) {
      const pt = reluNormalizedPoint(preacts[hovered.index]);
      const x0 = g.cx - GW / 2, base = GY + GH;
      const px = x0 + pt.x * GW;
      const py = base - pt.y * GH;
      ctx.setLineDash([2, 2]);
      ctx.strokeStyle = "rgba(231,237,243,0.4)";
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(px, base); ctx.lineTo(px, py); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath(); ctx.arc(px, py, 2.4, 0, Math.PI * 2); ctx.fill();
    }
  }
}

/** The input layer's full 28x28 grid: 784 small boxes, drawn empty so the
 *  drawing's ink (rendered later) sits inside a visible pixel grid. */
function drawInputGrid(ctx, positions) {
  const cells = positions[COL.input];
  ctx.strokeStyle = "rgba(120,150,170,0.14)";
  ctx.lineWidth = 0.4;
  for (let i = 0; i < cells.length; i++) {
    const p = cells[i];
    const s = p.cell;
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(p.x - s / 2, p.y - s / 2, s - 0.5, s - 0.5);
    ctx.strokeRect(p.x - s / 2, p.y - s / 2, s - 0.5, s - 0.5);
  }
  // Boundary around the whole 28x28 grid so the input field's extent is clear.
  const s = cells[0].cell;
  const x0 = cells[0].x - s / 2, y0 = cells[0].y - s / 2;
  const x1 = cells[cells.length - 1].x + s / 2, y1 = cells[cells.length - 1].y + s / 2;
  ctx.strokeStyle = "rgba(140,170,190,0.45)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
}

/**
 * Animate MNIST normalization in the input area: the raw drawing (at its real
 * position/size) morphs to the 20px, center-of-mass-centered crop the network
 * actually receives. Ends aligned with the 28x28 grid so it hands off cleanly.
 */
function drawPreprocess(ctx, positions, prep) {
  const { geometry: g, inkCanvas, t } = prep;
  // The geometric morph happens in the first MORPH_FRAC of the timeline; the
  // remainder holds so the stacked step captions stay readable.
  const MORPH_FRAC = 0.36;
  const mt = clamp01(t / MORPH_FRAC);
  const ease = mt < 0.5 ? 2 * mt * mt : 1 - Math.pow(-2 * mt + 2, 2) / 2; // easeInOutQuad

  const cells = positions[COL.input];
  const cell = cells[0].cell;
  const gx0 = cells[0].x - cell / 2, gy0 = cells[0].y - cell / 2; // grid top-left
  const src2area = (28 * cell) / g.w; // source (drawing) px -> input-area px

  // Start: the drawing at its actual place/size within the input frame.
  const sx = gx0 + g.minX * src2area, sy = gy0 + g.minY * src2area;
  const sw = g.boxW * src2area, sh = g.boxH * src2area;

  // End: the normalized crop — 20px-scaled, COM-centered at (14,14) in 28-space.
  const dx28 = 14 - (g.comX - g.minX) * g.scale;
  const dy28 = 14 - (g.comY - g.minY) * g.scale;
  const ex = gx0 + dx28 * cell, ey = gy0 + dy28 * cell;
  const ew = g.boxW * g.scale * cell, eh = g.boxH * g.scale * cell;

  const lerp2 = (a, b) => a + (b - a) * ease;
  const rx = lerp2(sx, ex), ry = lerp2(sy, ey), rw = lerp2(sw, ew), rh = lerp2(sh, eh);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(inkCanvas, g.minX, g.minY, g.boxW, g.boxH, rx, ry, rw, rh);

  // Crop outline that travels with the ink.
  ctx.strokeStyle = "rgba(79,227,238,0.7)";
  ctx.lineWidth = 1;
  ctx.strokeRect(rx, ry, rw, rh);

  // Step captions: each distinct line fades in below the previous (scrolling
  // downward) as its step happens, then all linger through the hold.
  const baseY = gy0 + 28 * cell + 18;
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(140,170,190,0.9)";
  ctx.font = "600 11px ui-monospace, monospace";
  ctx.fillText("normalizing input", 121, baseY);

  const steps = [
    { at: 0.04, text: "1 · crop to the ink" },
    { at: 0.16, text: "2 · scale to 20 px" },
    { at: 0.28, text: "3 · center by mass" },
  ];
  ctx.font = "500 11px ui-monospace, monospace";
  steps.forEach((s, i) => {
    const a = clamp01((t - s.at) / 0.05);
    if (a <= 0) return;
    ctx.fillStyle = `rgba(120,200,210,${a})`;
    ctx.fillText(s.text, 121, baseY + 18 + i * 16);
  });
}

/** Draw the focused "why this digit?" sub-network edges, brightly. */
function drawFocusEdges(ctx, positions, focus) {
  for (const e of focus.edges) {
    const a = positions[COL[e.from]][e.src];
    const b = positions[COL[e.to]][e.dst];
    if (!a || !b) continue;
    const color = e.sign > 0 ? EXCITE : INHIBIT;
    ctx.strokeStyle = rgba(color, clamp01(0.35 + e.strength * 0.6));
    ctx.lineWidth = 0.8 + e.strength * 2.4;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}
