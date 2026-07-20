import { useEffect, useMemo, useRef, useState } from "react";
import {
  COL,
  LAYER_KEYS,
  LAYER_INFO,
  NORMALIZE_INFO,
  W,
  H,
  computePositions,
  hitTest,
  describeNeuron,
  reluNormalizedPoint,
} from "../lib/inspect.js";
import { registerPopover, closeOtherPopovers } from "../lib/popovers.js";

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
const INFO_TIP_W = 300; // fixed CSS px width of the info tooltip (matches .info-tip)

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
const mix = ([r1, g1, b1], [r2, g2, b2], t) => [
  Math.round(lerp(r1, r2, t)),
  Math.round(lerp(g1, g2, t)),
  Math.round(lerp(b1, b2, t)),
];
const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

export default function NetworkView({ trace, progress, focus, prep, live, onPickOutput, onInspect }) {
  const canvasRef = useRef(null);
  const positions = useMemo(computePositions, []);
  const infoIconsRef = useRef([]); // clickable info-icon hitboxes, filled by draw()
  const [hovered, setHovered] = useState(null);
  const [info, setInfo] = useState(null); // clicked layer explanation { layer, px, py }
  const [overIcon, setOverIcon] = useState(false);
  const [hint, setHint] = useState(null); // one-time "click a node" nudge { x, y, leaving }
  const hintTimers = useRef([]);
  const hintDoneRef = useRef(false); // plays the nudge at most once per session

  const dismissHint = () => {
    if (hintTimers.current.length) {
      hintTimers.current.forEach(clearTimeout);
      hintTimers.current = [];
    }
    setHint((h) => (h ? null : h));
  };

  // Register the canvas layer-info tooltip in the shared popover registry so it
  // closes when any other "i" popover opens (and vice versa).
  const infoCloseRef = useRef(() => setInfo(null));
  useEffect(() => registerPopover(infoCloseRef.current), []);

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
    draw(ctx, positions, trace, progress, hovered, focusSets, prep, infoIconsRef.current, info?.layer ?? null);
  }, [positions, trace, progress, hovered, focusSets, prep, info]);

  // One-time onboarding nudge: once the forward pass has fully settled (progress
  // reaches the last layer, so the probability bars are done), wait 0.5s, then
  // point a cursor at a Hidden 1 node and "click" it with a hint, and fade the
  // whole thing out 4s later. Skipped in Live mode.
  useEffect(() => {
    if (!trace || focus || live || hintDoneRef.current) return;
    if (progress < trace.layers.length) return; // not settled yet
    hintDoneRef.current = true;
    const timers = hintTimers.current;
    timers.push(setTimeout(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      // Point at the brightest Hidden 1 neuron so the cursor lands on a lit node.
      const acts = trace.layers[COL.hidden_1].activations;
      let peak = 0;
      for (let i = 1; i < acts.length; i++) if (acts[i] > acts[peak]) peak = i;
      const p = positions[COL.hidden_1][peak];
      setHint({ x: (p.x / W) * rect.width, y: (p.y / H) * rect.height, leaving: false });
      timers.push(setTimeout(() => setHint((h) => (h ? { ...h, leaving: true } : h)), 4000));
      timers.push(setTimeout(() => setHint(null), 4450));
    }, 500));
  }, [progress, trace, focus, live, positions]);

  // Only retire the nudge when the drawing is cleared — hovering or clicking
  // nodes leaves it up so it stays explicit rather than vanishing by accident.
  useEffect(() => {
    if (!trace) dismissHint();
  }, [trace]);

  const toLogical = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      rect,
      lx: ((e.clientX - rect.left) / rect.width) * W,
      ly: ((e.clientY - rect.top) / rect.height) * H,
    };
  };

  const iconAt = (lx, ly) =>
    infoIconsRef.current.find((ic) => Math.hypot(lx - ic.x, ly - ic.y) <= ic.r + 4) || null;

  const onMove = (e) => {
    const { rect, lx, ly } = toLogical(e);

    // Cursor feedback when over a clickable info icon.
    const on = !!iconAt(lx, ly);
    setOverIcon((prev) => (prev === on ? prev : on));

    if (!trace) {
      if (hovered) setHovered(null);
      return;
    }
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
    setOverIcon(false);
  };

  // Click an info icon to toggle its layer explanation; an output digit to
  // trace it; a hidden neuron to inspect it; empty space clears.
  const onClick = (e) => {
    const { rect, lx, ly } = toLogical(e);

    const ic = iconAt(lx, ly);
    if (ic) {
      if (info && info.layer === ic.layer) {
        setInfo(null); // second click on the same icon toggles it off
      } else {
        // Fixed-width tooltip centered on the icon, clamped fully on-stage.
        const half = INFO_TIP_W / 2, pad = 10;
        const minL = half + pad, maxL = rect.width - half - pad;
        const center = (ic.x / W) * rect.width;
        const px = maxL > minL ? Math.max(minL, Math.min(maxL, center)) : rect.width / 2;
        const py = (ic.y / H) * rect.height + 14;
        closeOtherPopovers(infoCloseRef.current); // opening → close other popovers
        setInfo({ layer: ic.layer, px, py });
      }
      return;
    }

    setInfo(null); // click away closes the explanation
    if (!trace) return;
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
  const infoTip = info
    ? info.layer === "normalize" ? NORMALIZE_INFO : LAYER_INFO[LAYER_KEYS[info.layer]]
    : null;
  const clickable = hovered && (hovered.layer === COL.output || hovered.layer === COL.hidden_1 || hovered.layer === COL.hidden_2);
  const cursor = overIcon || clickable ? "pointer" : "crosshair";

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
      {hint && (
        <div
          className={`node-hint ${hint.leaving ? "leaving" : ""}`}
          style={{ left: `${hint.x}px`, top: `${hint.y}px` }}
          aria-hidden="true"
        >
          <span className="node-hint-ring" />
          <svg className="node-hint-cursor" viewBox="0 0 24 24" width="22" height="22">
            <path
              d="M4 3 L4 21 L9 16 L12 22 L15 20.5 L12 15 L19 15 Z"
              fill="#fff"
              stroke="#0b0f14"
              strokeWidth="1.3"
              strokeLinejoin="round"
            />
          </svg>
          <span className="node-hint-label">Click a node to learn more</span>
        </div>
      )}
    </div>
  );
}

function draw(ctx, positions, trace, progress, hovered, focus, prep, icons, infoLayer) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, W, H);

  // Column headers, each with a clickable info icon whose hitbox is recorded.
  // x's are the centers of each layer's compartment (between the separators at
  // 282/533/765, bounded by the canvas edges) so headers sit centered in-column.
  const heads = [
    ["INPUT · 28×28 = 784 px", 141],
    ["HIDDEN 1 · 512 nodes · ReLU", 408],
    ["HIDDEN 2 · 512 nodes · ReLU", 649],
    ["OUTPUT · 10 · softmax", 862],
  ];
  if (icons) icons.length = 0;
  heads.forEach(([label, x], L) => {
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.font = "600 12px ui-monospace, monospace";
    ctx.fillStyle = "#5c6f7e";
    ctx.fillText(label, x, 26);
    const ix = x + ctx.measureText(label).width / 2 + 11; // superscript, after the text
    const iy = 18;
    drawInfoIcon(ctx, ix, iy, infoLayer === L);
    if (icons) icons.push({ layer: L, x: ix, y: iy, r: 7 });
  });

  // Structural scaffolding, always visible: dividers between the four layers,
  // the full 28x28 input grid, and each layer's activation-function glyph.
  drawSeparators(ctx);
  drawInputGrid(ctx, positions);
  drawActivationGlyphs(ctx, trace, hovered);
  drawNormalizeHeader(ctx, positions, icons); // always visible (has its own info icon)

  // Fresh / cleared state: show the empty scaffolding, no forward pass yet.
  if (!trace) return;

  // Normalization morph plays in the input area before the forward pass; once
  // done, its struck-through step list persists (drawn in the normal path below).
  if (prep && !prep.persisted) {
    drawPreprocessMorph(ctx, positions, prep);
    drawNormalizeSteps(ctx, positions, prep.t);
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
        // A stream of pulses flowing source -> destination; stronger edges carry
        // more of them. The trail spacing shrinks to 0 as the signal lands
        // (tv -> 1) so every pulse arrives at the destination — none left
        // suspended mid-edge when the transition completes.
        const nP = s > 0.5 ? 3 : s > 0.2 ? 2 : 1;
        const rad = (0.8 + s * 1.6) * (em > 1 ? 1.4 : 1);
        for (let j = 0; j < nP; j++) {
          const pf = tv - j * 0.16 * (1 - tv);
          if (pf <= 0) continue;
          const hx = lerp(a.x, b.x, pf);
          const hy = lerp(a.y, b.y, pf);
          const pa = (0.3 + s * 0.6) * (1 - j * 0.28) * (tv < 1 ? 1 : 0.7) * em;
          ctx.fillStyle = rgba(color, clamp01(pa));
          ctx.beginPath();
          ctx.arc(hx, hy, rad, 0, Math.PI * 2);
          ctx.fill();
        }
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

  // Hidden layers as glowing nodes. Neurons charge up (an ease-out "fill" —
  // weighted-sum accumulation) as the layer reveals; those ReLU clipped to zero
  // (pre-activation <= 0) are shown as muted "dead" markers.
  for (const name of ["hidden_1", "hidden_2"]) {
    const L = COL[name];
    const layer = trace.layers[L];
    const lit = layerLit(L);
    if (lit <= 0) continue;
    const peak = layer.peak || 1;
    const preacts = layer.preacts;
    const fill = 1 - Math.pow(1 - lit, 2); // accumulation: rush in, settle
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
        const norm = clamp01(layer.activations[i] / peak) * fill;
        const col = mix([26, 37, 48], EXCITE, norm);
        ctx.fillStyle = rgba(col, (0.4 + norm * 0.6) * d);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.7 + norm * 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  drawOutput(ctx, positions, trace, layerLit(3), progress, focus);

  // Winner "commit" connector: a line flicks from the drawing to the chosen digit.
  if (!focus && trace.layers[3] && progress > 3.4) {
    const flick = Math.sin(clamp01((progress - 3.4) / 0.6) * Math.PI);
    if (flick > 0.02) {
      const inCells = positions[COL.input];
      const c = inCells[0].cell;
      const mid = { x: inCells[0].x - c / 2 + 14 * c, y: inCells[0].y - c / 2 + 14 * c };
      const win = positions[COL.output][trace.prediction];
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = rgba(EXCITE, 0.45 * flick);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(mid.x, mid.y);
      ctx.lineTo(win.x, win.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

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

  // The normalization step list stays (checked off) after the morph.
  if (prep && prep.persisted) drawNormalizeSteps(ctx, positions, 1);
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

}

/** Small clickable info icon: a solid cyan disc with a dark "i" (high contrast). */
function drawInfoIcon(ctx, x, y, active) {
  ctx.save();
  ctx.shadowColor = "rgba(79,227,238,0.85)";
  ctx.shadowBlur = active ? 10 : 6;
  ctx.beginPath();
  ctx.arc(x, y, 6.5, 0, Math.PI * 2);
  ctx.fillStyle = active ? "rgba(150,245,255,1)" : "rgba(79,227,238,1)";
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = "#06131a";
  ctx.font = "800 10px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("i", x, y + 0.5);
  ctx.textBaseline = "alphabetic";
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
      // The softmax squashing curve: raw scores -> (0, 1). For one class vs. the
      // rest, softmax reduces exactly to this logistic S-curve.
      ctx.strokeStyle = "rgba(140,170,190,0.18)";
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x0, base); ctx.lineTo(x0 + GW, base); ctx.stroke();
      ctx.strokeStyle = "rgba(120,200,210,0.75)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      const steps = 24;
      for (let i = 0; i <= steps; i++) {
        const u = i / steps;
        const y = 1 / (1 + Math.exp(-(u - 0.5) * 12)); // sigmoid over logit range ~[-6, 6]
        const px = x0 + u * GW;
        const py = base - y * GH;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    } else {
      // Input has no activation function — it's a raw brightness. Show a 0..1
      // intensity ramp (black -> white) instead of a misleading function curve.
      const grad = ctx.createLinearGradient(x0, 0, x0 + GW, 0);
      grad.addColorStop(0, "#0b0f14");
      grad.addColorStop(1, "#e7edf3");
      ctx.fillStyle = grad;
      ctx.fillRect(x0, GY, GW, GH);
      ctx.strokeStyle = "rgba(140,170,190,0.25)";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x0, GY, GW, GH);
    }
  }

  // (B) When hovering a neuron, mark it on its layer's activation glyph: a point
  // on the ReLU curve for hidden layers; a probability level on the softmax
  // distribution for the output.
  if (trace && hovered) {
    const L = hovered.layer;
    const base = GY + GH;
    const dot = (x, y) => {
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
    };
    const dash = (x1, y1, x2, y2) => {
      ctx.setLineDash([2, 2]);
      ctx.strokeStyle = "rgba(231,237,243,0.4)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    if (L === COL.input) {
      const g = GLYPHS[COL.input];
      const x0 = g.cx - GW / 2;
      const v = clamp01(trace.layers[0].activations[hovered.index]); // intensity 0..1
      const px = x0 + v * GW;
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, GY - 1);
      ctx.lineTo(px, base + 1);
      ctx.stroke();
      dot(px, GY - 3); // caret above the ramp at this pixel's brightness
    } else if ((L === COL.hidden_1 || L === COL.hidden_2) && trace.layers[L].preacts) {
      const g = GLYPHS[L];
      const pt = reluNormalizedPoint(trace.layers[L].preacts[hovered.index]);
      const px = g.cx - GW / 2 + pt.x * GW;
      const py = base - pt.y * GH;
      dash(px, base, px, py); // vertical drop to the point on the curve
      dot(px, py);
    } else if (L === COL.output) {
      const g = GLYPHS[COL.output];
      const x0 = g.cx - GW / 2;
      const logits = trace.layers[COL.output].logits;
      const prob = clamp01(trace.layers[COL.output].activations[hovered.index]);
      // softmax_i = sigmoid(logit_i - logsumexp(other logits)). Using that margin
      // as x places the point exactly ON the S-curve and is exactly accurate.
      let frac = prob;
      if (logits) {
        const i = hovered.index;
        let m = -Infinity;
        for (let j = 0; j < logits.length; j++) if (j !== i) m = Math.max(m, logits[j]);
        let sum = 0;
        for (let j = 0; j < logits.length; j++) if (j !== i) sum += Math.exp(logits[j] - m);
        const margin = logits[i] - (m + Math.log(sum));
        frac = clamp01((margin + 6) / 12);
      }
      const px = x0 + frac * GW;
      const py = base - prob * GH;
      dash(px, base, px, py); // vertical drop to the point on the curve
      dot(px, py);
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
/** The ink crop morphing to its normalized 20px/centered placement. */
function drawPreprocessMorph(ctx, positions, prep) {
  const { geometry: g, inkCanvas, t } = prep;
  const MORPH_FRAC = 0.5; // morph completes in the first half; strikes finish after
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
}

/**
 * The normalization step list below the grid: a centered heading, then the
 * monospaced "N · ..." steps (left-aligned as a centered block). `t` (0..1)
 * drives each step's fade-in and its strikethrough as it completes; pass t=1 to
 * render the finished, fully-struck list that persists after the morph.
 */
// Each step's text only appears once the previous step's checkmark has fully
// faded in (appear[n] = done[n-1] + DONE_DUR), so it reads as check-then-next.
const NORMALIZE_STEPS = [
  { appear: 0.04, done: 0.16, text: "1 · crop to the ink" },
  { appear: 0.28, done: 0.40, text: "2 · scale to 20 px" },
  { appear: 0.52, done: 0.64, text: "3 · center by mass" },
  { appear: 0.76, done: 0.88, text: "4 · sample to 28×28 px" },
];

function _normalizeLayout(positions) {
  const cells = positions[COL.input];
  const cell = cells[0].cell;
  const gx0 = cells[0].x - cell / 2, gy0 = cells[0].y - cell / 2;
  return { gridMid: gx0 + 14 * cell, baseY: gy0 + 28 * cell + 22 };
}

/** The "Normalizing Input" heading + info icon — drawn ALWAYS (even when idle),
 *  so people can read what normalization means before running. */
function drawNormalizeHeader(ctx, positions, icons) {
  const { gridMid, baseY } = _normalizeLayout(positions);
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "rgba(140,170,190,0.9)";
  ctx.font = "600 14px ui-monospace, monospace";
  ctx.fillText("Normalizing Input", gridMid, baseY);
  if (icons) {
    const ix = gridMid + ctx.measureText("Normalizing Input").width / 2 + 12;
    const iy = baseY - 5;
    drawInfoIcon(ctx, ix, iy, false);
    icons.push({ layer: "normalize", x: ix, y: iy, r: 7 });
  }
}

/** The step checklist below the heading — only during/after the normalization morph. */
function drawNormalizeSteps(ctx, positions, t) {
  const { gridMid, baseY } = _normalizeLayout(positions);
  ctx.textAlign = "left";
  ctx.font = "500 11px ui-monospace, monospace";
  const maxW = Math.max(...NORMALIZE_STEPS.map((s) => ctx.measureText(s.text).width));
  const stepsX = gridMid - maxW / 2;
  const DONE_DUR = 0.12;

  NORMALIZE_STEPS.forEach((s, i) => {
    const reveal = clamp01((t - s.appear) / 0.05);
    if (reveal <= 0) return;
    const done = clamp01((t - s.done) / DONE_DUR); // 0 -> 1 as it completes
    const y = baseY + 20 + i * 16;
    ctx.font = "500 11px ui-monospace, monospace";
    ctx.fillStyle = `rgba(120,200,210,${reveal})`;
    ctx.fillText(s.text, stepsX, y);
    if (done > 0) {
      ctx.font = "700 17px ui-monospace, monospace";
      ctx.fillStyle = `rgba(110,231,183,${done})`;
      ctx.fillText("✓", stepsX + maxW + 8, y + 1);
    }
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
