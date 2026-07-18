import { useEffect, useRef } from "react";

const EXCITE = [79, 227, 238];
const INHIBIT = [224, 119, 110];

/**
 * Inspects a single neuron: its weight image ("what is it looking for?", hidden_1
 * only) and the actual weighted-sum math on the current drawing. Renders nothing
 * until a neuron is selected.
 */
export default function NeuronInspector({ data, loading, onClose }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data?.weightImage) return;
    const N = 28, cell = 4;
    canvas.width = N * cell;
    canvas.height = N * cell;
    const ctx = canvas.getContext("2d");
    const w = data.weightImage;
    let maxAbs = 0;
    for (const v of w) maxAbs = Math.max(maxAbs, Math.abs(v));
    maxAbs = maxAbs || 1;

    // "Develop" the template with a top-down wipe + a scan line, so it feels
    // like the neuron's receptive field is being revealed.
    let raf = 0;
    const start = performance.now();
    const DURATION = 650;
    const render = (now) => {
      const t = Math.min(1, (now - start) / DURATION);
      const revealRows = t * N;
      ctx.fillStyle = "#0b0f14";
      ctx.fillRect(0, 0, N * cell, N * cell);
      for (let i = 0; i < w.length; i++) {
        const r = (i / N) | 0;
        if (r > revealRows) continue;
        const mag = Math.abs(w[i]) / maxAbs;
        if (mag < 0.03) continue;
        const [rr, gg, bb] = w[i] >= 0 ? EXCITE : INHIBIT;
        ctx.fillStyle = `rgba(${rr},${gg},${bb},${mag})`;
        ctx.fillRect((i % N) * cell, r * cell, cell, cell);
      }
      if (t < 1) {
        ctx.fillStyle = "rgba(150,240,250,0.6)";
        ctx.fillRect(0, revealRows * cell, N * cell, 1.5);
        raf = requestAnimationFrame(render);
      }
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [data]);

  if (loading) return <div className="inspector">Inspecting neuron…</div>;
  if (!data) return null;

  const { layer, index, topTerms, bias, z, a, activation, sourceLayer } = data;
  const fn = activation === "softmax" ? "softmax" : "ReLU";
  const outLabel = activation === "softmax" ? `${(a * 100).toFixed(1)}%` : a.toFixed(3);
  const termName = (src) =>
    sourceLayer === "input" ? `px ${(src / 28) | 0},${src % 28}` : `#${src}`;
  const signed = (v, p = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(p)}`;

  return (
    <div className="inspector">
      <div className="inspector-head">
        <span>Neuron <strong>{layer} #{index}</strong></span>
        <button className="explain-clear" onClick={onClose}>Close</button>
      </div>

      {data.weightImage && (
        <div className="inspector-section">
          <span className="explain-label">What it looks for · weight image</span>
          <div className="weight-image">
            <canvas ref={canvasRef} />
            <span className="muted tiny">teal excites · warm inhibits · brightness = |weight|</span>
          </div>
        </div>
      )}

      <div className="inspector-section">
        <span className="explain-label">
          The math on this drawing · top {topTerms.length} of {sourceLayer}
        </span>
        <ul className="term-list">
          {topTerms.map((t, i) => (
            <li key={i}>
              <span className="mono term-src">{termName(t.src)}</span>
              <span className="mono term-calc">{signed(t.weight)} × {t.value.toFixed(2)}</span>
              <span className={`mono ${t.contribution >= 0 ? "excite-text" : "inhibit-text"}`}>
                {signed(t.contribution)}
              </span>
            </li>
          ))}
          <li className="term-bias">
            <span className="mono term-src">bias</span>
            <span className="mono term-calc" />
            <span className={`mono ${bias >= 0 ? "excite-text" : "inhibit-text"}`}>{signed(bias)}</span>
          </li>
        </ul>
        <div className="term-result">
          z = {z.toFixed(3)} <span className="muted">→ {fn} →</span> a = {outLabel}
        </div>
        <span className="muted tiny">
          Only the strongest terms are shown; z is the full weighted sum of all {sourceLayer} inputs + bias.
        </span>
      </div>
    </div>
  );
}
