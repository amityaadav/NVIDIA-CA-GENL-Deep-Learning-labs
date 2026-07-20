import { useEffect, useRef } from "react";
import LegendItem from "./LegendItem.jsx";

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

  const { layer, index, topTerms, bias, z, a, activation, sourceLayer,
          restCount, restContribution } = data;
  const fn = activation === "softmax" ? "softmax" : "ReLU";
  const outLabel = activation === "softmax" ? `${(a * 100).toFixed(1)}%` : a.toFixed(3);
  const isInput = sourceLayer === "input";
  const termName = (src) =>
    isInput ? `px ${(src / 28) | 0},${src % 28}` : `#${src}`;
  const signed = (v, p = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(p)}`;
  // Column headers + z formula adapt to what the source layer actually is:
  // pixels feed hidden_1 (value = ink intensity), activations feed the rest.
  const srcHead = isInput ? "pixel (row, col)" : "neuron";
  const valWord = isInput ? "intensity" : "activation";

  return (
    <div className="inspector">
      <div className="inspector-head">
        <span>Neuron <strong>{layer} #{index}</strong></span>
        <button className="explain-clear" onClick={onClose}>Close</button>
      </div>

      {data.weightImage && (
        <div className="inspector-section">
          <LegendItem
            className="explain-label weight-info"
            title="Weight image · the neuron's template"
            ariaLabel="What the weight image shows and where it comes from"
            label="What it looks for · weight image"
          >
            <p>
              This is the neuron's <strong>784 learned weights</strong> — one per input
              pixel — laid back onto the 28×28 grid. <strong>Teal</strong> means a positive
              weight (ink there <em>excites</em> this neuron); <strong>warm</strong> means a
              negative weight (ink there <em>inhibits</em> it); brightness is the weight's
              strength <strong>|w|</strong>. So it's a picture of the ink pattern this
              particular neuron is tuned to look for.
            </p>
            <p>
              Nobody drew this template. The weights started as small <strong>random</strong>
              numbers; during training the network saw thousands of labeled MNIST digits, and
              after each guess <strong>backpropagation</strong> measured how every weight
              affected the error and <strong>gradient descent</strong> nudged it a little to do
              better. Over many passes those nudges self-organized into this arrangement of
              positive and negative weights — whatever helped the network tell digits apart.
            </p>
            <p>
              On your drawing, each pixel's intensity is multiplied by its weight here — exactly
              the terms in the table below. Ink landing on <em>teal</em> squares <strong>adds</strong>
              to the neuron's total <em>z</em> (excitation); ink on <em>warm</em> squares{" "}
              <strong>subtracts</strong> (inhibition); blank pixels add nothing. When your strokes
              line up with the teal pattern and avoid the warm regions, <em>z</em> climbs and ReLU
              lets the neuron <strong>fire</strong>; otherwise it stays quiet (or dead).
            </p>
          </LegendItem>
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
        <table className="term-table">
          <thead>
            <tr>
              <th>{srcHead}</th>
              <th className="num">weight × {valWord}</th>
              <th className="num">contribution</th>
            </tr>
          </thead>
          <tbody>
            {topTerms.map((t, i) => (
              <tr key={i}>
                <td className="mono">{termName(t.src)}</td>
                <td className="mono term-calc">{signed(t.weight)} × {t.value.toFixed(2)}</td>
                <td className={`mono num ${t.contribution >= 0 ? "excite-text" : "inhibit-text"}`}>
                  {signed(t.contribution)}
                </td>
              </tr>
            ))}
            {restCount > 0 && (
              <tr className="term-rest">
                <td className="mono">+ {restCount.toLocaleString()} more</td>
                <td className="mono term-calc">×</td>
                <td className={`mono num ${restContribution >= 0 ? "excite-text" : "inhibit-text"}`}>
                  {restContribution != null ? signed(restContribution) : "×"}
                </td>
              </tr>
            )}
            <tr className="term-bias">
              <td className="mono">bias</td>
              <td className="mono term-calc" />
              <td className={`mono num ${bias >= 0 ? "excite-text" : "inhibit-text"}`}>{signed(bias)}</td>
            </tr>
          </tbody>
        </table>
        <div className="term-formula mono muted">
          z = Σ (weight × {valWord}) + bias
        </div>
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
