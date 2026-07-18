import { useCallback, useEffect, useRef, useState } from "react";
import DrawCanvas from "./components/DrawCanvas.jsx";
import NetworkView from "./components/NetworkView.jsx";
import Controls from "./components/Controls.jsx";
import ContributorPanel from "./components/ContributorPanel.jsx";
import NeuronInspector from "./components/NeuronInspector.jsx";
import LegendItem from "./components/LegendItem.jsx";
import { useAnimation } from "./hooks/useAnimation.js";
import { canvasToInput, normalize } from "./lib/preprocess.js";
import { runInference, runExplain, runNeuron } from "./lib/api.js";

const PHASES = 4; // input + 2 hidden + output

export default function App() {
  const drawRef = useRef(null);
  const pixelsRef = useRef(null); // last pixels sent, reused for "why this digit?"
  const [trace, setTrace] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState("");
  const [focus, setFocus] = useState(null); // the "why this digit?" explanation
  const [focusLoading, setFocusLoading] = useState(false);
  const [inspect, setInspect] = useState(null); // { data, loading } for a picked neuron
  const [live, setLive] = useState(false); // predict continuously while drawing
  const [prep, setPrep] = useState(null); // { geometry, inkCanvas, t } normalization morph
  const prepRafRef = useRef(0);

  const anim = useAnimation(PHASES);
  const { play, restart, snapToEnd } = anim;

  const cancelPreprocess = useCallback(() => {
    if (prepRafRef.current) cancelAnimationFrame(prepRafRef.current);
    prepRafRef.current = 0;
  }, []);

  // Animate the crop -> scale -> center normalization, then run `then`.
  const playPreprocess = useCallback((geometry, inkCanvas, then) => {
    cancelPreprocess();
    // Morph runs in the first half; the rest finishes striking through the step
    // list. Paced slowly so it's readable. When done, the struck captions persist
    // (prep.persisted) below the grid while the forward pass plays — they don't vanish.
    const DURATION = 4800;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / DURATION);
      if (t < 1) {
        setPrep({ geometry, inkCanvas, t, persisted: false });
        prepRafRef.current = requestAnimationFrame(tick);
      } else {
        prepRafRef.current = 0;
        setPrep({ geometry, inkCanvas, t: 1, persisted: true });
        then?.();
      }
    };
    setPrep({ geometry, inkCanvas, t: 0, persisted: false });
    prepRafRef.current = requestAnimationFrame(tick);
  }, [cancelPreprocess]);

  const handleRun = useCallback(async () => {
    const canvas = drawRef.current?.canvas();
    if (!canvas) return;
    const norm = normalize(canvas);
    if (!norm) {
      setStatus("error");
      setError("Draw a digit first — the canvas is blank.");
      return;
    }
    pixelsRef.current = norm.input;
    setStatus("loading");
    setError("");
    setFocus(null);
    setInspect(null);
    cancelPreprocess();
    try {
      const result = await runInference(norm.input);
      setTrace(result);
      setStatus("ready");
      restart();
      // Show the normalization morph, then play the forward pass.
      playPreprocess(norm.geometry, norm.inkCanvas, () => requestAnimationFrame(() => play()));
    } catch (e) {
      setStatus("error");
      setError(e.message || "Something went wrong reaching the model.");
    }
  }, [play, restart, playPreprocess, cancelPreprocess]);

  // Live mode: infer + snap straight to the final state (no sweep) as you draw.
  const runSnap = useCallback(async () => {
    const canvas = drawRef.current?.canvas();
    if (!canvas) return;
    const pixels = canvasToInput(canvas);
    if (!pixels) return;
    pixelsRef.current = pixels;
    cancelPreprocess();
    setPrep(null); // live mode skips the normalization captions
    setFocus(null);
    setInspect(null);
    try {
      const result = await runInference(pixels);
      setTrace(result);
      setStatus("ready");
      snapToEnd();
    } catch {
      /* stay quiet during live drawing */
    }
  }, [snapToEnd, cancelPreprocess]);

  const handleStrokeEnd = useCallback(() => {
    if (live) runSnap();
  }, [live, runSnap]);

  const toggleLive = useCallback(() => {
    setLive((v) => {
      if (!v) requestAnimationFrame(() => runSnap()); // reflect current drawing at once
      return !v;
    });
  }, [runSnap]);

  // Click an output digit -> trace why the network chose it (null clears focus).
  const handlePickOutput = useCallback(async (target) => {
    if (target == null || !pixelsRef.current) {
      setFocus(null);
      return;
    }
    setInspect(null);
    setFocusLoading(true);
    try {
      const result = await runExplain(pixelsRef.current, target);
      setFocus(result);
    } catch (e) {
      setFocus(null);
      setStatus("error");
      setError(e.message || "Couldn't trace that digit.");
    } finally {
      setFocusLoading(false);
    }
  }, []);

  // Click a hidden neuron -> inspect its weights + math (null clears).
  const handleInspect = useCallback(async (sel) => {
    if (!sel || !pixelsRef.current) {
      setInspect(null);
      return;
    }
    setFocus(null);
    setInspect({ data: null, loading: true });
    try {
      const data = await runNeuron(pixelsRef.current, sel.layer, sel.index);
      setInspect({ data, loading: false });
    } catch (e) {
      setInspect(null);
      setStatus("error");
      setError(e.message || "Couldn't inspect that neuron.");
    }
  }, []);

  const handleClear = useCallback(() => {
    drawRef.current?.clear();
    pixelsRef.current = null;
    cancelPreprocess();
    setPrep(null);
    setTrace(null);
    setStatus("idle");
    setError("");
    setFocus(null);
    setInspect(null);
    restart();
  }, [restart, cancelPreprocess]);

  // Convenience: Enter runs inference, Escape clears.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Enter") handleRun();
      if (e.key === "Escape") handleClear();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleRun, handleClear]);

  // Total connections in the fully-connected net vs. how many we actually draw
  // (only the strongest few per layer — see TOP_EDGES_PER_TRANSITION).
  const totalConnections = trace
    ? trace.layers[0].size * trace.layers[1].size +
      trace.layers[1].size * trace.layers[2].size +
      trace.layers[2].size * trace.layers[3].size
    : 0;
  const shownConnections = trace
    ? trace.transitions.reduce((n, t) => n + t.links.length, 0)
    : 0;
  const layerSizes = trace ? trace.layers.map((l) => l.size) : [];
  const perTransition = trace ? trace.transitions.map((t) => t.links.length) : [];
  const shownPct = totalConnections ? (shownConnections / totalConnections) * 100 : 0;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Watch a neural network think</h1>
        <p>
          Draw a digit; it's normalized the way MNIST expects and pushed through a
          784&nbsp;→&nbsp;512&nbsp;→&nbsp;512&nbsp;→&nbsp;10 network. The animation replays the
          forward pass — neurons light by activation, and the strongest weighted paths trace how
          the answer forms.
        </p>
      </header>

      <main className="layout">
        <section className="panel draw-panel">
          <h2>1 · Draw</h2>
          <DrawCanvas ref={drawRef} onStrokeEnd={handleStrokeEnd} />
          <div className="draw-actions">
            <button className="primary" onClick={handleRun} disabled={status === "loading"}>
              {status === "loading" ? "Running…" : "Run inference"}
            </button>
            <button onClick={handleClear}>Clear</button>
          </div>
          <div className="draw-options">
            <button
              className={`toggle ${live ? "on" : ""}`}
              onClick={toggleLive}
              aria-pressed={live}
              title="Predict continuously as you draw"
            >
              <span className="toggle-dot" /> Live prediction
            </button>
          </div>
          <p className="hint">Enter to run · Esc to clear</p>
          {status === "ready" && trace && (
            <p className="verdict">
              Prediction: <strong>{trace.prediction}</strong>
              <span> · {Math.round(trace.probs[trace.prediction] * 100)}% confident</span>
            </p>
          )}
          {status === "error" && <p className="error">{error}</p>}
        </section>

        <section className="panel viz-panel">
          <h2>2 · Watch the forward pass</h2>
          <div className="legend-bar">
            <LegendItem dotClass="excite" label="excitatory path" title="Excitatory connection (teal)">
              A connection whose signal <strong>pushes the next neuron up</strong>. On this
              drawing its contribution — the source neuron's activation multiplied by the
              connection's weight — is <strong>positive</strong>, so it's evidence <em>for</em>
              the neuron it feeds (and ultimately for a digit). Brighter and thicker lines mean
              a larger contribution. Because a neuron's inputs are never negative (pixels and
              post-ReLU activations are ≥ 0), the sign is simply the sign of the weight.
            </LegendItem>
            <LegendItem dotClass="inhibit" label="inhibitory path" title="Inhibitory connection (orange)">
              A connection whose signal <strong>pushes the next neuron down</strong>. Its
              contribution (activation × weight) is <strong>negative</strong> here — evidence
              <em> against</em> the neuron it feeds. Brighter/thicker means more strongly negative.
              In the output layer these are the connections voting to <em>rule a digit out</em>;
              in the first layer most of the strongest connections are inhibitory, because ink
              landing on a spot contradicts the many neurons that expect it blank.
            </LegendItem>
            <LegendItem dotClass="dead" label="dead neuron (ReLU → 0)" title="Dead neuron">
              A hidden neuron whose weighted sum (its pre-activation <em>z</em>) came out ≤ 0.
              ReLU — <strong>max(0, z)</strong> — clamps that to exactly <strong>0</strong>, so
              the neuron outputs nothing and passes no signal onward for this input. Shown as a
              small dim dot. This is normal and common: on a typical digit <em>most</em> hidden
              neurons are dead (the network is "sparse"), and precisely which ones fire is what
              encodes the pattern.
            </LegendItem>
            <span className="muted">Node brightness = activation · click any ⓘ to learn more</span>
            {trace && (
              <LegendItem
                className="conn-count"
                title="Why only the strongest connections?"
                ariaLabel="Why only the strongest connections are shown"
                label={
                  <>
                    Showing {shownConnections.toLocaleString()} of{" "}
                    {totalConnections.toLocaleString()} connections (strongest only)
                  </>
                }
              >
                <p>
                  This network is <strong>fully connected</strong> — every neuron links to every
                  neuron in the next layer. That is a lot of wires:
                </p>
                <p className="legend-math">
                  input → hidden 1: {layerSizes[0]} × {layerSizes[1]} ={" "}
                  {(layerSizes[0] * layerSizes[1]).toLocaleString()}<br />
                  hidden 1 → hidden 2: {layerSizes[1]} × {layerSizes[2]} ={" "}
                  {(layerSizes[1] * layerSizes[2]).toLocaleString()}<br />
                  hidden 2 → output: {layerSizes[2]} × {layerSizes[3]} ={" "}
                  {(layerSizes[2] * layerSizes[3]).toLocaleString()}<br />
                  <strong>total = {totalConnections.toLocaleString()}</strong>
                </p>
                <p>
                  Drawing all {totalConnections.toLocaleString()} lines is an unreadable hairball
                  (and slow to render every frame). So we keep only the{" "}
                  <strong>{perTransition[0]} strongest</strong> connections in each of the 3
                  transitions — {perTransition.join(" + ")} ={" "}
                  <strong>{shownConnections}</strong>, about {shownPct.toFixed(3)}% of them.
                </p>
                <p>
                  "Strongest" is judged per drawing: a connection's contribution is{" "}
                  <strong>source activation × weight</strong>, and we rank by absolute value
                  (keeping both excitatory and inhibitory). So the visible web reflects where the
                  signal actually flows for <em>your</em> digit — it changes every time you draw.
                </p>
              </LegendItem>
            )}
          </div>
          <NetworkView
            trace={trace}
            progress={anim.progress}
            focus={focus}
            prep={prep}
            onPickOutput={handlePickOutput}
            onInspect={handleInspect}
          />
          <Controls anim={anim} hasTrace={!!trace} />
          {trace && inspect && (
            <NeuronInspector
              data={inspect.data}
              loading={inspect.loading}
              onClose={() => setInspect(null)}
            />
          )}
          {trace && !inspect && (
            <ContributorPanel
              trace={trace}
              focus={focus}
              loading={focusLoading}
              onClear={() => handlePickOutput(null)}
            />
          )}
        </section>
      </main>
    </div>
  );
}
