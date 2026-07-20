import { useCallback, useEffect, useRef, useState } from "react";
import DrawCanvas from "./components/DrawCanvas.jsx";
import NetworkView from "./components/NetworkView.jsx";
import Controls from "./components/Controls.jsx";
import ContributorPanel from "./components/ContributorPanel.jsx";
import NeuronInspector from "./components/NeuronInspector.jsx";
import LegendItem from "./components/LegendItem.jsx";
import TrainingPanel from "./components/TrainingPanel.jsx";
import { useAnimation } from "./hooks/useAnimation.js";
import { canvasToInput, normalize } from "./lib/preprocess.js";
import { runInference, runExplain, runNeuron, getModel } from "./lib/api.js";

const PHASES = 4; // input + 2 hidden + output

// Link target for "NVIDIA's Deep Learning Lab 1 on Neural Networks" in the
// header. Set to the Lab 1 GitLab file URL to render it as a link; null renders
// the same text plainly (no broken link) until a URL is provided.
const LAB1_URL = "https://github.com/amityaadav/NVIDIA-Deep-Learning-labs/blob/main/lab1_nn_mnist.py";

// The "Watch it learn" tab is hidden for the public demo (inference only).
// Flip to true to restore the training tab and its controls.
const TRAINING_ENABLED = false;

// Connections slider: how many top edges to draw per transition. The log-scaled
// slider (0→1000) maps to N edges/transition, from the default 40 (120 total,
// exact-parity view) up to the full fully-connected net. Per-transition caps are
// nSrc×nDst; when N hits the largest (784×512) every transition is fully drawn,
// so the total reaches 784·512 + 512·512 + 512·10 = 668,672.
const EDGE_MIN = 40;
const EDGE_MAX = 784 * 512; // largest transition; N here => whole net shown
const sliderToN = (s) => Math.round(EDGE_MIN * Math.pow(EDGE_MAX / EDGE_MIN, s / 1000));

export default function App() {
  const drawRef = useRef(null);
  const vizRef = useRef(null); // the "watch the forward pass" section (for mobile scroll)
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
  const [mode, setMode] = useState("infer"); // "infer" | "train"
  const [edgeSlider, setEdgeSlider] = useState(0); // Connections slider position 0→1000
  const edgeNRef = useRef(EDGE_MIN); // current edges/transition (kept in a ref so runs read it)
  const modelRef = useRef(null); // cached model, for synchronous slider recompute

  // Grab the (already-preloading) model so the slider can recompute edges
  // in-place without re-running the animation.
  useEffect(() => {
    getModel().then((m) => { modelRef.current = m; }).catch(() => {});
  }, []);

  // 1800ms/phase = half the previous default sweep speed (the speed knob still
  // scales from here). Only affects the forward-pass particles, not the input
  // normalization morph (which has its own timer).
  const anim = useAnimation(PHASES, { msPerPhase: 1800 });
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
    // On a stacked (mobile) layout, scroll to the animation so it isn't missed.
    // matchMedia matches the exact CSS breakpoint; rAF lets layout settle first.
    if (window.matchMedia("(max-width: 900px)").matches) {
      requestAnimationFrame(() =>
        vizRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
      );
    }
    try {
      const result = await runInference(norm.input, edgeNRef.current);
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
      const result = await runInference(pixels, edgeNRef.current);
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

  // Move the Connections slider: recompute the trace's edges in place (same
  // forward pass, more/fewer top connections) so the web re-densifies without
  // replaying the animation. Only recomputes once a digit has been run.
  const onEdgeSlider = useCallback((s) => {
    setEdgeSlider(s);
    const n = sliderToN(s);
    edgeNRef.current = n;
    if (modelRef.current && pixelsRef.current) {
      setTrace(modelRef.current.predict(pixelsRef.current, n));
    }
  }, []);

  // Convenience: Enter runs inference, Escape clears (only in the infer tab).
  useEffect(() => {
    if (mode !== "infer") return;
    const onKey = (e) => {
      if (e.key === "Enter") handleRun();
      if (e.key === "Escape") handleClear();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleRun, handleClear, mode]);

  // Total connections in the fully-connected net vs. how many we actually draw
  // (only the strongest ~40 per layer). The architecture is fixed, so these are
  // known even before a run — the note is always shown.
  const layerSizes = trace ? trace.layers.map((l) => l.size) : [784, 512, 512, 10];
  const transCaps = [
    layerSizes[0] * layerSizes[1],
    layerSizes[1] * layerSizes[2],
    layerSizes[2] * layerSizes[3],
  ];
  const totalConnections = transCaps.reduce((n, k) => n + k, 0);
  // After a run, count the edges we actually drew; before one, show what the
  // slider's current setting would draw (each transition capped at its size).
  const perTransition = trace
    ? trace.transitions.map((t) => t.links.length)
    : transCaps.map((cap) => Math.min(edgeNRef.current, cap));
  const shownConnections = perTransition.reduce((n, k) => n + k, 0);
  const shownPct = totalConnections ? (shownConnections / totalConnections) * 100 : 0;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Watch a neural network think</h1>
        <p>
          An artificial neural network is a stack of simple units — "neurons" — that
          learn from examples to turn an input into an answer. This one recognizes
          handwritten digits from <strong>MNIST</strong>, a classic dataset of{" "}
          <strong>60,000</strong> training and <strong>10,000</strong> validation images, each a
          28×28 grayscale digit (0–9). The model was already trained during NVIDIA's Deep
          Learning{" "}
          {LAB1_URL ? (
            <a href={LAB1_URL} target="_blank" rel="noopener noreferrer">Lab&nbsp;1</a>
          ) : (
            "Lab 1"
          )}{" "}
          on Neural Networks — you're watching pure <strong>inference</strong>.
        </p>
      </header>

      {TRAINING_ENABLED && (
        <div className="mode-tabs" role="tablist">
          <button
            role="tab"
            aria-selected={mode === "infer"}
            className={`mode-tab ${mode === "infer" ? "on" : ""}`}
            onClick={() => setMode("infer")}
          >
            Watch it think
          </button>
          <button
            role="tab"
            aria-selected={mode === "train"}
            className={`mode-tab ${mode === "train" ? "on" : ""}`}
            onClick={() => setMode("train")}
          >
            Watch it learn
          </button>
        </div>
      )}

      {TRAINING_ENABLED && mode === "train" && <TrainingPanel />}

      {(!TRAINING_ENABLED || mode === "infer") && (
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

        <section className="panel viz-panel" ref={vizRef}>
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
          </div>
          <NetworkView
            trace={trace}
            progress={anim.progress}
            focus={focus}
            prep={prep}
            live={live}
            onPickOutput={handlePickOutput}
            onInspect={handleInspect}
          />
          <Controls anim={anim} hasTrace={!!trace}>
            <label className="conn-ctrl" title="How many of the strongest connections to draw">
              Connections
              <input
                type="range"
                min="0"
                max="1000"
                step="1"
                value={edgeSlider}
                onChange={(e) => onEdgeSlider(Number(e.target.value))}
                aria-label="Number of connections to draw"
              />
              <span className="conn-val">
                {shownConnections.toLocaleString()} / {totalConnections.toLocaleString()}
              </span>
            </label>
          </Controls>
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
      )}
    </div>
  );
}
