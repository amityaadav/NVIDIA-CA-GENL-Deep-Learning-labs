import { useCallback, useEffect, useRef, useState } from "react";
import DrawCanvas from "./components/DrawCanvas.jsx";
import NetworkView from "./components/NetworkView.jsx";
import Controls from "./components/Controls.jsx";
import ContributorPanel from "./components/ContributorPanel.jsx";
import NeuronInspector from "./components/NeuronInspector.jsx";
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
    // Morph runs in the first ~36% (see MORPH_FRAC in drawPreprocess); the rest
    // is a hold so the stacked step captions stay readable for a few seconds.
    const DURATION = 4200;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / DURATION);
      setPrep({ geometry, inkCanvas, t });
      if (t < 1) {
        prepRafRef.current = requestAnimationFrame(tick);
      } else {
        prepRafRef.current = 0;
        setPrep(null);
        then?.();
      }
    };
    setPrep({ geometry, inkCanvas, t: 0 });
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
  }, [snapToEnd]);

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

      <footer className="app-footer">
        <span className="legend"><i className="dot excite" /> excitatory path</span>
        <span className="legend"><i className="dot inhibit" /> inhibitory path</span>
        <span className="legend"><i className="dot dead" /> dead neuron (ReLU → 0)</span>
        <span className="muted">Node brightness = activation. Hover a neuron to see its z → a on the curve.</span>
      </footer>
    </div>
  );
}
