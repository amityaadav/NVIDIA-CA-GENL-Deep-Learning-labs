import { useCallback, useEffect, useRef, useState } from "react";
import DrawCanvas from "./components/DrawCanvas.jsx";
import NetworkView from "./components/NetworkView.jsx";
import Controls from "./components/Controls.jsx";
import { useAnimation } from "./hooks/useAnimation.js";
import { canvasToInput } from "./lib/preprocess.js";
import { runInference } from "./lib/api.js";

const PHASES = 4; // input + 2 hidden + output

export default function App() {
  const drawRef = useRef(null);
  const [trace, setTrace] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState("");

  const anim = useAnimation(PHASES);
  const { play, restart } = anim;

  const handleRun = useCallback(async () => {
    const canvas = drawRef.current?.canvas();
    if (!canvas) return;
    const pixels = canvasToInput(canvas);
    if (!pixels) {
      setStatus("error");
      setError("Draw a digit first — the canvas is blank.");
      return;
    }
    setStatus("loading");
    setError("");
    try {
      const result = await runInference(pixels);
      setTrace(result);
      setStatus("ready");
      restart();
      requestAnimationFrame(() => play()); // auto-play once the trace is in
    } catch (e) {
      setStatus("error");
      setError(e.message || "Something went wrong reaching the model.");
    }
  }, [play, restart]);

  const handleClear = useCallback(() => {
    drawRef.current?.clear();
    setTrace(null);
    setStatus("idle");
    setError("");
    restart();
  }, [restart]);

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
          <DrawCanvas ref={drawRef} />
          <div className="draw-actions">
            <button className="primary" onClick={handleRun} disabled={status === "loading"}>
              {status === "loading" ? "Running…" : "Run inference"}
            </button>
            <button onClick={handleClear}>Clear</button>
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
          <NetworkView trace={trace} progress={anim.progress} />
          <Controls anim={anim} hasTrace={!!trace} />
        </section>
      </main>

      <footer className="app-footer">
        <span className="legend"><i className="dot excite" /> excitatory path</span>
        <span className="legend"><i className="dot inhibit" /> inhibitory path</span>
        <span className="muted">Node brightness = activation, normalized per layer.</span>
      </footer>
    </div>
  );
}
