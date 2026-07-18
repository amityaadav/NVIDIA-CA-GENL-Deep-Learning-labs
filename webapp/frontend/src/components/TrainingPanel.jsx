import { useEffect, useRef, useState } from "react";
import { openTrainStream } from "../lib/api.js";
import TrainNetworkView from "./TrainNetworkView.jsx";

const BATCH_SIZES = [8, 16, 32, 64, 128];

/**
 * "Watch it learn": configures a training run and streams live metrics from the
 * backend, drawing the loss curve in real time. Trains a fresh model on a small
 * MNIST subset — the committed model used for inference is never touched.
 */
export default function TrainingPanel() {
  const [lr, setLr] = useState(0.1);
  const [batchSize, setBatchSize] = useState(32);
  const [epochs, setEpochs] = useState(3);
  const [metrics, setMetrics] = useState([]); // per-step { step, loss, trainAcc, validAcc }
  const [snapshot, setSnapshot] = useState(null); // { step, ids, images } latest weight templates
  const [net, setNet] = useState(null); // { trace, learning } latest network snapshot
  const [netMode, setNetMode] = useState("activations"); // "activations" | "learning"
  const [status, setStatus] = useState("idle"); // idle | training | done | diverged | error
  const esRef = useRef(null);

  const stop = () => {
    esRef.current?.close();
    esRef.current = null;
  };

  useEffect(() => stop, []); // close the stream on unmount

  const start = () => {
    stop();
    setMetrics([]);
    setSnapshot(null);
    setNet(null);
    setStatus("training");
    const es = openTrainStream({ lr, batchSize, epochs });
    esRef.current = es;
    es.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.status === "starting") return;
      if (m.diverged) { setStatus("diverged"); stop(); return; }
      if (m.done) { setStatus("done"); stop(); return; }
      if (m.templates) setSnapshot({ step: m.step, ids: m.templateIds, images: m.templates });
      if (m.sampleTrace) setNet({ trace: m.sampleTrace, learning: m.learning, label: m.sampleLabel });
      setMetrics((prev) => [...prev, m]);
    };
    es.onerror = () => { setStatus((s) => (s === "training" ? "error" : s)); stop(); };
  };

  const reset = () => { stop(); setMetrics([]); setSnapshot(null); setNet(null); setStatus("idle"); };

  const busy = status === "training";
  const last = metrics[metrics.length - 1];
  const lastValid = [...metrics].reverse().find((m) => m.validAcc != null);
  const progress = last ? last.step / last.totalSteps : 0;

  return (
    <section className="panel train-panel">
      <h2>Train a network from scratch</h2>
      <p className="train-intro">
        This starts a brand-new network with <strong>random</strong> weights and trains it on
        thousands of MNIST digits with gradient descent. Watch the <strong>loss</strong> — how
        wrong it is — fall as it learns. The model used in the other tab is left untouched.
      </p>

      <div className="train-controls">
        <label className="train-field">
          <span>Learning rate <b>{lr}</b></span>
          <input type="range" min="0.001" max="3" step="0.001" value={lr}
            disabled={busy} onChange={(e) => setLr(Number(e.target.value))} />
          <span className="train-hint">too high → diverges · too low → crawls</span>
        </label>
        <label className="train-field">
          <span>Batch size</span>
          <select value={batchSize} disabled={busy} onChange={(e) => setBatchSize(Number(e.target.value))}>
            {BATCH_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="train-field">
          <span>Epochs <b>{epochs}</b></span>
          <input type="range" min="1" max="10" step="1" value={epochs}
            disabled={busy} onChange={(e) => setEpochs(Number(e.target.value))} />
        </label>
        <div className="train-actions">
          {busy
            ? <button className="primary" onClick={stop}>■ Stop</button>
            : <button className="primary" onClick={start}>▶ Train</button>}
          <button onClick={reset} disabled={busy && metrics.length === 0}>Reset</button>
        </div>
      </div>

      <LossChart metrics={metrics} />

      <div className="train-stats">
        <Stat label="Step" value={last ? `${last.step} / ${last.totalSteps}` : "—"} />
        <Stat label="Loss" value={last ? last.loss.toFixed(4) : "—"} />
        <Stat label="Train acc" value={last ? `${(last.trainAcc * 100).toFixed(1)}%` : "—"} />
        <Stat label="Valid acc" value={lastValid ? `${(lastValid.validAcc * 100).toFixed(1)}%` : "—"} />
      </div>

      <p className={`train-status train-status-${status}`}>
        {status === "idle" && "Set the parameters and press Train."}
        {status === "training" && `Training… ${Math.round(progress * 100)}%`}
        {status === "done" && "Done — the loss curve above is this network learning."}
        {status === "diverged" && "Diverged — the learning rate was too high and the loss blew up. Lower it and retry."}
        {status === "error" && "Lost the training stream. Is the backend running?"}
      </p>

      {net && <SamplePrediction trace={net.trace} label={net.label} />}

      {net && (
        <div className="train-net">
          <div className="train-net-head">
            <span className="explain-label">The network, live</span>
            <div className="net-toggle">
              <button className={netMode === "activations" ? "on" : ""} onClick={() => setNetMode("activations")}>
                Activations
              </button>
              <button className={netMode === "learning" ? "on" : ""} onClick={() => setNetMode("learning")}>
                Learning
              </button>
            </div>
          </div>
          <TrainNetworkView trace={net.trace} learning={net.learning} mode={netMode} />
          <span className="muted tiny">
            {netMode === "activations"
              ? "A fixed digit pushed through the current model — watch the activations organize and the prediction sharpen as it trains."
              : "How hard each neuron and connection is being pushed by the gradients right now (amber = strongly updating)."}
          </span>
        </div>
      )}

      {snapshot && <WeightTemplates snapshot={snapshot} />}
    </section>
  );
}

const EXCITE = [79, 227, 238];
const INHIBIT = [224, 119, 110];

/**
 * A grid of Hidden-1 neuron weight images that updates as snapshots stream —
 * you watch each neuron's "what it looks for" template form from random noise.
 */
function WeightTemplates({ snapshot }) {
  return (
    <div className="templates">
      <span className="explain-label">
        What hidden-1 neurons are learning to look for · weights at step {snapshot.step}
      </span>
      <div className="templates-grid">
        {snapshot.images.map((img, i) => (
          <Template key={snapshot.ids[i]} weights={img} id={snapshot.ids[i]} />
        ))}
      </div>
      <span className="muted tiny">
        teal = excites · warm = inhibits · brightness = |weight|. They start as noise and organize.
      </span>
    </div>
  );
}

function Template({ weights, id }) {
  const ref = useRef(null);
  useEffect(() => {
    const N = 28, cell = 2;
    const canvas = ref.current;
    canvas.width = N * cell;
    canvas.height = N * cell;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, N * cell, N * cell);
    let maxAbs = 0;
    for (const v of weights) maxAbs = Math.max(maxAbs, Math.abs(v));
    maxAbs = maxAbs || 1;
    for (let i = 0; i < weights.length; i++) {
      const mag = Math.abs(weights[i]) / maxAbs;
      if (mag < 0.04) continue;
      const [r, g, b] = weights[i] >= 0 ? EXCITE : INHIBIT;
      ctx.fillStyle = `rgba(${r},${g},${b},${mag})`;
      ctx.fillRect((i % N) * cell, ((i / N) | 0) * cell, cell, cell);
    }
  }, [weights]);
  return (
    <div className="template">
      <canvas ref={ref} />
      <span className="template-id">#{id}</span>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="train-stat">
      <span className="train-stat-label">{label}</span>
      <span className="train-stat-value">{value}</span>
    </div>
  );
}

/**
 * The clearest "is it learning?" story: the fixed digit, the network's current
 * guess + confidence (right/wrong), and the full 10-class distribution sharpening.
 */
function SamplePrediction({ trace, label }) {
  const digitRef = useRef(null);
  useEffect(() => {
    const N = 28, cell = 3;
    const canvas = digitRef.current;
    canvas.width = N * cell;
    canvas.height = N * cell;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, N * cell, N * cell);
    const px = trace.layers[0].activations;
    for (let i = 0; i < px.length; i++) {
      const v = px[i];
      if (v < 0.03) continue;
      const g = Math.round(v * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect((i % N) * cell, ((i / N) | 0) * cell, cell, cell);
    }
  }, [trace]);

  const { probs, prediction: pred } = trace;
  const correct = pred === label;
  return (
    <div className="sample-pred">
      <div className="sample-left">
        <canvas ref={digitRef} className="sample-digit" />
        <span className="muted tiny">true label: <strong>{label}</strong></span>
      </div>
      <div className="sample-right">
        <div className={`sample-verdict ${correct ? "ok" : "no"}`}>
          predicts <strong>{pred}</strong> · {Math.round(probs[pred] * 100)}% {correct ? "✓ correct" : "✗ wrong"}
        </div>
        <div className="sample-bars">
          {probs.map((p, d) => (
            <div key={d} className={`sample-row ${d === label ? "is-true" : ""}`}>
              <span className="sample-d">{d}</span>
              <div className="sample-track">
                <div
                  className="sample-fill"
                  style={{
                    width: `${Math.max(1, p * 100)}%`,
                    background: d === pred ? (correct ? "var(--accent)" : "var(--warm)") : "var(--accent-dim)",
                  }}
                />
              </div>
              <span className="sample-pct">{Math.round(p * 100)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Live loss curve: loss (teal) over training steps, auto-scaled. */
function LossChart({ metrics }) {
  const ref = useRef(null);
  const W = 720, H = 260, PAD = 34;

  useEffect(() => {
    const canvas = ref.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, W, H);

    const maxLoss = Math.max(0.5, ...metrics.map((m) => m.loss));
    const totalSteps = metrics.length ? metrics[metrics.length - 1].totalSteps : 1;

    // Axes.
    ctx.strokeStyle = "rgba(140,170,190,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD, PAD - 6); ctx.lineTo(PAD, H - PAD); ctx.lineTo(W - PAD + 6, H - PAD);
    ctx.stroke();
    ctx.fillStyle = "#5c6f7e";
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText(maxLoss.toFixed(1), PAD - 6, PAD);
    ctx.fillText("0", PAD - 6, H - PAD + 3);
    ctx.textAlign = "center";
    ctx.fillText("loss", PAD + (W - 2 * PAD) / 2, H - 8);

    if (!metrics.length) {
      ctx.fillStyle = "#3a4a57";
      ctx.font = "500 14px system-ui, sans-serif";
      ctx.fillText("Press Train to watch the loss fall.", W / 2, H / 2);
      return;
    }

    const x = (step) => PAD + (step / totalSteps) * (W - 2 * PAD);
    const y = (loss) => H - PAD - (loss / maxLoss) * (H - 2 * PAD);
    const yAcc = (acc) => H - PAD - acc * (H - 2 * PAD); // 0..1 over full height

    // Accuracy line (amber) — up as the loss comes down.
    ctx.strokeStyle = "rgba(255,196,84,0.85)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    metrics.forEach((m, i) => {
      const px = x(m.step), py = yAcc(m.trainAcc);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Loss line (teal).
    ctx.strokeStyle = "rgba(79,227,238,0.9)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    metrics.forEach((m, i) => {
      const px = x(m.step), py = y(m.loss);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Head marker at the latest loss point.
    const lastM = metrics[metrics.length - 1];
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.beginPath();
    ctx.arc(x(lastM.step), y(lastM.loss), 2.6, 0, Math.PI * 2);
    ctx.fill();

    // Legend.
    ctx.textAlign = "left";
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.fillStyle = "rgba(79,227,238,0.9)";
    ctx.fillText("— loss", W - PAD - 118, PAD - 12);
    ctx.fillStyle = "rgba(255,196,84,0.9)";
    ctx.fillText("— train accuracy", W - PAD - 74, PAD - 12);
  }, [metrics]);

  return <canvas ref={ref} className="loss-chart" style={{ aspectRatio: `${W} / ${H}` }} />;
}
