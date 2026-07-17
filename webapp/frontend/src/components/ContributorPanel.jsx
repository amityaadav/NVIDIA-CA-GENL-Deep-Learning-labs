import { describeInputRegion, runnerUp } from "../lib/inspect.js";

/**
 * Plain-language summary of a "why this digit?" trace: the strongest hidden
 * neurons feeding the chosen output, where the driving ink sits, and the
 * runner-up class. Renders nothing until an output digit is focused.
 */
export default function ContributorPanel({ trace, focus, loading, onClear }) {
  if (loading) {
    return <div className="explain-panel">Tracing contributors…</div>;
  }
  if (!focus || !trace) {
    return (
      <p className="explain-hint muted">
        Tip: click an output digit to trace why the network chose it.
      </p>
    );
  }

  // Strongest hidden_2 → output edges are the top drivers of this class.
  const drivers = focus.edges
    .filter((e) => e.to === "output")
    .slice(0, 4);
  const region = describeInputRegion(focus.nodes.input);
  const ru = runnerUp(trace.probs, focus.target);

  return (
    <div className="explain-panel">
      <div className="explain-head">
        <span>
          Why <strong>{focus.target}</strong>?
          <span className="muted"> · {(focus.targetProb * 100).toFixed(1)}% confidence</span>
        </span>
        <button className="explain-clear" onClick={onClear}>Clear</button>
      </div>

      <div className="explain-section">
        <span className="explain-label">Top hidden drivers</span>
        <ul className="explain-list">
          {drivers.map((e, i) => (
            <li key={i}>
              <span className="mono">hidden_2 #{e.src}</span>
              <span className={e.sign > 0 ? "excite-text" : "inhibit-text"}>
                {e.sign > 0 ? "excites" : "inhibits"} · {(e.strength * 100).toFixed(0)}%
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className="explain-section">
        <span className="explain-label">Driving ink</span>
        <span>concentrated in the <strong>{region}</strong> of the drawing</span>
      </div>

      {ru.digit >= 0 && (
        <div className="explain-section">
          <span className="explain-label">Next-closest</span>
          <span>digit <strong>{ru.digit}</strong> at {(ru.prob * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
}
