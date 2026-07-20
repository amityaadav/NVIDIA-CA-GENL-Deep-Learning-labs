/**
 * Speed control for the forward-pass animation (it auto-plays on Run).
 * `children`, when given, render as a second control after a separator (the
 * Connections slider lives here so both sliders sit together under the graph).
 */
export default function Controls({ anim, hasTrace, children }) {
  const { speed, setSpeed } = anim;
  return (
    <div className="controls" role="group" aria-label="Animation controls">
      <label className="speed">
        Speed
        <input
          type="range"
          min="0.25"
          max="2.5"
          step="0.25"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          disabled={!hasTrace}
        />
        <span className="speed-val">{speed.toFixed(2)}×</span>
      </label>
      {children && <span className="ctrl-sep" aria-hidden="true" />}
      {children}
    </div>
  );
}
