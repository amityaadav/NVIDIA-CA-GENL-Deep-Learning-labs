/** Playback controls for the forward-pass animation. */
export default function Controls({ anim, hasTrace }) {
  const { playing, toggle, stepForward, stepBack, restart, speed, setSpeed } = anim;
  return (
    <div className="controls" role="group" aria-label="Animation controls">
      <button onClick={stepBack} disabled={!hasTrace} title="Previous layer" aria-label="Previous layer">‹</button>
      <button className="primary" onClick={toggle} disabled={!hasTrace}>
        {playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <button onClick={stepForward} disabled={!hasTrace} title="Next layer" aria-label="Next layer">›</button>
      <button onClick={restart} disabled={!hasTrace} title="Restart" aria-label="Restart">↻</button>
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
    </div>
  );
}
