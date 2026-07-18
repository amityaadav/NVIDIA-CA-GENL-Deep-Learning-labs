/** Speed control for the forward-pass animation (it auto-plays on Run). */
export default function Controls({ anim, hasTrace }) {
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
    </div>
  );
}
