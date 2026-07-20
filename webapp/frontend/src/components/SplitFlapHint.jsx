import { useEffect, useState } from "react";

/**
 * The empty-canvas prompt, animated like a split-flap departure board.
 *
 * Two phases:
 *  - "reveal" runs once when the component mounts (fresh page load or after the
 *    Clear button) — cells start blank and flip in, settling left-to-right.
 *  - "idle" then repeats every few seconds: the text stays on the board and each
 *    letter briefly flips in place, landing back on the same character, so the
 *    words never disappear again.
 *
 * Purely decorative (the real text is the aria-label).
 */
const CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LINES = ["DRAW A DIGIT,", "THEN RUN INFERENCE"];
const CELLS = LINES.join("\n").split(""); // chars, incl. spaces and the "\n" break

const isFixed = (c) => c === " " || c === "\n"; // spaces/breaks don't spin
const STAGGER = 85; // ms of extra spin per position -> left-to-right ripple
const REVEAL_SPIN = 520; // ms each cell spins during the initial reveal
const FLAP_SPIN = 340; // ms each cell flips during an in-place idle flap
const TICK = 55; // ms between flap changes
const IDLE_EVERY = 5000; // ms between idle flaps

export default function SplitFlapHint() {
  const [display, setDisplay] = useState(() => CELLS.map((c) => (isFixed(c) ? c : " ")));
  // { mode: "reveal" | "idle", n } — n bumps so each idle flap re-runs the effect.
  const [cycle, setCycle] = useState({ mode: "reveal", n: 0 });

  useEffect(() => {
    const reveal = cycle.mode === "reveal";
    const spin = reveal ? REVEAL_SPIN : FLAP_SPIN;
    const start = performance.now();
    let next;
    const id = setInterval(() => {
      const t = performance.now() - start;
      setDisplay(
        CELLS.map((target, i) => {
          if (isFixed(target)) return target;
          const from = i * STAGGER;
          if (t >= from + spin) return target; // settled
          if (t >= from) return CHARSET[(Math.random() * CHARSET.length) | 0]; // flipping
          return reveal ? " " : target; // reveal starts blank; idle keeps the letter
        }),
      );
      if (t >= (CELLS.length - 1) * STAGGER + spin) {
        clearInterval(id);
        next = setTimeout(() => setCycle((c) => ({ mode: "idle", n: c.n + 1 })), IDLE_EVERY);
      }
    }, TICK);
    return () => {
      clearInterval(id);
      clearTimeout(next);
    };
  }, [cycle]);

  return (
    <div className="flap-board" role="img" aria-label="Draw a digit, then run inference">
      {CELLS.map((c, i) =>
        c === "\n" ? (
          <span key={i} className="flap-break" />
        ) : (
          <span key={i} className={`flap ${c === " " ? "flap-gap" : ""}`} aria-hidden="true">
            {/* keyed by the glyph so each change remounts and replays the flip */}
            <span key={display[i]} className="flap-glyph">
              {display[i] === " " ? " " : display[i]}
            </span>
          </span>
        ),
      )}
    </div>
  );
}
