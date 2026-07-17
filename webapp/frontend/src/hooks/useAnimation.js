import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Drives a continuous `progress` value from 0 to `phases` (one phase per layer).
 * Layer L is fully "lit" when progress >= L; the transition into layer L
 * animates while progress is in [L-1, L]. NetworkView reads progress to draw.
 *
 * Exposes play/pause/step/restart and a speed multiplier -- the Animation
 * Controller from the architecture doc.
 */
export function useAnimation(phases, { msPerPhase = 900 } = {}) {
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const rafRef = useRef(0);
  const lastRef = useRef(0);
  const speedRef = useRef(speed);
  speedRef.current = speed;

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
  }, []);

  useEffect(() => {
    if (!playing) return;
    lastRef.current = performance.now();

    const tick = (now) => {
      const dt = now - lastRef.current;
      lastRef.current = now;
      setProgress((p) => {
        const next = p + (dt / msPerPhase) * speedRef.current;
        if (next >= phases) {
          setPlaying(false);
          return phases;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return stop;
  }, [playing, phases, msPerPhase, stop]);

  const play = useCallback(() => {
    setProgress((p) => (p >= phases ? 0 : p)); // replay from start if finished
    setPlaying(true);
  }, [phases]);

  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => (playing ? pause() : play()), [playing, play, pause]);

  const restart = useCallback(() => {
    setPlaying(false);
    setProgress(0);
  }, []);

  const stepForward = useCallback(() => {
    setPlaying(false);
    setProgress((p) => Math.min(phases, Math.floor(p + 1e-6) + 1));
  }, [phases]);

  const stepBack = useCallback(() => {
    setPlaying(false);
    setProgress((p) => Math.max(0, Math.ceil(p - 1e-6) - 1));
  }, []);

  return { progress, playing, speed, setSpeed, play, pause, toggle, restart, stepForward, stepBack };
}
