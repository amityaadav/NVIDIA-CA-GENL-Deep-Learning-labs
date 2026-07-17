import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useAnimation } from "./useAnimation.js";

const PHASES = 4;

describe("useAnimation (the Animation Controller)", () => {
  it("starts paused at progress 0, speed 1", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    expect(result.current.progress).toBe(0);
    expect(result.current.playing).toBe(false);
    expect(result.current.speed).toBe(1);
  });

  it("stepForward advances one whole layer at a time and clamps at phases", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    for (let expected = 1; expected <= PHASES; expected++) {
      act(() => result.current.stepForward());
      expect(result.current.progress).toBe(expected);
    }
    // Already at the end: stays clamped, never exceeds phases.
    act(() => result.current.stepForward());
    expect(result.current.progress).toBe(PHASES);
  });

  it("stepBack walks back down and clamps at 0", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    act(() => result.current.stepForward());
    act(() => result.current.stepForward());
    expect(result.current.progress).toBe(2);
    act(() => result.current.stepBack());
    expect(result.current.progress).toBe(1);
    act(() => result.current.stepBack());
    act(() => result.current.stepBack());
    expect(result.current.progress).toBe(0);
  });

  it("stepping pauses playback", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    act(() => result.current.play());
    expect(result.current.playing).toBe(true);
    act(() => result.current.stepForward());
    expect(result.current.playing).toBe(false);
  });

  it("play sets playing true; pause and toggle flip it", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    act(() => result.current.play());
    expect(result.current.playing).toBe(true);
    act(() => result.current.pause());
    expect(result.current.playing).toBe(false);
    act(() => result.current.toggle());
    expect(result.current.playing).toBe(true);
  });

  it("restart resets progress to 0 and stops playback", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    act(() => result.current.stepForward());
    act(() => result.current.play());
    act(() => result.current.restart());
    expect(result.current.progress).toBe(0);
    expect(result.current.playing).toBe(false);
  });

  it("setSpeed updates the speed multiplier", () => {
    const { result } = renderHook(() => useAnimation(PHASES));
    act(() => result.current.setSpeed(2.5));
    expect(result.current.speed).toBe(2.5);
  });
});
