import { describe, expect, it } from "vitest";
import { canvasToInput, normalize } from "./preprocess.js";

const SIZE = 280;

/** A 280x280 canvas painted white (like DrawCanvas's blank state). */
function blankCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, SIZE, SIZE);
  return canvas;
}

/** Paint a dark filled rectangle (ink) onto a blank canvas. */
function canvasWithInkRect(x, y, w, h) {
  const canvas = blankCanvas();
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#111";
  ctx.fillRect(x, y, w, h);
  return canvas;
}

/** Center of mass (col, row) of a 784-length white-on-black input. */
function centerOfMass(input) {
  let sum = 0, cx = 0, cy = 0;
  for (let i = 0; i < input.length; i++) {
    const v = input[i];
    const c = i % 28, r = (i / 28) | 0;
    sum += v; cx += v * c; cy += v * r;
  }
  return { cx: cx / sum, cy: cy / sum, sum };
}

describe("canvasToInput", () => {
  it("returns null for a blank canvas", () => {
    expect(canvasToInput(blankCanvas())).toBeNull();
  });

  it("produces a 784-length array of values in [0,1]", () => {
    const input = canvasToInput(canvasWithInkRect(90, 90, 100, 100));
    expect(input).toHaveLength(28 * 28);
    expect(input.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it("inverts to white-on-black: ink is bright, background is ~0", () => {
    const input = canvasToInput(canvasWithInkRect(90, 90, 100, 100));
    expect(Math.max(...input)).toBeGreaterThan(0.5); // ink present and bright
    expect(input[0]).toBeLessThan(0.1); // top-left corner is background
  });

  it("centers an off-center drawing by center of mass", () => {
    // Ink pushed into the top-left quadrant; preprocessing must re-center it so
    // the center of mass lands near the middle of the 28x28 field (~14,14).
    const input = canvasToInput(canvasWithInkRect(20, 20, 70, 70));
    const { cx, cy, sum } = centerOfMass(input);
    expect(sum).toBeGreaterThan(0);
    expect(cx).toBeGreaterThan(11);
    expect(cx).toBeLessThan(17);
    expect(cy).toBeGreaterThan(11);
    expect(cy).toBeLessThan(17);
  });

  it("normalize() returns null for a blank canvas", () => {
    expect(normalize(blankCanvas())).toBeNull();
  });

  it("normalize() exposes geometry, an ink canvas, and the same input as canvasToInput", () => {
    const canvas = canvasWithInkRect(60, 40, 90, 120);
    const result = normalize(canvas);
    expect(result.input).toEqual(canvasToInput(canvas));
    expect(result.inkCanvas).toBeTruthy();
    const g = result.geometry;
    // Bounding box should roughly match the drawn rectangle (± stroke/threshold).
    expect(g.minX).toBeGreaterThanOrEqual(55);
    expect(g.minY).toBeGreaterThanOrEqual(35);
    expect(g.boxW).toBeGreaterThan(70);
    expect(g.boxH).toBeGreaterThan(100);
    // Longest side scales to 20px; center of mass lands inside the box.
    expect(g.scale).toBeCloseTo(20 / Math.max(g.boxW, g.boxH), 5);
    expect(g.comX).toBeGreaterThan(g.minX);
    expect(g.comY).toBeGreaterThan(g.minY);
  });

  it("scales the ink to fit within the 20px inner box", () => {
    // Whatever the source size, the ink's extent should be ~20px, so it never
    // fills the whole 28px field edge-to-edge.
    const input = canvasToInput(canvasWithInkRect(10, 10, 250, 250));
    let minC = 28, maxC = -1, minR = 28, maxR = -1;
    for (let i = 0; i < input.length; i++) {
      if (input[i] > 0.1) {
        const c = i % 28, r = (i / 28) | 0;
        minC = Math.min(minC, c); maxC = Math.max(maxC, c);
        minR = Math.min(minR, r); maxR = Math.max(maxR, r);
      }
    }
    expect(maxC - minC + 1).toBeLessThanOrEqual(22); // ~20px + a little smoothing
    expect(maxR - minR + 1).toBeLessThanOrEqual(22);
  });
});
