import { describe, expect, it } from "vitest";
import {
  COL,
  LAYER_KEYS,
  computePositions,
  hitTest,
  describeNeuron,
  describeInputRegion,
  runnerUp,
  reluNormalizedPoint,
  LAYER_INFO,
} from "./inspect.js";

const positions = computePositions();

/** Minimal trace with known activation values for the describe tests. */
function fakeTrace() {
  const layer = (name, size, fill, extra = {}) => ({
    name,
    size,
    activations: Array.from({ length: size }, (_, i) => fill(i)),
    peak: 1,
    ...extra,
  });
  return {
    prediction: 3,
    probs: [],
    layers: [
      layer("input", 784, (i) => (i === 0 ? 0.75 : 0)),
      // #5 fires (z=8.25), #6 is dead (z=-1.5 -> relu 0).
      layer("hidden_1", 512, (i) => (i === 5 ? 8.25 : 0), {
        preacts: Array.from({ length: 512 }, (_, i) => (i === 5 ? 8.25 : i === 6 ? -1.5 : 0)),
      }),
      layer("hidden_2", 512, () => 0, {
        preacts: Array.from({ length: 512 }, () => -1),
      }),
      layer("output", 10, (i) => (i === 3 ? 0.912 : 0), {
        logits: Array.from({ length: 10 }, (_, i) => (i === 3 ? 6.2 : 0)),
      }),
    ],
    transitions: [],
  };
}

describe("computePositions", () => {
  it("produces one position array per layer with the right counts", () => {
    expect(positions).toHaveLength(4);
    expect(positions[COL.input]).toHaveLength(784);
    expect(positions[COL.hidden_1]).toHaveLength(512);
    expect(positions[COL.hidden_2]).toHaveLength(512);
    expect(positions[COL.output]).toHaveLength(10);
  });

  it("gives input cells a cell size and finite coordinates", () => {
    const p = positions[COL.input][0];
    expect(p.cell).toBeGreaterThan(0);
    expect(Number.isFinite(p.x) && Number.isFinite(p.y)).toBe(true);
  });
});

describe("hitTest", () => {
  it("returns null when the point is far from every neuron", () => {
    expect(hitTest(positions, -100, -100)).toBeNull();
  });

  it("hits the exact neuron when pointing at its center", () => {
    const target = positions[COL.hidden_1][42];
    const hit = hitTest(positions, target.x, target.y);
    expect(hit).not.toBeNull();
    expect(hit.layer).toBe(COL.hidden_1);
    expect(hit.index).toBe(42);
  });

  it("resolves an output node (large hit radius) from a few px away", () => {
    const target = positions[COL.output][7];
    const hit = hitTest(positions, target.x + 8, target.y - 6);
    expect(hit.layer).toBe(COL.output);
    expect(hit.index).toBe(7);
  });

  it("reports a normalized distance in [0,1]", () => {
    const target = positions[COL.output][2];
    const hit = hitTest(positions, target.x, target.y);
    expect(hit.norm).toBeGreaterThanOrEqual(0);
    expect(hit.norm).toBeLessThanOrEqual(1);
  });
});

describe("describeNeuron", () => {
  const trace = fakeTrace();

  it("labels an input pixel with its (row, col) and intensity", () => {
    const d = describeNeuron(trace, { layer: COL.input, index: 0 });
    expect(d.label).toBe("INPUT · px (0, 0)");
    expect(d.value).toBe("intensity 0.75");
  });

  it("labels a hidden neuron with its index, activation, and z→a mapping", () => {
    const d = describeNeuron(trace, { layer: COL.hidden_1, index: 5 });
    expect(d.label).toBe("HIDDEN_1 · #5");
    expect(d.value).toBe("activation 8.25");
    expect(d.sub).toBe("z 8.25 → ReLU → 8.25");
  });

  it("flags a dead hidden neuron (negative pre-activation) in its sub line", () => {
    const d = describeNeuron(trace, { layer: COL.hidden_1, index: 6 });
    expect(d.value).toBe("activation 0.00");
    expect(d.sub).toBe("z -1.50 → ReLU → 0.00 · dead");
  });

  it("labels an output neuron with the digit, confidence, and logit→softmax", () => {
    const d = describeNeuron(trace, { layer: COL.output, index: 3 });
    expect(d.label).toBe("OUTPUT · digit 3");
    expect(d.value).toBe("91.2% confidence");
    expect(d.sub).toBe("logit 6.20 → softmax → 91.2%");
  });

  it("notes that input pixels have no activation function", () => {
    const d = describeNeuron(trace, { layer: COL.input, index: 0 });
    expect(d.sub).toBe("no activation (raw pixel)");
  });

  it("LAYER_KEYS and COL are inverse maps", () => {
    LAYER_KEYS.forEach((name, i) => expect(COL[name]).toBe(i));
  });
});

describe("describeInputRegion", () => {
  const rc = (r, c) => r * 28 + c; // 28x28 row-major index

  it("falls back when there are no pixels", () => {
    expect(describeInputRegion([])).toBe("no clear region");
  });

  it("names the center", () => {
    expect(describeInputRegion([rc(14, 14), rc(13, 14)])).toBe("center");
  });

  it("names a corner as 'vert horiz'", () => {
    expect(describeInputRegion([rc(2, 2), rc(3, 3)])).toBe("top left");
    expect(describeInputRegion([rc(25, 25), rc(24, 24)])).toBe("bottom right");
  });

  it("names a pure vertical or horizontal offset with one word", () => {
    expect(describeInputRegion([rc(2, 14)])).toBe("top");
    expect(describeInputRegion([rc(14, 25)])).toBe("right");
  });
});

describe("reluNormalizedPoint", () => {
  it("maps z = 0 to the kink: x centered, y at the floor", () => {
    expect(reluNormalizedPoint(0, 6)).toEqual({ x: 0.5, y: 0 });
  });

  it("maps a positive z onto the rising part of the curve", () => {
    expect(reluNormalizedPoint(3, 6)).toEqual({ x: 0.75, y: 0.5 });
  });

  it("keeps negative z on the flat floor (y = 0)", () => {
    const p = reluNormalizedPoint(-3, 6);
    expect(p.x).toBe(0.25);
    expect(p.y).toBe(0);
  });

  it("clamps values beyond the display range to [0,1]", () => {
    expect(reluNormalizedPoint(100, 6)).toEqual({ x: 1, y: 1 });
    expect(reluNormalizedPoint(-100, 6)).toEqual({ x: 0, y: 0 });
  });
});

describe("LAYER_INFO", () => {
  it("has a title and body for every layer", () => {
    for (const name of LAYER_KEYS) {
      expect(LAYER_INFO[name]).toBeTruthy();
      expect(LAYER_INFO[name].title.length).toBeGreaterThan(0);
      expect(LAYER_INFO[name].body.length).toBeGreaterThan(0);
    }
  });

  it("explains the activation function used at each layer", () => {
    expect(LAYER_INFO.hidden_1.body).toMatch(/ReLU/);
    expect(LAYER_INFO.output.body).toMatch(/[Ss]oftmax/);
    expect(LAYER_INFO.input.body).toMatch(/no activation/i);
  });
});

describe("runnerUp", () => {
  it("returns the highest-probability digit that is not the excluded one", () => {
    const probs = [0.05, 0.7, 0.2, 0.05, 0, 0, 0, 0, 0, 0];
    expect(runnerUp(probs, 1)).toEqual({ digit: 2, prob: 0.2 });
  });

  it("skips the excluded index even if it is the max", () => {
    const probs = [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0, 0];
    const ru = runnerUp(probs, 1);
    expect(ru.digit).toBe(0);
    expect(ru.prob).toBeCloseTo(0.1);
  });
});
