import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it, beforeAll } from "vitest";
import { Model } from "./model.js";
import fixture from "./parity.fixture.json";

// Load the exported weights straight off disk (no fetch needed in the test).
function loadModel() {
  const buf = readFileSync(path.resolve(process.cwd(), "public/model/weights.f32"));
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  return new Model(f32);
}

const near = (a, b, tol = 1e-3) => Math.abs(a - b) <= tol;
const pairSet = (links) => new Set(links.map((l) => `${l.src}->${l.dst}`));

describe("JS model matches the Python backend (parity)", () => {
  let model, trace;
  beforeAll(() => {
    model = loadModel();
    trace = model.predict(fixture.pixels);
  });

  it("predicts the same digit", () => {
    expect(trace.prediction).toBe(fixture.predict.prediction);
  });

  it("produces the same probabilities", () => {
    trace.probs.forEach((p, i) => expect(near(p, fixture.predict.probs[i])).toBe(true));
    expect(near(trace.probs.reduce((a, b) => a + b, 0), 1)).toBe(true);
  });

  it("matches hidden + output activations", () => {
    for (let L = 1; L <= 3; L++) {
      const a = trace.layers[L].activations, b = fixture.predict.layers[L].activations;
      expect(a.length).toBe(b.length);
      for (let i = 0; i < a.length; i++) expect(near(a[i], b[i])).toBe(true);
    }
  });

  it("selects the same top edges per transition", () => {
    trace.transitions.forEach((t, k) => {
      expect(pairSet(t.links)).toEqual(pairSet(fixture.predict.transitions[k].links));
    });
  });

  it("matches the explain (why-this-digit) sub-network", () => {
    const e = model.explain(fixture.pixels, fixture.predict.prediction);
    expect(e.target).toBe(fixture.explain.target);
    expect(near(e.targetProb, fixture.explain.targetProb)).toBe(true);
    expect(new Set(e.nodes.hidden_2)).toEqual(new Set(fixture.explain.nodes.hidden_2));
    expect(new Set(e.nodes.hidden_1)).toEqual(new Set(fixture.explain.nodes.hidden_1));
  });

  it("matches a hidden_1 neuron breakdown + weight image", () => {
    const n = model.neuron(fixture.pixels, "hidden_1", 42);
    expect(near(n.z, fixture.neuron_h1.z)).toBe(true);
    expect(near(n.a, fixture.neuron_h1.a)).toBe(true);
    expect(near(n.bias, fixture.neuron_h1.bias)).toBe(true);
    expect(n.weightImage.length).toBe(784);
    expect(new Set(n.topTerms.map((t) => t.src)))
      .toEqual(new Set(fixture.neuron_h1.topTerms.map((t) => t.src)));
  });

  it("matches an output neuron (logit → softmax)", () => {
    const n = model.neuron(fixture.pixels, "output", fixture.predict.prediction);
    expect(n.activation).toBe("softmax");
    expect(near(n.z, fixture.neuron_out.z)).toBe(true);
    expect(near(n.a, fixture.neuron_out.a)).toBe(true);
    expect(n.weightImage).toBeUndefined();
  });
});
