import { describe, expect, it, vi } from "vitest";

// api.js delegates to the in-browser model; mock it so we test the wiring only.
// (model.test.js proves the model itself matches the Python backend.)
vi.mock("./model.js", () => ({
  getModel: vi.fn(async () => ({
    predict: (pixels) => ({ prediction: 3, pixels }),
    explain: (pixels, target) => ({ target, pixels }),
    neuron: (pixels, layer, index) => ({ layer, index }),
  })),
}));

import { runInference, runExplain, runNeuron } from "./api.js";

describe("api delegates to the in-browser model", () => {
  it("runInference → model.predict(pixels)", async () => {
    const r = await runInference([0.1, 0.2]);
    expect(r.prediction).toBe(3);
    expect(r.pixels).toEqual([0.1, 0.2]);
  });

  it("runExplain → model.explain(pixels, target)", async () => {
    expect((await runExplain([], 7)).target).toBe(7);
  });

  it("runNeuron → model.neuron(pixels, layer, index)", async () => {
    expect(await runNeuron([], "hidden_1", 5)).toEqual({ layer: "hidden_1", index: 5 });
  });
});
