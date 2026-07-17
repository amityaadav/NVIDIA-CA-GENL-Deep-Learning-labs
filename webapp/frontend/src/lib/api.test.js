import { afterEach, describe, expect, it, vi } from "vitest";
import { runInference, runExplain, runNeuron } from "./api.js";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("runInference", () => {
  it("POSTs pixels to /inference and returns the parsed trace", async () => {
    const trace = { prediction: 3, probs: [], layers: [], transitions: [] };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => trace,
    });
    vi.stubGlobal("fetch", fetchMock);

    const pixels = new Array(784).fill(0);
    const result = await runInference(pixels);

    expect(result).toEqual(trace);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/inference$/);
    expect(opts.method).toBe("POST");
    expect(opts.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(opts.body)).toEqual({ pixels });
  });

  it("throws with the status code when the response is not ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        text: async () => "Expected 784 pixels, got 3",
      }),
    );

    await expect(runInference([0, 0, 0])).rejects.toThrow(/422/);
  });
});

describe("runExplain", () => {
  it("POSTs pixels + target to /explain and returns the parsed trace", async () => {
    const explanation = { target: 7, nodes: {}, edges: [] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => explanation });
    vi.stubGlobal("fetch", fetchMock);

    const pixels = new Array(784).fill(0);
    const result = await runExplain(pixels, 7);

    expect(result).toEqual(explanation);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/explain$/);
    expect(JSON.parse(opts.body)).toEqual({ pixels, target: 7 });
  });

  it("throws when the response is not ok", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, text: async () => "" }));
    await expect(runExplain([], 3)).rejects.toThrow(/500/);
  });
});

describe("runNeuron", () => {
  it("POSTs pixels + layer + index to /neuron and returns the breakdown", async () => {
    const breakdown = { layer: "hidden_1", index: 42, topTerms: [] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => breakdown });
    vi.stubGlobal("fetch", fetchMock);

    const pixels = new Array(784).fill(0);
    const result = await runNeuron(pixels, "hidden_1", 42);

    expect(result).toEqual(breakdown);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/neuron$/);
    expect(JSON.parse(opts.body)).toEqual({ pixels, layer: "hidden_1", index: 42 });
  });

  it("throws when the response is not ok", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 422, text: async () => "" }));
    await expect(runNeuron([], "output", 99)).rejects.toThrow(/422/);
  });
});
