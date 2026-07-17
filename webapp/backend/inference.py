"""Run one forward pass and produce the full activation trace the frontend
animates: per-layer activations plus the strongest weighted paths between
consecutive layers (the "watch it think" signal path).
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F

from model import MNISTNet, LAYER_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "mnist.pth")

# How many of the strongest edges to surface for each layer transition. The
# frontend draws these as the animated signal path; a few dozen reads clearly
# without turning into the 400k-edge hairball.
TOP_EDGES_PER_TRANSITION = 40

# Backward class-trace ("why this digit?") beam widths. Kept small so the
# highlighted sub-network stays legible per the product's "legibility over
# completeness" principle: the strongest few contributors into the target
# output, then a narrow fan-out back through each layer.
TRACE_TOP_HIDDEN2 = 6   # strongest hidden_2 neurons feeding the chosen output
TRACE_FANOUT_HIDDEN1 = 3  # strongest hidden_1 neurons feeding each of those
TRACE_FANOUT_INPUT = 3    # strongest input pixels feeding each of those

# How many of a neuron's strongest input terms to surface in its math breakdown.
NEURON_TOP_TERMS = 8

# Which layer feeds each layer (the source of a neuron's inputs).
SOURCE_LAYER = {"hidden_1": "input", "hidden_2": "hidden_1", "output": "hidden_2"}
LAYER_SIZE = {"hidden_1": 512, "hidden_2": 512, "output": 10}


class Predictor:
    """Loads the trained net once and serves activation traces."""

    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.device = torch.device("cpu")
        self.model = MNISTNet().to(self.device)
        # weights_only=True: we save a plain state_dict, so this is safe and
        # matches PyTorch's future default (also silences the load warning).
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Hooks capture each named layer's output tensor during forward().
        self._captured: dict[str, torch.Tensor] = {}
        for name in LAYER_NAMES:
            layer = getattr(self.model, name)
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._captured[name] = out.detach()

        return hook

    def _activations(self, pixels: list[float]):
        """Run one forward pass and return (post-activation, pre-activation).

        Post-activation (ReLU applied, softmax at output) values are what the
        neurons actually fire. The forward hooks sit on the Linear layers, so the
        captured tensors are the PRE-activations (raw z before ReLU; raw logits
        before softmax) -- exactly what the activation-function view needs.
        Shared by predict() and explain() so both see the same numbers.
        """
        x = torch.tensor(pixels, dtype=torch.float32, device=self.device).view(1, 1, 28, 28)

        self._captured.clear()
        with torch.no_grad():
            logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

        z1 = self._captured["hidden_1"].squeeze(0)
        z2 = self._captured["hidden_2"].squeeze(0)
        act = {
            "input": x.view(-1),
            "hidden_1": F.relu(z1),
            "hidden_2": F.relu(z2),
            "output": probs,
        }
        pre = {"hidden_1": z1, "hidden_2": z2, "output": logits.squeeze(0)}
        return act, pre

    def predict(self, pixels: list[float]) -> dict:
        """pixels: 784 floats in [0, 1], row-major 28x28. Returns the trace."""
        act, pre = self._activations(pixels)
        probs = act["output"]

        layers = [
            _layer_payload("input", act["input"], shape=[28, 28]),
            _layer_payload("hidden_1", act["hidden_1"], preacts=pre["hidden_1"]),
            _layer_payload("hidden_2", act["hidden_2"], preacts=pre["hidden_2"]),
            _layer_payload("output", act["output"], logits=pre["output"]),
        ]

        transitions = [
            self._top_edges("input", "hidden_1", act["input"], self.model.hidden_1.weight),
            self._top_edges("hidden_1", "hidden_2", act["hidden_1"], self.model.hidden_2.weight),
            self._top_edges("hidden_2", "output", act["hidden_2"], self.model.output.weight),
        ]

        return {
            "prediction": int(probs.argmax().item()),
            "probs": [round(p, 5) for p in probs.tolist()],
            "layers": layers,
            "transitions": transitions,
        }

    def explain(self, pixels: list[float], target: int) -> dict:
        """"Why this digit?" -- the sub-network that drove output `target`.

        Walks backward from the target output neuron, keeping the strongest
        contributors at each step (contribution = source activation x weight, the
        real effect on THIS input). Returns the selected node indices per layer
        and the edges connecting them, ready to highlight while dimming the rest.
        """
        act, _pre = self._activations(pixels)
        probs = act["output"]

        # 1) output[target] <- hidden_2 : strongest hidden_2 neurons for this class.
        h2_contrib = act["hidden_2"] * self.model.output.weight[target]
        h2_sel, edges_out = _beam_from_single(
            h2_contrib, dst_id=target, from_="hidden_2", to="output", k=TRACE_TOP_HIDDEN2,
        )

        # 2) hidden_2[sel] <- hidden_1 : what fed each of those hidden_2 neurons.
        h1_sel, edges_h2 = _beam_fanout(
            act["hidden_1"], self.model.hidden_2.weight, h2_sel,
            from_="hidden_1", to="hidden_2", k=TRACE_FANOUT_HIDDEN1,
        )

        # 3) hidden_1[sel] <- input : which pixels fed those hidden_1 neurons.
        in_sel, edges_in = _beam_fanout(
            act["input"], self.model.hidden_1.weight, h1_sel,
            from_="input", to="hidden_1", k=TRACE_FANOUT_INPUT,
        )

        return {
            "target": int(target),
            "prediction": int(probs.argmax().item()),
            "targetProb": round(float(probs[target]), 5),
            "nodes": {"hidden_2": h2_sel, "hidden_1": h1_sel, "input": in_sel},
            "edges": edges_out + edges_h2 + edges_in,
        }

    def neuron(self, pixels: list[float], layer: str, index: int) -> dict:
        """One neuron's computation on this input: its weighted-sum breakdown.

        Returns the bias, pre-activation z, post-activation a, the strongest
        input terms (weight x source-value), and -- for hidden_1, whose inputs
        are the 28x28 pixels -- the neuron's weights as an image ("what it looks
        for"). Powers the neuron inspector: the weight template + the math.
        """
        act, pre = self._activations(pixels)
        src = act[SOURCE_LAYER[layer]]
        linear = getattr(self.model, layer)
        w = linear.weight[index].detach()
        bias = float(linear.bias[index])

        contrib = w * src
        absv = contrib.abs()
        k = min(NEURON_TOP_TERMS, absv.numel())
        _, idx = torch.topk(absv, k)
        top_terms = [{
            "src": int(i),
            "weight": round(float(w[i]), 5),
            "value": round(float(src[i]), 5),
            "contribution": round(float(contrib[i]), 5),
        } for i in idx.tolist()]

        result = {
            "layer": layer,
            "index": int(index),
            "sourceLayer": SOURCE_LAYER[layer],
            "activation": "softmax" if layer == "output" else "relu",
            "bias": round(bias, 5),
            "z": round(float(pre[layer][index]), 5),   # logit for output
            "a": round(float(act[layer][index]), 5),    # softmax prob for output
            "topTerms": top_terms,
        }
        # Only hidden_1's weights map to the 28x28 image, so only it gets a template.
        if layer == "hidden_1":
            result["weightImage"] = [round(float(v), 5) for v in w.tolist()]
        return result

    def _top_edges(self, src_name: str, dst_name: str, src_act: torch.Tensor, weight: torch.Tensor):
        """Strongest edges src_i -> dst_j ranked by |weight[j, i] * activation_i|.

        This is the actual contribution each connection makes on THIS input, so
        the highlighted path reflects how the network reached its answer rather
        than being decorative. weight has shape [dst, src].
        """
        contrib = weight.detach() * src_act.unsqueeze(0)  # [dst, src]
        flat = contrib.abs().flatten()
        k = min(TOP_EDGES_PER_TRANSITION, flat.numel())
        top_vals, top_idx = torch.topk(flat, k)
        n_src = weight.shape[1]

        max_v = float(top_vals[0]) if float(top_vals[0]) > 0 else 1.0
        links = []
        for flat_i in top_idx.tolist():
            dst_j, src_i = divmod(flat_i, n_src)
            signed = float(contrib[dst_j, src_i])
            links.append({
                "src": src_i,
                "dst": dst_j,
                "strength": round(abs(signed) / max_v, 4),  # 0..1 for opacity/width
                "sign": 1 if signed >= 0 else -1,            # excitatory vs inhibitory
            })
        return {"from": src_name, "to": dst_name, "links": links}


def _edge(from_: str, to: str, src: int, dst: int, signed: float, max_v: float):
    """One highlighted edge, normalized to the strongest edge in its transition."""
    return {
        "from": from_,
        "to": to,
        "src": int(src),
        "dst": int(dst),
        "strength": round(abs(signed) / max_v, 4),  # 0..1 for opacity/width
        "sign": 1 if signed >= 0 else -1,            # excitatory vs inhibitory
    }


def _beam_from_single(contrib: torch.Tensor, dst_id: int, from_: str, to: str, k: int):
    """Top-k source neurons feeding a single destination neuron.

    contrib[i] is source i's contribution into dst_id. Returns the selected
    source indices (strongest first) and their edges, normalized to the top edge.
    """
    absv = contrib.abs()
    k = min(k, absv.numel())
    _, idx = torch.topk(absv, k)
    idx = idx.tolist()
    max_v = float(contrib[idx[0]].abs()) if idx else 0.0
    max_v = max_v if max_v > 0 else 1.0
    sel = list(idx)
    edges = [_edge(from_, to, i, dst_id, float(contrib[i]), max_v) for i in idx]
    return sel, edges


def _beam_fanout(src_act: torch.Tensor, weight: torch.Tensor, dsts: list[int],
                 from_: str, to: str, k: int):
    """For each destination in `dsts`, the top-k source neurons feeding it.

    weight has shape [dst, src]; contribution is src_act * weight[dst]. Edges are
    normalized to the single strongest contribution across the whole transition
    so strengths stay comparable. Returns (sorted unique source indices, edges).
    """
    raw = []  # (src, dst, signed)
    for dst in dsts:
        contrib = src_act * weight[dst]
        absv = contrib.abs()
        kk = min(k, absv.numel())
        _, idx = torch.topk(absv, kk)
        for i in idx.tolist():
            raw.append((i, dst, float(contrib[i])))

    if not raw:
        return [], []

    max_v = max(abs(s) for _, _, s in raw) or 1.0
    sel = sorted({src for src, _, _ in raw})
    edges = [_edge(from_, to, src, dst, signed, max_v) for src, dst, signed in raw]
    return sel, edges


def _layer_payload(name: str, values: torch.Tensor, shape: Optional[list[int]] = None,
                   preacts: Optional[torch.Tensor] = None,
                   logits: Optional[torch.Tensor] = None):
    vals = values.tolist()
    peak = max(vals) if vals else 1.0
    payload = {
        "name": name,
        "size": len(vals),
        # Normalized 0..1 for brightness, plus raw values for tooltips.
        "activations": [round(v, 5) for v in vals],
        "peak": round(peak, 5),
    }
    if shape is not None:
        payload["shape"] = shape
    # Pre-activations power the activation-function view: z (before ReLU) for
    # hidden layers, raw logits (before softmax) for the output layer.
    if preacts is not None:
        payload["preacts"] = [round(v, 5) for v in preacts.tolist()]
    if logits is not None:
        payload["logits"] = [round(v, 5) for v in logits.tolist()]
    return payload
