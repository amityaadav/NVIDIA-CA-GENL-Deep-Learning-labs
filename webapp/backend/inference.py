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


class Predictor:
    """Loads the trained net once and serves activation traces."""

    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.device = torch.device("cpu")
        self.model = MNISTNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
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

    def predict(self, pixels: list[float]) -> dict:
        """pixels: 784 floats in [0, 1], row-major 28x28. Returns the trace."""
        x = torch.tensor(pixels, dtype=torch.float32, device=self.device).view(1, 1, 28, 28)

        self._captured.clear()
        with torch.no_grad():
            logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

        # Post-activation values (ReLU applied) are what the neurons actually
        # fire, so they're what we visualize -- not the raw pre-activations.
        act = {
            "input": x.view(-1),
            "hidden_1": F.relu(self._captured["hidden_1"]).squeeze(0),
            "hidden_2": F.relu(self._captured["hidden_2"]).squeeze(0),
            "output": probs,
        }

        layers = [
            _layer_payload("input", act["input"], shape=[28, 28]),
            _layer_payload("hidden_1", act["hidden_1"]),
            _layer_payload("hidden_2", act["hidden_2"]),
            _layer_payload("output", act["output"]),
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


def _layer_payload(name: str, values: torch.Tensor, shape: Optional[list[int]] = None):
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
    return payload
