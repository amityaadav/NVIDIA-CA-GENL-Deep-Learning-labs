"""Export mnist.pth to a flat little-endian float32 blob the browser loads.

Layout (concatenated, in this order):
  hidden_1.weight [512,784], hidden_1.bias [512],
  hidden_2.weight [512,512], hidden_2.bias [512],
  output.weight  [10,512],   output.bias  [10]

The JS model (frontend/src/lib/model.js) slices it back out with the same fixed
shapes. Run once when the weights change:  python export_weights.py
"""
import os

import torch

from model import MNISTNet

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "frontend", "public", "model", "weights.f32")


def main():
    model = MNISTNet()
    model.load_state_dict(
        torch.load(os.path.join(HERE, "mnist.pth"), map_location="cpu", weights_only=True)
    )
    tensors = [
        model.hidden_1.weight, model.hidden_1.bias,
        model.hidden_2.weight, model.hidden_2.bias,
        model.output.weight, model.output.bias,
    ]
    blob = b"".join(
        t.detach().contiguous().view(-1).numpy().astype("<f4").tobytes() for t in tensors
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(blob)
    total = sum(t.numel() for t in tensors)
    print(f"wrote {len(blob)} bytes ({total} float32 values) to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
