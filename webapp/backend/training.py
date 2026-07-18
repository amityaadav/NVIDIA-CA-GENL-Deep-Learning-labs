"""Live training for the "Watch it learn" tab.

Trains a FRESH, randomly-initialized MNISTNet on a small subset of MNIST with
plain SGD, yielding per-batch metrics so the frontend can draw the loss curve in
real time. The canonical mnist.pth is never touched -- this shows the journey
toward a trained model, not the committed one.

`iter_metrics` takes datasets as arguments so it can be tested with a tiny
synthetic dataset (no download); `run` wires in the real, cached MNIST subset.
"""
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms.v2 as transforms

from inference import forward_trace, TOP_EDGES_PER_TRANSITION
from model import MNISTNet

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# Full MNIST training set (60k). Validation uses a 2k slice of the test set,
# checked periodically, to keep the live valid-accuracy cheap.
TRAIN_SUBSET = 60000
VALID_SUBSET = 2000
VALID_EVERY = 20  # evaluate the validation subset every N batches

# Weight snapshots: track a spread of hidden_1 neurons and stream their weight
# images periodically, so the UI can show templates forming from noise.
N_SNAPSHOT = 12
SNAPSHOT_COUNT = 16  # ~this many snapshots across a run


def _snapshot_neurons(n_hidden):
    return [round(i * (n_hidden - 1) / (N_SNAPSHOT - 1)) for i in range(N_SNAPSHOT)]


def _top_grad_edges(from_: str, to: str, grad: torch.Tensor) -> dict:
    """Top connections by |weight gradient| — the biggest weight updates this step.

    grad has shape [dst, src]. `sign` is the direction the weight MOVES
    (opposite the gradient): +1 = increasing, -1 = decreasing.
    """
    absg = grad.abs().flatten()
    k = min(TOP_EDGES_PER_TRANSITION, absg.numel())
    vals, idx = torch.topk(absg, k)
    n_src = grad.shape[1]
    max_v = float(vals[0]) if float(vals[0]) > 0 else 1.0
    links = []
    for flat_i in idx.tolist():
        dst_j, src_i = divmod(flat_i, n_src)
        g = float(grad[dst_j, src_i])
        links.append({
            "src": src_i,
            "dst": dst_j,
            "strength": round(abs(g) / max_v, 4),
            "sign": 1 if -g >= 0 else -1,  # weight moves opposite the gradient
        })
    return {"from": from_, "to": to, "links": links}


def _learning_snapshot(model) -> dict:
    """How much each neuron/connection is changing right now (from the gradients).

    Node value = L2 norm of a neuron's incoming weight gradient (how hard it's
    being pushed); edges = the largest single weight updates per transition.
    """
    g1, g2, go = model.hidden_1.weight.grad, model.hidden_2.weight.grad, model.output.weight.grad
    norms = lambda g: [round(v, 5) for v in g.norm(dim=1).tolist()]
    return {
        "nodes": {"hidden_1": norms(g1), "hidden_2": norms(g2), "output": norms(go)},
        "edges": [
            _top_grad_edges("input", "hidden_1", g1),
            _top_grad_edges("hidden_1", "hidden_2", g2),
            _top_grad_edges("hidden_2", "output", go),
        ],
    }

_datasets = None


def get_datasets():
    """Cached (train_subset, valid_subset). Downloads MNIST on first use."""
    global _datasets
    if _datasets is None:
        trans = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )
        train_set = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=trans)
        valid_set = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=trans)
        _datasets = (
            Subset(train_set, range(TRAIN_SUBSET)),
            Subset(valid_set, range(VALID_SUBSET)),
        )
    return _datasets


def iter_metrics(train_set, valid_set, lr: float, batch_size: int, epochs: int):
    """Yield a metric dict per training batch, then a final done/diverged dict.

    Each metric: { step, epoch, totalSteps, loss, trainAcc, validAcc|None }.
    Plain SGD so the learning rate visibly matters (converge vs. crawl vs. blow up).
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set))
    vx, vy = next(iter(valid_loader))
    fixed_x = vx[0:1]  # one fixed digit, re-run through the model each snapshot

    model = MNISTNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * epochs
    step = 0

    snap_ids = _snapshot_neurons(model.hidden_1.weight.shape[0])
    snap_every = max(1, total // SNAPSHOT_COUNT)

    def validate():
        model.eval()
        with torch.no_grad():
            acc = (model(vx).argmax(1) == vy).float().mean().item()
        model.train()
        return acc

    def templates():
        w = model.hidden_1.weight.detach()
        return [[round(float(v), 4) for v in w[i].tolist()] for i in snap_ids]

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            step += 1

            loss_v = loss.item()
            if not math.isfinite(loss_v):
                # Learning rate too high -> the loss blew up. A real lesson.
                yield {"diverged": True, "step": step, "totalSteps": total}
                return

            train_acc = (out.argmax(1) == y).float().mean().item()
            do_valid = step % VALID_EVERY == 0 or step == total
            metric = {
                "step": step,
                "epoch": epoch + 1,
                "totalSteps": total,
                "loss": round(loss_v, 5),
                "trainAcc": round(train_acc, 5),
                "validAcc": round(validate(), 5) if do_valid else None,
            }
            # At snapshot steps, attach the weight templates plus the two network
            # views: the fixed sample's forward pass (activations) and where the
            # gradients are pushing (learning). Grads are still present post-step.
            if step == 1 or step % snap_every == 0 or step == total:
                metric["templateIds"] = snap_ids
                metric["templates"] = templates()
                metric["sampleTrace"] = forward_trace(model, fixed_x)
                metric["sampleLabel"] = int(vy[0])  # the fixed digit's true class
                metric["learning"] = _learning_snapshot(model)
            yield metric

    yield {"done": True, "step": step, "totalSteps": total}


def run(lr: float, batch_size: int, epochs: int):
    """Metric generator over the real MNIST subset."""
    train_set, valid_set = get_datasets()
    return iter_metrics(train_set, valid_set, lr, batch_size, epochs)
