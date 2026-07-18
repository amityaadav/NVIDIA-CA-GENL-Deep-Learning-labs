"""Live-training metric generator and the /train endpoint's validation.

Uses a tiny synthetic dataset so tests need no MNIST download or network.
"""
import torch
from torch.utils.data import TensorDataset

from training import iter_metrics


def _fake_dataset(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return TensorDataset(x, y)


def test_iter_metrics_shape_and_progress():
    train = _fake_dataset(64)
    valid = _fake_dataset(20, seed=1)
    metrics = list(iter_metrics(train, valid, lr=0.1, batch_size=16, epochs=2))

    # 64/16 = 4 batches/epoch x 2 epochs = 8 steps, plus a final done event.
    steps = [m for m in metrics if "loss" in m]
    assert len(steps) == 8
    assert metrics[-1] == {"done": True, "step": 8, "totalSteps": 8}

    for i, m in enumerate(steps, start=1):
        assert m["step"] == i
        assert m["totalSteps"] == 8
        assert 0.0 <= m["trainAcc"] <= 1.0
        assert m["validAcc"] is None or 0.0 <= m["validAcc"] <= 1.0


def test_weight_snapshots_are_attached():
    train = _fake_dataset(64)
    valid = _fake_dataset(20, seed=1)
    steps = [m for m in iter_metrics(train, valid, lr=0.1, batch_size=16, epochs=1) if "loss" in m]
    snaps = [m for m in steps if "templates" in m]
    assert snaps, "expected at least one weight snapshot"
    first = snaps[0]
    assert len(first["templateIds"]) == 12
    assert len(first["templates"]) == 12
    assert all(len(t) == 784 for t in first["templates"])  # hidden_1 weights = 28x28


def test_snapshot_includes_activation_and_learning_traces():
    train = _fake_dataset(64)
    valid = _fake_dataset(20, seed=1)
    snap = next(m for m in iter_metrics(train, valid, lr=0.1, batch_size=16, epochs=1) if "templates" in m)

    # A: fixed-sample forward trace (same shape /inference returns).
    trace = snap["sampleTrace"]
    assert [l["name"] for l in trace["layers"]] == ["input", "hidden_1", "hidden_2", "output"]
    assert len(trace["transitions"]) == 3
    assert 0 <= trace["prediction"] <= 9

    # B: learning snapshot — per-neuron grad norms + top weight-update edges.
    learn = snap["learning"]
    assert len(learn["nodes"]["hidden_1"]) == 512
    assert len(learn["nodes"]["output"]) == 10
    assert all(v >= 0 for v in learn["nodes"]["hidden_1"])  # norms are non-negative
    assert [(e["from"], e["to"]) for e in learn["edges"]] == \
        [("input", "hidden_1"), ("hidden_1", "hidden_2"), ("hidden_2", "output")]
    for e in learn["edges"]:
        assert all(link["sign"] in (1, -1) and 0 <= link["strength"] <= 1 for link in e["links"])


def test_validation_runs_on_the_last_step():
    train = _fake_dataset(32)
    valid = _fake_dataset(20, seed=1)
    steps = [m for m in iter_metrics(train, valid, lr=0.1, batch_size=16, epochs=1) if "loss" in m]
    assert steps[-1]["validAcc"] is not None  # validated on the final step


def test_high_learning_rate_diverges():
    train = _fake_dataset(64)
    valid = _fake_dataset(20, seed=1)
    # An extreme learning rate blows the weights up -> non-finite loss.
    metrics = list(iter_metrics(train, valid, lr=1000.0, batch_size=16, epochs=5))
    assert metrics[-1].get("diverged") is True
    assert not any(m.get("done") for m in metrics)  # stopped early, no clean finish


def test_loss_generally_decreases_on_a_learnable_signal():
    # Labels correlate with mean pixel intensity, so SGD should reduce the loss.
    g = torch.Generator().manual_seed(3)
    x = torch.rand(200, 1, 28, 28, generator=g)
    y = (x.mean(dim=(1, 2, 3)) * 10).clamp(0, 9).long()
    ds = TensorDataset(x, y)
    losses = [m["loss"] for m in iter_metrics(ds, ds, lr=0.5, batch_size=32, epochs=4) if "loss" in m]
    assert losses[-1] < losses[0]


# --- HTTP boundary (validation only; a full stream would train for real) ---

def test_train_rejects_out_of_range_lr(client):
    assert client.get("/train", params={"lr": 99}).status_code == 422


def test_train_rejects_bad_batch_size(client):
    assert client.get("/train", params={"batch_size": 7}).status_code == 422


def test_train_rejects_out_of_range_epochs(client):
    assert client.get("/train", params={"epochs": 0}).status_code == 422
    assert client.get("/train", params={"epochs": 99}).status_code == 422
