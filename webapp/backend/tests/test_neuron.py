"""The /neuron endpoint: a single neuron's weighted-sum breakdown, and the
hidden_1 weight image ("what is this neuron looking for?")."""
import pytest

from inference import NEURON_TOP_TERMS


def _bar_pixels():
    pixels = [0.0] * 784
    for row in range(28):
        pixels[row * 28 + 14] = 1.0
    return pixels


@pytest.fixture(scope="module")
def h1_neuron(predictor):
    return predictor.neuron(_bar_pixels(), "hidden_1", 42)


def test_breakdown_shape(h1_neuron):
    assert h1_neuron["layer"] == "hidden_1"
    assert h1_neuron["index"] == 42
    assert h1_neuron["sourceLayer"] == "input"
    assert h1_neuron["activation"] == "relu"
    assert 0 < len(h1_neuron["topTerms"]) <= NEURON_TOP_TERMS


def test_terms_are_weight_times_value(h1_neuron):
    for t in h1_neuron["topTerms"]:
        assert t["contribution"] == pytest.approx(t["weight"] * t["value"], abs=1e-3)


def test_terms_sorted_by_absolute_contribution(h1_neuron):
    mags = [abs(t["contribution"]) for t in h1_neuron["topTerms"]]
    assert mags == sorted(mags, reverse=True)


def test_z_equals_full_weighted_sum_plus_bias(predictor):
    # z must equal (sum over ALL inputs of weight*value) + bias, not just the
    # surfaced top terms. Verify against the full contribution recomputed here.
    import torch

    pixels = _bar_pixels()
    n = predictor.neuron(pixels, "hidden_1", 42)
    w = predictor.model.hidden_1.weight[42].detach()
    x = torch.tensor(pixels, dtype=torch.float32)
    full = float((w * x).sum()) + n["bias"]
    assert n["z"] == pytest.approx(full, abs=1e-3)


def test_relu_of_z_equals_a(h1_neuron):
    assert h1_neuron["a"] == pytest.approx(max(0.0, h1_neuron["z"]), abs=1e-4)


def test_hidden_1_has_weight_image(h1_neuron):
    assert "weightImage" in h1_neuron
    assert len(h1_neuron["weightImage"]) == 784


def test_hidden_2_and_output_have_no_weight_image(predictor):
    pixels = _bar_pixels()
    assert "weightImage" not in predictor.neuron(pixels, "hidden_2", 10)
    out = predictor.neuron(pixels, "output", 3)
    assert "weightImage" not in out
    assert out["activation"] == "softmax"
    # For output, a is the softmax probability (0..1) and z is the raw logit.
    assert 0.0 <= out["a"] <= 1.0


def test_output_neuron_a_matches_prediction_prob(predictor):
    pixels = _bar_pixels()
    trace = predictor.predict(pixels)
    pred = trace["prediction"]
    n = predictor.neuron(pixels, "output", pred)
    assert n["a"] == pytest.approx(trace["probs"][pred], abs=1e-4)


# --- HTTP boundary ---

def test_neuron_endpoint_happy_path(client):
    res = client.post("/neuron", json={"pixels": _bar_pixels(), "layer": "hidden_1", "index": 0})
    assert res.status_code == 200
    assert res.json()["layer"] == "hidden_1"


def test_neuron_rejects_bad_layer(client):
    res = client.post("/neuron", json={"pixels": _bar_pixels(), "layer": "input", "index": 0})
    assert res.status_code == 422  # Literal rejects "input"


def test_neuron_rejects_index_out_of_range(client):
    res = client.post("/neuron", json={"pixels": _bar_pixels(), "layer": "output", "index": 10})
    assert res.status_code == 422
    assert "out of range" in res.text


def test_neuron_rejects_wrong_pixel_count(client):
    res = client.post("/neuron", json={"pixels": [0.0] * 5, "layer": "hidden_1", "index": 0})
    assert res.status_code == 422
