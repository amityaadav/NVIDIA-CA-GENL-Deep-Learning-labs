"""The other half of the contract: the trace payload the frontend animates.
These tests pin the response shape, value ranges, and the invariants the
visualization depends on (post-ReLU activations, normalized strengths, signs)."""
import math

import pytest

from inference import TOP_EDGES_PER_TRANSITION

LAYER_ORDER = ["input", "hidden_1", "hidden_2", "output"]
LAYER_SIZES = {"input": 784, "hidden_1": 512, "hidden_2": 512, "output": 10}
TRANSITIONS = [("input", "hidden_1"), ("hidden_1", "hidden_2"), ("hidden_2", "output")]


def _bar_pixels():
    """A vertical bar down the middle: not a real digit, but non-blank so every
    layer carries signal (unlike an all-zero input)."""
    pixels = [0.0] * 784
    for row in range(28):
        pixels[row * 28 + 14] = 1.0
    return pixels


@pytest.fixture(scope="module")
def trace(predictor):
    # Use a non-blank input so activation/strength invariants are meaningful.
    return predictor.predict(_bar_pixels())


def test_top_level_keys(trace):
    assert set(trace) == {"prediction", "probs", "layers", "transitions"}


def test_prediction_matches_argmax_of_probs(trace):
    assert trace["prediction"] == max(range(10), key=lambda i: trace["probs"][i])


def test_probs_are_a_distribution(trace):
    probs = trace["probs"]
    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-3)


def test_layers_shape_and_order(trace):
    layers = trace["layers"]
    assert [l["name"] for l in layers] == LAYER_ORDER
    for layer in layers:
        assert layer["size"] == LAYER_SIZES[layer["name"]]
        assert len(layer["activations"]) == layer["size"]
        assert all(math.isfinite(v) for v in layer["activations"])


def test_input_layer_carries_image_shape(trace):
    assert trace["layers"][0]["shape"] == [28, 28]


def test_hidden_activations_are_post_relu(trace):
    # Visualized activations are post-ReLU, so they can never be negative.
    for name in ("hidden_1", "hidden_2"):
        layer = next(l for l in trace["layers"] if l["name"] == name)
        assert min(layer["activations"]) >= 0.0


def test_output_activations_equal_probs(trace):
    output = next(l for l in trace["layers"] if l["name"] == "output")
    assert output["activations"] == trace["probs"]


def test_hidden_layers_carry_preactivations(trace):
    # Each hidden layer exposes pre-activations (raw z before ReLU), and
    # relu(z) must reproduce the post-activation we visualize.
    for name in ("hidden_1", "hidden_2"):
        layer = next(l for l in trace["layers"] if l["name"] == name)
        assert "preacts" in layer
        assert len(layer["preacts"]) == layer["size"]
        for z, a in zip(layer["preacts"], layer["activations"]):
            assert a == pytest.approx(max(0.0, z), abs=1e-4)


def test_input_has_no_preactivations(trace):
    input_layer = trace["layers"][0]
    assert "preacts" not in input_layer  # raw pixels have no activation function


def test_output_carries_logits_that_softmax_to_probs(trace):
    import math

    output = next(l for l in trace["layers"] if l["name"] == "output")
    logits = output["logits"]
    assert len(logits) == 10
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    softmaxed = [e / total for e in exps]
    for p, q in zip(softmaxed, trace["probs"]):
        assert p == pytest.approx(q, abs=1e-4)


def test_transitions_shape(trace):
    transitions = trace["transitions"]
    assert [(t["from"], t["to"]) for t in transitions] == TRANSITIONS


def test_links_are_valid_and_bounded(trace):
    for t in trace["transitions"]:
        src_size = LAYER_SIZES[t["from"]]
        dst_size = LAYER_SIZES[t["to"]]
        assert 0 < len(t["links"]) <= TOP_EDGES_PER_TRANSITION
        for link in t["links"]:
            assert 0 <= link["src"] < src_size
            assert 0 <= link["dst"] < dst_size
            assert 0.0 <= link["strength"] <= 1.0
            assert link["sign"] in (1, -1)


def test_strengths_are_normalized_and_descending(trace):
    # strength is |contribution| / max, so the strongest edge is 1.0 and the
    # list is sorted strongest-first (topk order).
    for t in trace["transitions"]:
        strengths = [link["strength"] for link in t["links"]]
        assert strengths[0] == pytest.approx(1.0)
        assert strengths == sorted(strengths, reverse=True)


def test_a_real_looking_digit_predicts_in_range(predictor):
    # A non-blank input must still produce a valid distribution over 10 classes.
    out = predictor.predict(_bar_pixels())
    assert 0 <= out["prediction"] <= 9
    assert math.isclose(sum(out["probs"]), 1.0, abs_tol=1e-3)


def test_blank_input_yields_zero_strength_edges(predictor):
    # Degenerate but valid: with no ink, input->hidden_1 contributions are all
    # zero, so that transition's edge strengths fall back to 0.0 rather than
    # being normalized to 1.0. Pinned so this behavior is intentional, not a bug.
    out = predictor.predict([0.0] * 784)
    first = out["transitions"][0]
    assert first["from"] == "input"
    assert all(link["strength"] == 0.0 for link in first["links"])
