""""Why this digit?" backward class-trace: the /explain endpoint and the beam.

These pin the sub-network payload the frontend highlights: valid node indices
per layer, edges that connect only those nodes, normalized strengths, and signs.
"""
import pytest

from inference import TRACE_TOP_HIDDEN2, TRACE_FANOUT_HIDDEN1, TRACE_FANOUT_INPUT

LAYER_SIZES = {"input": 784, "hidden_1": 512, "hidden_2": 512, "output": 10}


def _bar_pixels():
    pixels = [0.0] * 784
    for row in range(28):
        pixels[row * 28 + 14] = 1.0
    return pixels


@pytest.fixture(scope="module")
def explanation(predictor):
    return predictor.explain(_bar_pixels(), target=7)


def test_top_level_shape(explanation):
    assert set(explanation) == {"target", "prediction", "targetProb", "nodes", "edges"}
    assert explanation["target"] == 7
    assert 0.0 <= explanation["targetProb"] <= 1.0


def test_selected_nodes_are_valid_and_bounded(explanation):
    nodes = explanation["nodes"]
    assert 0 < len(nodes["hidden_2"]) <= TRACE_TOP_HIDDEN2
    assert len(nodes["hidden_1"]) <= TRACE_TOP_HIDDEN2 * TRACE_FANOUT_HIDDEN1
    assert len(nodes["input"]) <= len(nodes["hidden_1"]) * TRACE_FANOUT_INPUT
    for layer, indices in nodes.items():
        assert all(0 <= i < LAYER_SIZES[layer] for i in indices)
        assert len(indices) == len(set(indices)), f"{layer} node indices must be unique"


def test_edges_connect_only_selected_nodes(explanation):
    nodes = explanation["nodes"]
    selectable = {
        ("hidden_2", "output"): (set(nodes["hidden_2"]), {explanation["target"]}),
        ("hidden_1", "hidden_2"): (set(nodes["hidden_1"]), set(nodes["hidden_2"])),
        ("input", "hidden_1"): (set(nodes["input"]), set(nodes["hidden_1"])),
    }
    seen_transitions = set()
    for e in explanation["edges"]:
        key = (e["from"], e["to"])
        assert key in selectable
        seen_transitions.add(key)
        src_ok, dst_ok = selectable[key]
        assert e["src"] in src_ok
        assert e["dst"] in dst_ok
        assert 0.0 <= e["strength"] <= 1.0
        assert e["sign"] in (1, -1)
    # All three backward transitions are represented.
    assert seen_transitions == set(selectable)


def test_each_transition_is_normalized_to_one(explanation):
    # Strongest edge in every transition normalizes to strength 1.0.
    for key in [("hidden_2", "output"), ("hidden_1", "hidden_2"), ("input", "hidden_1")]:
        strengths = [e["strength"] for e in explanation["edges"] if (e["from"], e["to"]) == key]
        assert max(strengths) == pytest.approx(1.0)


def test_explaining_the_predicted_class_reports_matching_prob(predictor):
    pixels = _bar_pixels()
    trace = predictor.predict(pixels)
    pred = trace["prediction"]
    exp = predictor.explain(pixels, target=pred)
    assert exp["prediction"] == pred
    assert exp["targetProb"] == pytest.approx(trace["probs"][pred], abs=1e-4)


# --- HTTP boundary ---

def test_explain_endpoint_happy_path(client):
    res = client.post("/explain", json={"pixels": _bar_pixels(), "target": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["target"] == 3
    assert set(body["nodes"]) == {"hidden_2", "hidden_1", "input"}


def test_explain_rejects_wrong_pixel_count(client):
    res = client.post("/explain", json={"pixels": [0.0] * 10, "target": 3})
    assert res.status_code == 422
    assert "Expected 784 pixels" in res.text


def test_explain_rejects_out_of_range_target(client):
    res = client.post("/explain", json={"pixels": _bar_pixels(), "target": 99})
    assert res.status_code == 422


def test_explain_rejects_missing_target(client):
    res = client.post("/explain", json={"pixels": _bar_pixels()})
    assert res.status_code == 422
