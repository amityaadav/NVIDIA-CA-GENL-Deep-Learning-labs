"""The network's shape is half of the shared contract. If these constants or
layer names drift, the frontend layout and the inference hooks break with it."""
import torch
import torch.nn as nn

from model import MNISTNet, LAYER_NAMES, INPUT_SIZE, HIDDEN_SIZE, N_CLASSES


def test_shape_constants():
    assert INPUT_SIZE == 28 * 28
    assert HIDDEN_SIZE == 512
    assert N_CLASSES == 10


def test_layer_names_are_the_contract():
    # These exact names are what the frontend's COL map and the README rely on.
    assert LAYER_NAMES == ["hidden_1", "hidden_2", "output"]


def test_named_layers_exist_and_are_linear():
    model = MNISTNet()
    for name in LAYER_NAMES:
        layer = getattr(model, name)
        assert isinstance(layer, nn.Linear), f"{name} must stay a named nn.Linear for the hooks"


def test_weight_matrix_dimensions():
    # inference reads weight[dst, src]; these dims must match the layer sizes.
    model = MNISTNet()
    assert model.hidden_1.weight.shape == (HIDDEN_SIZE, INPUT_SIZE)
    assert model.hidden_2.weight.shape == (HIDDEN_SIZE, HIDDEN_SIZE)
    assert model.output.weight.shape == (N_CLASSES, HIDDEN_SIZE)


def test_forward_maps_batch_to_logits():
    model = MNISTNet().eval()
    x = torch.zeros(4, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, N_CLASSES)


def test_forward_returns_raw_logits_not_probabilities():
    # forward() must NOT softmax — inference.py owns that at serialization time.
    model = MNISTNet().eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)[0]
    # Softmax outputs sum to 1; raw logits (essentially) never do.
    assert abs(out.sum().item() - 1.0) > 1e-3
