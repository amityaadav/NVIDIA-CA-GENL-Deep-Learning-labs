"""The MNIST feed-forward network featured in the visualizer.

Kept deliberately identical to Lab 1 so the weights and the on-screen
architecture (784 -> 512 -> 512 -> 10) are the same object. The layers are
named so the inference hooks and the frontend can refer to them by name.
"""
import torch.nn as nn

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 512
N_CLASSES = 10

# Layer names shared across the whole stack: training, hooks, and the JSON
# contract the frontend animates. Keep these in sync with the frontend.
LAYER_NAMES = ["hidden_1", "hidden_2", "output"]


class MNISTNet(nn.Module):
    """784 -> 512 -> 512 -> 10, ReLU between hidden layers.

    We expose the Linear layers as named attributes (rather than burying them
    in an nn.Sequential) so inference can register forward hooks per layer and
    read each layer's weight matrix for the "top weighted paths" animation.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.hidden_2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden_1(x))
        x = self.relu(self.hidden_2(x))
        return self.output(x)  # raw logits; softmax happens at serialization
