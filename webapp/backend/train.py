"""Train the MNIST net once and save its weights to mnist.pth.

Run this a single time to produce the weights file the API serves:

    python train.py

Takes a couple of minutes on CPU. Downloads MNIST to ./data on first run.
We save a state_dict (weights only) rather than the whole model object so the
file stays portable across PyTorch versions -- model.py owns the structure.
"""
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as transforms

from model import MNISTNet

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "mnist.pth")
DATA_DIR = os.path.join(HERE, "data")
EPOCHS = 5
BATCH_SIZE = 32


def build_loaders():
    # ToTensor scales pixels to [0, 1]. We intentionally skip mean/std
    # normalization so the frontend can send plain [0, 1] pixels and match.
    trans = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    train_set = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=trans)
    valid_set = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=trans)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    return train_loader, valid_loader


def accuracy(output, y):
    return (output.argmax(dim=1) == y).sum().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, valid_loader = build_loaders()
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    train_n = len(train_loader.dataset)
    valid_n = len(valid_loader.dataset)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += accuracy(out, y)

        model.eval()
        valid_correct = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                valid_correct += accuracy(model(x), y)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"loss={train_loss:.1f} "
            f"train_acc={train_correct / train_n:.4f} "
            f"valid_acc={valid_correct / valid_n:.4f}"
        )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
