"""Shared fixtures. The Predictor loads mnist.pth once at import, so both the
client and the raw predictor are module-cheap and reused across tests."""
import pytest
from fastapi.testclient import TestClient

from app import app, predictor as app_predictor


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def predictor():
    return app_predictor


@pytest.fixture
def blank_pixels():
    """784 zeros — a valid, all-blank input."""
    return [0.0] * (28 * 28)
