"""HTTP boundary: routes, input validation, and error mapping in app.py."""


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_inference_happy_path(client, blank_pixels):
    res = client.post("/inference", json={"pixels": blank_pixels})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"prediction", "probs", "layers", "transitions"}
    assert 0 <= body["prediction"] <= 9


def test_inference_rejects_wrong_pixel_count(client):
    res = client.post("/inference", json={"pixels": [0.0] * 100})
    assert res.status_code == 422
    assert "Expected 784 pixels, got 100" in res.text


def test_inference_rejects_missing_field(client):
    res = client.post("/inference", json={})
    assert res.status_code == 422  # pydantic validation


def test_inference_rejects_non_numeric_pixels(client):
    res = client.post("/inference", json={"pixels": ["not", "a", "number"]})
    assert res.status_code == 422


def test_cors_headers_present(client):
    # Default dev config is permissive ("*"); this just confirms the middleware
    # is wired so a browser origin gets a CORS response header back.
    res = client.get("/health", headers={"Origin": "http://localhost:8080"})
    assert res.headers.get("access-control-allow-origin") is not None
