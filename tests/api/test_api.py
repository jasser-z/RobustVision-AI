from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def png():
    b = BytesIO()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(b, "PNG")
    return b.getvalue()


def test_health():
    assert client.get("/health").status_code == 200


def test_models():
    assert client.get("/models").status_code == 200


def test_bad_image():
    assert (
        client.post(
            "/predict/baseline_cnn", files={"file": ("x.txt", b"bad", "text/plain")}
        ).status_code
        == 415
    )


def test_corruption_preview():
    assert (
        client.post(
            "/preview-corruption/jpeg?severity=2", files={"file": ("x.png", png(), "image/png")}
        ).status_code
        == 200
    )
