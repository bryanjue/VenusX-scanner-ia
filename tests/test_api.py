from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image

client = TestClient(app)


def test_scan_endpoint_accepts_image():
    """Test that sending a valid image to /api/scan returns 200 with a scan response."""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    response = client.post(
        "/api/scan",
        files={"file": ("test_image.png", img_byte_arr, "image/png")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "detected_items" in data
    assert "detected_barcodes" in data
    assert "processing_time_ms" in data


def test_scan_endpoint_rejects_non_image():
    """Test that sending a non-image file returns a 400 error."""
    response = client.post(
        "/api/scan",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


def test_health_endpoint_returns_model_info():
    """Test that the health endpoint returns model information."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert "model_loaded" in data
    assert "barcode_enabled" in data
    assert "venusx_api_connected" in data


def test_root_endpoint_returns_service_info():
    """Test that the root endpoint returns service status."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "VenusX AI Scanner"
    assert "version" in data
    assert "model_loaded" in data