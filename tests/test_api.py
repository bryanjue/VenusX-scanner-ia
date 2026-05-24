from fastapi.testclient import TestClient
from app.main import app  # We import the FastAPI app
import io
from PIL import Image

# We use TestClient to simulate requests without starting the real server
client = TestClient(app)

def test_scan_endpoint_returns_products():
    """
    Test that sending a valid image to /api/scan returns 
    a 200 OK status and a list of detected products.
    """
    # 1. Arrange: Create a dummy image in memory
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # 2. Act: Simulate the Frontend making a POST request
    response = client.post(
        "/api/scan",
        files={"file": ("test_image.png", img_byte_arr, "image/png")}
    )

    # 3. Assert: Check that the Backend responds as expected
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "detected_items" in data
    assert len(data["detected_items"]) > 0
    assert data["detected_items"][0]["name"] == "Mock Coca-Cola"