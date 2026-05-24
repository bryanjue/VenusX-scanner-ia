from PIL import Image
from app.schemas.products import DetectedProduct

def process_image(image: Image.Image) -> list[DetectedProduct]:
    # TODO: Here we will integrate the actual YOLOv8 or OpenCV logic.
    # For now, we simulate that it always detects these two products:
    
    return [
        DetectedProduct(product_id="1", name="Mock Coca-Cola", quantity=1, confidence=0.95),
        DetectedProduct(product_id="2", name="Mock Patatas", quantity=2, confidence=0.88)
    ]