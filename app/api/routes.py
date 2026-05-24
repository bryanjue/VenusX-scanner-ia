from fastapi import APIRouter, File, UploadFile
import io
from PIL import Image
from app.services.vision_model import process_image
from app.schemas.products import ScanResponse

router = APIRouter()

@router.post("/scan", response_model=ScanResponse)
async def scan_ticket(file: UploadFile = File(...)):
    # Read the received image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Pass the image to our AI service
    detected_items = process_image(image)
    
    return ScanResponse(
        success=True,
        message="Image processed successfully",
        detected_items=detected_items
    )