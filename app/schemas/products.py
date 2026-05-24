from pydantic import BaseModel
from typing import List

# The rule for a single detected product
class DetectedProduct(BaseModel):
    product_id: str
    name: str
    quantity: int
    confidence: float

# The rule for the final API response
class ScanResponse(BaseModel):
    success: bool
    message: str
    detected_items: List[DetectedProduct]