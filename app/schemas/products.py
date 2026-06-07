from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Detection Input ─────────────────────────────────────────

class ScanRequest(BaseModel):
    """Metadata that can be sent alongside the image (optional)."""
    store_id: Optional[str] = None
    source: Optional[str] = "manual"  # "manual", "webcam", "batch"


# ── Detection Output ─────────────────────────────────────────

class DetectedBarcode(BaseModel):
    """A single barcode found in the image."""
    data: str = Field(..., description="Decoded barcode value (e.g. 8423456789012)")
    type: str = Field(..., description="Symbology type, e.g. EAN13, CODE128, QR")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Normalised bounding-box corners in [0, 1] relative to image size
    polygon: list[dict[str, float]] = Field(default_factory=list)


class DetectedProduct(BaseModel):
    """A single product detected in the image either by YOLO or barcode."""

    # Identification
    product_id: Optional[str] = None
    barcode: Optional[str] = None          # Resolved barcode (from lookup)
    name: str = "Unknown"
    description: Optional[str] = None

    # Detection info
    confidence: float = Field(..., ge=0.0, le=1.0)
    detection_method: str = Field(
        ..., description="How the product was detected: 'yolo', 'barcode', 'combined'"
    )

    # Price (when available from VenusX API lookup)
    price: float = Field(default=0.0, ge=0.0)

    # Quantity / count
    quantity: int = Field(default=1, ge=1)

    # Optional bounding box (normalised 0-1)
    bbox: Optional[dict[str, float]] = None  # {"x1": ..., "y1": ..., "x2": ..., "y2": ...}

    # Coco class label if detected by YOLO
    coco_class: Optional[str] = None
    coco_class_id: Optional[int] = None


class ScanResponse(BaseModel):
    success: bool
    message: str
    detected_items: list[DetectedProduct] = Field(default_factory=list)
    detected_barcodes: list[DetectedBarcode] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ── Health ────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_loaded: bool
    model_path: str
    device: str
    barcode_enabled: bool
    venusx_api_connected: bool
    classes: Optional[list[dict[str, Any]]] = None