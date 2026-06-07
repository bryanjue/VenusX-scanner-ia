import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application configuration loaded from environment variables with sensible defaults."""

    # Server
    HOST: str = os.getenv("SCANNER_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SCANNER_PORT", "5000"))
    DEBUG: bool = os.getenv("SCANNER_DEBUG", "false").lower() == "true"

    # YOLO Model
    MODEL_PATH: str = os.getenv(
        "YOLO_MODEL_PATH",
        str(Path(__file__).parent.parent.parent / "ml_models" / "best.pt"),
    )
    CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU", "0.45"))
    DEVICE: str = os.getenv("YOLO_DEVICE", "cpu")
    USE_COCO_FALLBACK: bool = (
        os.getenv("YOLO_USE_COCO_FALLBACK", "true").lower() == "true"
    )
    COCO_MODEL: str = os.getenv("YOLO_COCO_MODEL", "yolov8n.pt")

    # Barcode Scanner
    BARCODE_ENABLED: bool = (
        os.getenv("BARCODE_ENABLED", "true").lower() == "true"
    )

    # VenusX API (product lookup)
    VENUSX_API_URL: str = os.getenv("VENUSX_API_URL", "http://localhost:8080")
    VENUSX_API_TIMEOUT: int = int(os.getenv("VENUSX_API_TIMEOUT", "5"))

    # CORS
    ALLOWED_ORIGINS: str = os.getenv("CORS_ALLOWED_ORIGINS", "*")

    # Product Mappings
    COCO_PRODUCT_MAP_RAW: Optional[str] = os.getenv("COCO_PRODUCT_MAP")

    @property
    def COCO_PRODUCT_MAP(self) -> dict[int, int]:
        raw = self.COCO_PRODUCT_MAP_RAW
        if not raw:
            return {}
        mapping: dict[int, int] = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            cls_id, barcode = pair.split(":", 1)
            try:
                mapping[int(cls_id.strip())] = int(barcode.strip())
            except ValueError:
                continue
        return mapping


settings = Settings()
