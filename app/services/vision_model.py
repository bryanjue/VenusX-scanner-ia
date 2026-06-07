"""
Vision model service using Ultralytics YOLOv8 for product detection.

The service loads a YOLO model at startup (either a custom trained model
or the COCO-pretrained fallback) and provides inference capabilities.

Usage in a TPV:
  1. Customer places a product on the counter.
  2. Staff takes a photo (webcam or phone camera).
  3. The image is sent to /api/scan.
  4. This service detects objects and resolves them to product IDs.
  5. The frontend adds the detected products to the cart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.schemas.products import DetectedProduct

logger = logging.getLogger(__name__)


# ── Image preprocessing helpers ──────────────────────────────


def _preprocess_for_detection(image: Image.Image) -> np.ndarray:
    """Apply CLAHE contrast enhancement + sharpening for better YOLO detections."""
    img_array = np.array(image.convert("RGB"))

    # Convert to LAB for contrast enhancement on L channel (preserves color)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # Apply mild sharpening
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced_rgb, -1, sharpen_kernel)

    return sharpened


def _resize_if_needed(
    img_array: np.ndarray,
    max_dim: int = 1280,
) -> np.ndarray:
    """Downscale large images for faster inference while preserving accuracy."""
    h, w = img_array.shape[:2]
    if max(h, w) <= max_dim:
        return img_array

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized image from %dx%d to %dx%d", w, h, new_w, new_h)
    return resized


# ── Lazy-loaded singleton ──────────────────────────────────


class VisionModelService:
    """Singleton that holds the YOLO model in memory."""

    # COCO classes relevant to a TPV/shop context (filter out person, car, etc.)
    COCO_PRODUCT_CLASSES: set[int] = {
        24, 25, 26, 27, 28,       # bags & luggage
        31, 32, 33, 34, 35, 36, 37, 38,  # sports equipment
        39, 40, 41, 42, 43, 44, 45,  # bottle, glass, cup, cutlery, bowl
        46, 47, 48, 49, 50, 51,    # fruits & vegetables
        52, 53, 54, 55,            # prepared food
        56, 57, 58, 60,            # furniture
        62, 63, 64, 65, 66, 67,    # electronics
        68, 69, 70, 72,            # appliances
        73, 74, 75, 76, 77, 78, 79, # misc products
    }

    # Spanish names for COCO classes in a TPV/supermarket context
    COCO_SPANISH_NAMES: dict[int, str] = {
        39: "Botella", 40: "Copa", 41: "Taza",
        42: "Tenedor", 43: "Cuchillo", 44: "Cuchara", 45: "Bol",
        46: "Plátano", 47: "Manzana", 48: "Bocadillo",
        49: "Naranja", 50: "Brócoli", 51: "Zanahoria",
        52: "Perrito Caliente", 53: "Pizza", 54: "Dónut", 55: "Pastel",
        56: "Silla", 57: "Sofá", 58: "Planta", 60: "Mesa",
        62: "Televisor", 63: "Portátil", 64: "Ratón",
        65: "Teléfono", 66: "Tablet", 67: "Monitor",
        68: "Microondas", 69: "Horno", 70: "Tostadora", 72: "Nevera",
        73: "Libro", 74: "Reloj", 75: "Jarrón",
        76: "Tijeras", 77: "Peluche", 78: "Secador", 79: "Cepillo",
    }

    def __init__(self) -> None:
        self._model: Any = None          # YOLO model instance
        self._model_path: str = ""
        self._loaded: bool = False
        self._device: str = settings.DEVICE
        self._class_names: dict[int, str] = {}
        self._model_is_coco: bool = False

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    # ── Load / Unload ─────────────────────────────────────────────

    def load(self) -> None:
        """Load the YOLO model from the configured path (or COCO fallback).

        Call this once during application startup.
        """
        from ultralytics import YOLO  # slow import – defer

        # 1. Try custom trained model
        custom_path = Path(settings.MODEL_PATH)
        if custom_path.exists():
            logger.info("Loading custom YOLO model from %s", custom_path)
            self._model = YOLO(str(custom_path))
            self._model_path = str(custom_path)
            self._loaded = True
            self._model_is_coco = False
        elif settings.USE_COCO_FALLBACK:
            # 2. Fallback to COCO-pretrained
            coco_name = settings.COCO_MODEL
            logger.info("Custom model not found. Loading COCO fallback: %s", coco_name)
            self._model = YOLO(coco_name)
            self._model_path = coco_name
            self._loaded = True
            self._model_is_coco = True
        else:
            logger.warning("No YOLO model found at %s and COCO fallback disabled.", custom_path)
            self._loaded = False
            return

        # Store class names for later reference
        if hasattr(self._model, "names"):
            self._class_names = self._model.names
        logger.info(
            "YOLO model loaded with %d classes (COCO fallback: %s).",
            len(self._class_names),
            self._model_is_coco,
        )

    def unload(self) -> None:
        """Release model memory."""
        self._model = None
        self._loaded = False
        self._class_names = {}
        logger.info("YOLO model unloaded.")

    # ── Inference ──────────────────────────────────────────────

    def detect(
        self,
        image: Image.Image,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> list[DetectedProduct]:
        """Run YOLO inference on a PIL image and return detected products.

        Args:
            image: PIL Image in RGB mode.
            conf_threshold: Override the default confidence threshold.
            iou_threshold: Override the default IoU NMS threshold.

        Returns:
            A list of DetectedProduct objects (may be empty).
        """
        if not self._loaded or self._model is None:
            logger.warning("YOLO model not loaded – returning empty detection.")
            return []

        conf = conf_threshold or settings.CONFIDENCE_THRESHOLD
        iou = iou_threshold or settings.IOU_THRESHOLD

        # 1. Preprocess: enhance contrast and sharpen
        img_array = _preprocess_for_detection(image)

        # 2. Downscale if too large (improves speed)
        img_array = _resize_if_needed(img_array, max_dim=1280)

        # 3. Run inference
        results = self._model.predict(
            img_array,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
            half=self._device != "cpu",
        )

        return self._parse_results(results, img_array.shape[1], img_array.shape[0])

    # ── Result parsing ─────────────────────────────────────────────

    def _parse_results(
        self,
        results: Any,
        img_width: int,
        img_height: int,
    ) -> list[DetectedProduct]:
        """Convert Ultralytics Results into DetectedProduct list.

        When using COCO fallback, filters classes to product-relevant ones
        and resolves COCO class IDs to product barcodes when a mapping
        is configured via the COCO_PRODUCT_MAP env var.
        """
        detected: list[DetectedProduct] = []
        mapping = settings.COCO_PRODUCT_MAP

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                # When using COCO fallback, filter to product-relevant classes
                if self._model_is_coco and cls_id not in self.COCO_PRODUCT_CLASSES:
                    continue

                # Normalised bounding box
                x1, y1, x2, y2 = box.xyxyn[0].tolist()

                # Use Spanish name when using COCO fallback
                if self._model_is_coco:
                    class_name = self.COCO_SPANISH_NAMES.get(cls_id, self._class_names.get(cls_id, f"class_{cls_id}"))
                else:
                    class_name = self._class_names.get(cls_id, f"class_{cls_id}")

                # Resolve barcode from mapping if configured
                barcode = None
                product_id = None
                if mapping and cls_id in mapping:
                    barcode = str(mapping[cls_id])
                    product_id = barcode

                detected.append(
                    DetectedProduct(
                        product_id=product_id,
                        barcode=barcode,
                        name=class_name,
                        confidence=confidence,
                        detection_method="yolo",
                        quantity=1,
                        bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        coco_class=class_name,
                        coco_class_id=cls_id,
                    )
                )

        # Deduplicate by class (keep highest confidence for each class)
        return self._deduplicate_by_class(detected)

    def _deduplicate_by_class(
        self, items: list[DetectedProduct]
    ) -> list[DetectedProduct]:
        """Keep only the highest-confidence detection per COCO class.

        Prevents multiple overlapping detections of the same product.
        """
        best_per_class: dict[int, DetectedProduct] = {}

        for item in items:
            if item.coco_class_id is None:
                continue
            existing = best_per_class.get(item.coco_class_id)
            if existing is None or item.confidence > existing.confidence:
                best_per_class[item.coco_class_id] = item

        # Also include items without coco_class_id (custom models)
        no_class = [i for i in items if i.coco_class_id is None]

        return list(best_per_class.values()) + no_class


# Module-level singleton - import this from routes
model_service = VisionModelService()