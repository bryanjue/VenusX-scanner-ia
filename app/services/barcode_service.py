"""
Barcode detection service using pyzbar (zbar wrapper).

Multi-stage preprocessing pipeline to handle challenging real-world images:
  - Blurry / out-of-focus photos
  - Low contrast / poor lighting
  - Angled or skewed barcodes
  - Small barcodes in high-res images

For a TPV, barcode scanning from a camera photo is extremely useful
because most products already have printed barcodes.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.core.config import settings
from app.schemas.products import DetectedBarcode

logger = logging.getLogger(__name__)


def decode_barcodes(image: Image.Image) -> list[DetectedBarcode]:
    """Detect and decode all barcodes in a PIL image using multi-stage preprocessing.

    Tries increasingly aggressive preprocessing techniques to handle
    blurry, low-contrast, or angled barcodes commonly found in
    real-world TPV photos.

    Args:
        image: PIL Image (RGB or grayscale).

    Returns:
        A list of DetectedBarcode objects (may be empty).
    """
    if not settings.BARCODE_ENABLED:
        return []

    try:
        from pyzbar import pyzbar as pyzbar_lib
    except ImportError:
        logger.warning("pyzbar is not installed - barcode detection disabled.")
        return []

    # Convert PIL -> OpenCV BGR (for preprocessing)
    img_rgb = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = img_gray.shape

    # ---------- Stage 1: Try on original grayscale ----------
    barcodes_raw = pyzbar_lib.decode(img_gray)
    if barcodes_raw:
        return _build_results(barcodes_raw, height, width)

    # ---------- Stage 2: Try on original RGB (pyzbar can decode RGB too) ----------
    barcodes_raw = pyzbar_lib.decode(img_rgb)
    if barcodes_raw:
        return _build_results(barcodes_raw, height, width)

    # ---------- Stage 3: Contrast enhancement ----------
    enhanced = _enhance_contrast(img_gray)
    barcodes_raw = pyzbar_lib.decode(enhanced)
    if barcodes_raw:
        return _build_results(barcodes_raw, height, width)

    # ---------- Stage 4: Adaptive thresholding ----------
    try:
        thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        barcodes_raw = pyzbar_lib.decode(thresh)
        if barcodes_raw:
            return _build_results(barcodes_raw, height, width)
    except Exception:
        pass

    # ---------- Stage 5: Morphological operations (close gaps) ----------
    morphed = _morphological_preprocess(img_gray)
    barcodes_raw = pyzbar_lib.decode(morphed)
    if barcodes_raw:
        return _build_results(barcodes_raw, height, width)

    # ---------- Stage 6: Sharpen + contrast ----------
    sharpened = _sharpen_image(img_gray)
    barcodes_raw = pyzbar_lib.decode(sharpened)
    if barcodes_raw:
        return _build_results(barcodes_raw, height, width)

    # ---------- Stage 7: Upscale small images to improve small barcode detection ----------
    if width < 800 or height < 600:
        upscaled = cv2.resize(img_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        barcodes_raw = pyzbar_lib.decode(upscaled)
        if barcodes_raw:
            h2, w2 = upscaled.shape
            return _build_results(barcodes_raw, h2, w2)

    # ---------- Stage 8: Try bilateral filter (preserves edges while denoising) ----------
    try:
        denoised = cv2.bilateralFilter(img_gray, 9, 75, 75)
        barcodes_raw = pyzbar_lib.decode(denoised)
        if barcodes_raw:
            return _build_results(barcodes_raw, height, width)
    except Exception:
        pass

    # ---------- Stage 9: Histogram equalization ----------
    try:
        equalized = cv2.equalizeHist(img_gray)
        barcodes_raw = pyzbar_lib.decode(equalized)
        if barcodes_raw:
            return _build_results(barcodes_raw, height, width)
    except Exception:
        pass

    # ---------- Stage 10: Last resort - try sharpened PIL image ----------
    try:
        pil_sharp = image.convert("L").filter(ImageFilter.SHARPEN)
        img_np = np.array(pil_sharp)
        barcodes_raw = pyzbar_lib.decode(img_np)
        if barcodes_raw:
            return _build_results(barcodes_raw, height, width)
    except Exception:
        pass

    return []


# ── Preprocessing helpers ──────────────────────────────────


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _morphological_preprocess(gray: np.ndarray) -> np.ndarray:
    """Apply morphological black-hat and tophat to highlight barcode regions."""
    # Black hat: detects dark lines on light background (typical barcode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Top hat: detects light lines on dark background
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # Combine
    combined = cv2.add(blackhat, tophat)
    # Threshold to binary
    _, binary = cv2.threshold(combined, 10, 255, cv2.THRESH_BINARY)
    return binary


def _sharpen_image(gray: np.ndarray) -> np.ndarray:
    """Apply sharpening kernel to enhance barcode edges."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, kernel)
    return sharpened


# ── Result building ────────────────────────────────────────


def _build_results(
    barcodes_raw: list,
    height: int,
    width: int,
) -> list[DetectedBarcode]:
    """Convert raw pyzbar results to DetectedBarcode list with normalised coordinates."""
    results: list[DetectedBarcode] = []

    for barcode in barcodes_raw:
        data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        quality = _estimate_confidence(data, barcode_type)

        # Normalise polygon points to [0, 1]
        polygon = [
            {"x": p.x / width, "y": p.y / height}
            for p in barcode.polygon
        ]

        results.append(
            DetectedBarcode(
                data=data,
                type=barcode_type,
                confidence=quality,
                polygon=polygon,
            )
        )

    return results


def _estimate_confidence(data: str, barcode_type: str) -> float:
    """Return a heuristic confidence score (0-1) for a decoded barcode.

    Uses checksum validation for EAN/UPC codes and length heuristics for others.
    """
    # EAN-13 / UPC-A / EAN-8: validate with check digit
    if barcode_type in ("EAN13", "EAN-13") and len(data) == 13 and data.isdigit():
        if _validate_ean13(data):
            return 0.98
        return 0.70  # Invalid checksum

    if barcode_type in ("UPCA", "UPC-A") and len(data) == 12 and data.isdigit():
        return 0.95

    if barcode_type in ("EAN8", "EAN-8") and len(data) == 8 and data.isdigit():
        if _validate_ean8(data):
            return 0.95
        return 0.70

    # ISBN: 10 or 13 digits
    if barcode_type == "ISBN10" or barcode_type == "ISBN13":
        return 0.95

    # CODE128 / CODE39 / ITF: typically longer codes
    if len(data) >= 8:
        return 0.88
    if len(data) >= 4:
        return 0.80

    # QR / DataMatrix: typically shorter but still reliable
    if barcode_type in ("QRCODE", "DATAMATRIX"):
        return 0.92

    # Short codes - less reliable
    return 0.75


def _validate_ean13(code: str) -> bool:
    """Validate EAN-13 checksum digit."""
    if len(code) != 13 or not code.isdigit():
        return False
    total = 0
    for i, digit in enumerate(code[:12]):
        weight = 3 if i % 2 == 1 else 1
        total += int(digit) * weight
    expected = (10 - (total % 10)) % 10
    return expected == int(code[12])


def _validate_ean8(code: str) -> bool:
    """Validate EAN-8 checksum digit."""
    if len(code) != 8 or not code.isdigit():
        return False
    total = 0
    for i, digit in enumerate(code[:7]):
        weight = 3 if i % 2 == 1 else 1
        total += int(digit) * weight
    expected = (10 - (total % 10)) % 10
    return expected == int(code[7])
