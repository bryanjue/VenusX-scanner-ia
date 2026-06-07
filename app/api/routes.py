"""
API routes for the VenusX AI Scanner.

Endpoints:
  POST /api/scan          – Full scan: YOLO detection + barcode decoding.
  POST /api/scan/barcode  – Barcode-only scan (faster, no YOLO).
  GET  /api/health        – Model & service health check.
"""

from __future__ import annotations

import io
import logging
import time
from collections import OrderedDict
from typing import Optional

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.core.config import settings
from app.schemas.products import (
    DetectedProduct,
    ModelInfo,
    ScanResponse,
)
from app.services.barcode_service import decode_barcodes
from app.services.vision_model import model_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Product Lookup Cache (LRU) ──────────────────────────────

class _MISS:
    """Sentinel value to distinguish 'not in cache' from 'cached as None'."""
    pass

MISS = _MISS()


class ProductCache:
    """Simple in-memory LRU cache for product lookups."""

    def __init__(self, maxsize: int = 200, ttl: int = 300):
        self._cache: OrderedDict[str, tuple[float, Optional[dict]]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str):
        """Return the cached value, or MISS if not in cache / expired."""
        if key not in self._cache:
            return MISS
        ts, value = self._cache[key]
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return MISS
        self._cache.move_to_end(key)
        return value  # Could be None (cached miss) or dict (cached hit)

    def set(self, key: str, value: Optional[dict]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), value)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._cache)


_product_cache = ProductCache()


# ── Helpers ──────────────────────────────────────────────────

async def _lookup_product_by_barcode(barcode: str) -> Optional[dict]:
    """Query the VenusX API backend for product details by barcode.

    Uses an in-memory LRU cache to avoid repeated API calls.
    Correctly caches both found and not-found results.

    Returns the product JSON, or None if not found / API unreachable.
    """
    cached = _product_cache.get(barcode)
    if cached is not MISS:
        return cached  # Could be None (cached not-found) or dict (cached found)

    product = await _query_venusx_by_barcode(barcode)
    _product_cache.set(barcode, product)  # Cache even None to avoid repeated misses
    return product


async def _query_venusx_by_barcode(barcode: str) -> Optional[dict]:
    """Query VenusX API by barcode. Returns product dict or None."""
    try:
        async with httpx.AsyncClient(timeout=settings.VENUSX_API_TIMEOUT) as client:
            resp = await client.get(
                f"{settings.VENUSX_API_URL}/search",
                params={"Bar_code": barcode},
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                logger.info("Barcode %s not found in VenusX API (404)", barcode)
                return None
            logger.warning(
                "VenusX API returned %d for barcode %s", resp.status_code, barcode
            )
    except httpx.RequestError as exc:
        logger.warning("VenusX API unreachable for barcode %s: %s", barcode, exc)
    except Exception as exc:
        logger.error("Unexpected error querying VenusX API: %s", exc)
    return None


async def _lookup_products_bulk(
    barcodes: list[str],
) -> dict[str, Optional[dict]]:
    """Look up multiple barcodes in the VenusX API."""
    result: dict[str, Optional[dict]] = {}
    for bc in barcodes:
        product = await _lookup_product_by_barcode(bc)
        result[bc] = product
    return result


async def _query_venusx_by_name(name: str) -> Optional[dict]:
    """Query VenusX API by product name. Returns first matching product or None."""
    try:
        async with httpx.AsyncClient(timeout=settings.VENUSX_API_TIMEOUT) as client:
            resp = await client.get(
                f"{settings.VENUSX_API_URL}/search/name",
                params={"q": name},
            )
            if resp.status_code == 200:
                products = resp.json()
                if isinstance(products, list) and len(products) > 0:
                    return products[0]
                return None
            if resp.status_code == 404:
                return None
            logger.warning(
                "VenusX API returned %d for name search '%s'", resp.status_code, name
            )
    except httpx.RequestError as exc:
        logger.debug("VenusX API unreachable for name '%s': %s", name, exc)
    except Exception as exc:
        logger.error("Unexpected error in name lookup '%s': %s", name, exc)
    return None


async def _enrich_yolo_with_name_lookup(
    yolo_items: list[DetectedProduct],
) -> list[DetectedProduct]:
    """Try to enrich YOLO detections that have no barcode by looking up their
    Spanish name in the VenusX API product database."""
    enriched: list[DetectedProduct] = []
    for item in yolo_items:
        if item.barcode or not item.name or item.name == "Unknown":
            enriched.append(item)
            continue
        product = await _query_venusx_by_name(item.name)
        if product:
            enriched.append(
                DetectedProduct(
                    product_id=str(product.get("id", "")),
                    barcode=str(product.get("bar_code", "")),
                    name=product.get("name", item.name),
                    price=float(product.get("price", 0)),
                    confidence=item.confidence,
                    detection_method=item.detection_method,
                    quantity=item.quantity,
                    bbox=item.bbox,
                    coco_class=item.coco_class,
                    coco_class_id=item.coco_class_id,
                )
            )
        else:
            enriched.append(item)
    return enriched


async def _pil_from_upload(file: UploadFile) -> Image.Image:
    """Read an uploaded image file and return a PIL Image."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")


# ── Endpoints ────────────────────────────────────────────────

@router.post("/scan", response_model=ScanResponse)
async def scan_image(file: UploadFile = File(...)):
    """Full scan: runs YOLOv8 object detection + barcode decoding.

    Returns all detected products (from YOLO) and barcodes (from pyzbar).
    When a barcode is found, it also looks up product details via the
    VenusX API and merges them into the detected_items list.
    """
    start = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image = await _pil_from_upload(file)

    # 1. Run YOLO detection
    yolo_items = model_service.detect(image)

    # 2. Decode barcodes
    barcodes = decode_barcodes(image)

    # 3. Enrich barcodes with product data from VenusX API
    barcode_items: list[DetectedProduct] = []
    if barcodes:
        barcode_values = [b.data for b in barcodes if b.data]
        product_map = await _lookup_products_bulk(barcode_values)

        for bc in barcodes:
            product_info = product_map.get(bc.data)
            if product_info:
                name = product_info.get("name", "Unknown")
                pid = str(product_info.get("id", ""))
                price = float(product_info.get("price", 0))
                barcode_items.append(
                    DetectedProduct(
                        product_id=pid,
                        barcode=bc.data,
                        name=name,
                        price=price,
                        confidence=bc.confidence,
                        detection_method="barcode",
                        quantity=1,
                    )
                )
            else:
                barcode_items.append(
                    DetectedProduct(
                        barcode=bc.data,
                        name=f"Product {bc.data}",
                        confidence=bc.confidence,
                        detection_method="barcode",
                        quantity=1,
                    )
                )

    # 4. Try to enrich YOLO-only detections (no barcode) by name via VenusX API
    enriched_yolo = await _enrich_yolo_with_name_lookup(yolo_items)

    # 5. Merge results: prefer barcode items over YOLO for the same barcode
    all_items = _merge_detections(barcode_items, enriched_yolo)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ScanResponse(
        success=True,
        message=f"Detected {len(all_items)} product(s) and {len(barcodes)} barcode(s).",
        detected_items=all_items,
        detected_barcodes=barcodes,
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.post("/scan/barcode", response_model=ScanResponse)
async def scan_barcode(file: UploadFile = File(...)):
    """Barcode-only scan: faster endpoint that skips YOLO inference.

    Useful when the user only needs barcode detection (e.g. scanning
    a shelf label or a product barcode up close).
    """
    start = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image = await _pil_from_upload(file)
    barcodes = decode_barcodes(image)

    items: list[DetectedProduct] = []
    if barcodes:
        barcode_values = [b.data for b in barcodes if b.data]
        product_map = await _lookup_products_bulk(barcode_values)

        for bc in barcodes:
            product_info = product_map.get(bc.data)
            if product_info:
                name = product_info.get("name", f"Product {bc.data}")
                pid = str(product_info.get("id", ""))
                price = float(product_info.get("price", 0))
                items.append(
                    DetectedProduct(
                        product_id=pid,
                        barcode=bc.data,
                        name=name,
                        price=price,
                        confidence=bc.confidence,
                        detection_method="barcode",
                        quantity=1,
                    )
                )
            else:
                items.append(
                    DetectedProduct(
                        barcode=bc.data,
                        name=f"Product {bc.data}",
                        confidence=bc.confidence,
                        detection_method="barcode",
                        quantity=1,
                    )
                )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ScanResponse(
        success=True,
        message=f"Detected {len(barcodes)} barcode(s).",
        detected_items=items,
        detected_barcodes=barcodes,
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.get("/health", response_model=ModelInfo)
async def health_check():
    """Check if the AI model is loaded and the VenusX API is reachable."""
    venusx_ok = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(f"{settings.VENUSX_API_URL}/")
            venusx_ok = resp.status_code < 500
    except Exception:
        pass

    classes = None
    if model_service.is_loaded and model_service.class_names:
        classes = [
            {"id": cid, "name": cname}
            for cid, cname in model_service.class_names.items()
        ]

    return ModelInfo(
        model_loaded=model_service.is_loaded,
        model_path=model_service.model_path,
        device=model_service.device,
        barcode_enabled=settings.BARCODE_ENABLED,
        venusx_api_connected=venusx_ok,
        classes=classes,
    )


# ── Internal ──────────────────────────────────────────────────

def _merge_detections(
    barcode_items: list[DetectedProduct],
    yolo_items: list[DetectedProduct],
) -> list[DetectedProduct]:
    """Merge barcode-based and YOLO-based detections, deduplicating by barcode.

    Barcode detections are more reliable, so they take precedence when the
    same barcode appears in both lists.
    """
    seen_barcodes: set[str] = set()

    merged: list[DetectedProduct] = list(barcode_items)
    for item in barcode_items:
        if item.barcode:
            seen_barcodes.add(item.barcode)

    for yolo in yolo_items:
        if yolo.barcode and yolo.barcode in seen_barcodes:
            continue
        if yolo.barcode:
            seen_barcodes.add(yolo.barcode)
        merged.append(yolo)

    return merged