"""
Download and prepare YOLOv8 models for the VenusX AI Scanner.

Usage:
    python scripts/download_model.py              # Download COCO fallback
    python scripts/download_model.py --export onnx  # Also export to ONNX
    python scripts/download_model.py --custom path/to/best.pt  # Use custom model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "ml_models"
COCO_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def download_coco(model_name: str = "yolov8n.pt") -> Path:
    """Download a COCO-pretrained YOLOv8 model via Ultralytics.

    The model is cached in ~/.cache/ultralytics/ by Ultralytics itself,
    so this is essentially a no-op if it has been downloaded before.
    We copy it to ml_models/ for convenience.
    """
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLO model: %s (will download if not cached) ...", model_name)
    model = YOLO(model_name)
    dest = MODELS_DIR / model_name
    if not dest.exists():
        import shutil
        shutil.copy(model_name, dest)
        logger.info("Copied model to %s", dest)
    else:
        logger.info("Model already exists at %s", dest)
    return dest


def export_model(model_path: Path, fmt: str = "onnx") -> Path:
    """Export a YOLO model to ONNX or OpenVINO format for faster CPU inference."""
    from ultralytics import YOLO

    logger.info("Exporting %s to %s format ...", model_path, fmt.upper())
    model = YOLO(str(model_path))
    output = model.export(format=fmt)
    logger.info("Exported model: %s", output)
    return Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & prepare YOLOv8 models")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        choices=COCO_MODELS,
        help="COCO model to download (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--export",
        choices=["onnx", "openvino"],
        default=None,
        help="Export the model to an optimised format for CPU inference",
    )
    parser.add_argument(
        "--custom",
        type=str,
        default=None,
        help="Path to a custom trained .pt model (skips COCO download)",
    )
    args = parser.parse_args()

    if args.custom:
        model_path = Path(args.custom)
        if not model_path.exists():
            logger.error("Custom model not found: %s", model_path)
            sys.exit(1)
        logger.info("Using custom model: %s", model_path)
        # Copy to our models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = MODELS_DIR / "best.pt"
        shutil.copy(str(model_path), str(dest))
        logger.info("Custom model copied to %s", dest)
        model_path = dest
    else:
        model_path = download_coco(args.model)

    if args.export:
        export_path = export_model(model_path, args.export)
        logger.info("Exported model ready at: %s", export_path)

    logger.info("Done! Set YOLO_MODEL_PATH=%s to use this model.", model_path)


if __name__ == "__main__":
    main()
