"""
VenusX AI Scanner API – FastAPI application.

Provides image recognition (YOLOv8) and barcode scanning for the
VenusX Point-of-Sale (TPV) system.

Start the dev server:
    uvicorn app.main:app --reload --port 5000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.vision_model import model_service

# ── Logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (load model on startup) ───────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLO model when the app starts and clean up on shutdown."""
    logger.info("Loading YOLO vision model ...")
    model_service.load()
    if model_service.is_loaded:
        logger.info("YOLO model loaded successfully on '%s'.", model_service.device)
    else:
        logger.warning("YOLO model could not be loaded - endpoints will return empty results.")
    yield
    model_service.unload()
    logger.info("YOLO model unloaded.")


# ── App instance ───────────────────────────────────────────

app = FastAPI(
    title="VenusX AI Scanner API",
    description="AI-powered product detection and barcode scanning for VenusX TPV.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────

app.include_router(router, prefix="/api")


# ── Root health ───────────────────────────────────────────

@app.get("/")
async def read_root():
    return {
        "service": "VenusX AI Scanner",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_service.is_loaded,
    }