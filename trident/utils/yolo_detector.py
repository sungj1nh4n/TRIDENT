"""Ultralytics YOLO singleton for person detection.

Used by:
  - Ingest pipeline (worker): batch person-crop extraction per scene frame
  - API (/api/search-by-person): single-image person crop from uploaded reference
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from srpost.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in image pixel coords
    confidence: float


class YOLODetector:
    _instance: "YOLODetector | None" = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu") -> None:
        from ultralytics import YOLO
        weights = settings.yolo_weights_path
        if not weights or not Path(weights).exists():
            weights = "yolo26x.pt"
            logger.warning("yolo_weights_path not set or missing — falling back to %s", weights)
        logger.info("Loading YOLO %s on %s", weights, device)
        self._model = YOLO(weights)
        self._device = device

    @classmethod
    def get(cls, device: str = "cpu") -> "YOLODetector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(device=device)
        return cls._instance

    def detect_persons(
        self,
        image: Image.Image,
        conf: float = 0.4,
        iou: float = 0.5,
        max_persons: int = 10,
    ) -> list[Detection]:
        """Run YOLO and return person (class 0) detections sorted by confidence desc."""
        arr = np.array(image.convert("RGB"))
        preds = self._model(arr, verbose=False, conf=conf, iou=iou, classes=[0],
                            device=self._device)
        if not preds:
            return []
        r = preds[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)
        dets = [
            Detection(bbox=(int(b[0]), int(b[1]), int(b[2]), int(b[3])), confidence=float(c))
            for b, c in zip(xyxy, confs)
        ]
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets[:max_persons]

    def crop_persons(
        self,
        image: Image.Image,
        conf: float = 0.4,
        iou: float = 0.5,
        max_persons: int = 10,
    ) -> list[tuple[Image.Image, Detection]]:
        """Return list of (cropped PIL image, detection meta)."""
        dets = self.detect_persons(image, conf=conf, iou=iou, max_persons=max_persons)
        rgb = image.convert("RGB")
        out: list[tuple[Image.Image, Detection]] = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(rgb.width, x2); y2 = min(rgb.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((rgb.crop((x1, y1, x2, y2)), d))
        return out
