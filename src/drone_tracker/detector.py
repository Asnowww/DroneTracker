from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return max(1.0, x2 - x1), max(1.0, y2 - y1)


class YoloDroneDetector:
    def __init__(
        self,
        weights: str | Path,
        device: str | int = 0,
        imgsz: int = 960,
        conf: float = 0.3,
        iou: float = 0.5,
        half: bool = True,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        results: list[Any] = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        best: Detection | None = None
        for box in results[0].boxes:
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls = int(box.cls[0].item()) if box.cls is not None else 0
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            det = Detection((xyxy[0], xyxy[1], xyxy[2], xyxy[3]), conf, cls)
            if best is None or det.confidence > best.confidence:
                best = det
        return best

