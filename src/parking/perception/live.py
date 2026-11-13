"""CCTV 프레임 위 주차면/차량 인식."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from parking.app.paths import ROOT

MODELS_DIR = ROOT / "models"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {
    "occupied": (111, 71, 239),
    "empty": (182, 196, 46),
    "unknown": (94, 201, 244),
    "car": (232, 176, 74),
}


def list_models() -> list[dict]:
    """관제에서 고를 수 있는 모델 목록."""
    items = [
        {
            "label": "YOLOv8n 차량",
            "kind": "coco",
            "path": "yolov8n.pt",
        }
    ]
    if MODELS_DIR.is_dir():
        for path in sorted(MODELS_DIR.rglob("*.pt")):
            if "_runs" in path.parts:
                continue
            rel = path.relative_to(MODELS_DIR).as_posix()
            kind = "cnn" if "cnn" in rel.lower() else "occupancy"
            items.append({"label": rel, "kind": kind, "path": str(path)})
    return items


class LiveDetector:
    def __init__(self):
        self._model = None
        self._vehicle = None
        self._ready = False
        self.kind = "coco"
        self.path = "yolov8n.pt"
        self.label = "YOLOv8n 차량"
        self._lock = threading.Lock()

    def use(self, kind: str, path: str, label: str):
        with self._lock:
            self.kind = kind
            self.path = path
            self.label = label
            self._model = None
            self._vehicle = None
            self._ready = False

    def ensure(self) -> str:
        if self._ready:
            return self.label
        from ultralytics import YOLO

        if self.kind == "cnn":
            self._vehicle = YOLO("yolov8n.pt")
            self.label = "YOLOv8n 차량"
        else:
            self._model = YOLO(self.path)
        self._ready = True
        return self.label

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        with self._lock:
            self.ensure()
            if self.kind == "cnn":
                return self._vehicle_boxes(frame_bgr, self._vehicle)
            if self._model is None:
                return []
            return self._yolo_boxes(frame_bgr, self._model, self.kind)

    def _vehicle_boxes(self, frame_bgr: np.ndarray, model) -> list[dict]:
        return self._yolo_boxes(frame_bgr, model, "coco")

    def _yolo_boxes(self, frame_bgr: np.ndarray, model, kind: str) -> list[dict]:
        result = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.35)[0]
        boxes = []
        if result.boxes is None:
            return boxes
        names = result.names or {}
        mask_points = result.masks.xy if result.masks is not None else None
        for index, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if kind == "occupancy":
                label = "occupied" if cls_id == 1 else "empty"
            else:
                label = VEHICLE_CLASSES.get(cls_id)
                if not label:
                    continue
            points = None
            if mask_points is not None and index < len(mask_points):
                points = mask_points[index].tolist()
            boxes.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "points": points,
                    "label": label,
                    "conf": float(box.conf[0]),
                    "name": names.get(cls_id, label),
                }
            )
        return boxes


def draw_detections(frame_bgr: np.ndarray, boxes: list[dict]) -> np.ndarray:
    canvas = frame_bgr.copy()
    for box in boxes:
        label = box["label"]
        color = COLORS.get(label, COLORS["unknown"])
        points = box.get("points")
        if points is not None and len(points) >= 3:
            polygon = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, color, 2)
            x1, y1 = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        else:
            x1, y1, x2, y2 = box["xyxy"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {box['conf']:.2f}"
        cv2.putText(canvas, text, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def summarize(boxes: list[dict]) -> str:
    if not boxes:
        return "인식 없음"
    occupied = sum(1 for box in boxes if box["label"] == "occupied")
    empty = sum(1 for box in boxes if box["label"] == "empty")
    vehicles = sum(1 for box in boxes if box["label"] in VEHICLE_CLASSES.values())
    parts = []
    if vehicles:
        parts.append(f"차량 {vehicles}")
    if occupied or empty:
        parts.append(f"점유 {occupied} · 빈칸 {empty}")
    return " · ".join(parts) if parts else "인식 없음"
