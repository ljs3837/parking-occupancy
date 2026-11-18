"""CCTV 프레임 위 주차면/차량 인식."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from parking.app.paths import ROOT
from parking.common.device import yolo_device

MODELS_DIR = ROOT / "models"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_LABELS = set(VEHICLE_CLASSES.values())
SPACE_LABELS = {"parking_space", "empty", "occupied", "vacant", "not occupied"}
# OpenCV BGR
COLORS = {
    "parking_space": (255, 128, 0),  # 파랑
    "empty": (255, 128, 0),
    "occupied": (255, 128, 0),
    "vacant": (255, 128, 0),
    "unknown": (255, 128, 0),
    "car": (0, 220, 0),  # 초록
    "motorcycle": (0, 220, 0),
    "bus": (0, 220, 0),
    "truck": (0, 220, 0),
}
VEHICLE_WEIGHTS = "yolov8n.pt"


def list_models() -> list[dict]:
    """관제에서 고를 수 있는 모델 목록."""
    items = [
        {
            "label": "YOLOv8n 차량만",
            "kind": "coco",
            "path": VEHICLE_WEIGHTS,
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


def _draw_color(label: str) -> tuple[int, int, int]:
    if label in VEHICLE_LABELS:
        return COLORS["car"]
    if label in SPACE_LABELS:
        return COLORS["parking_space"]
    return COLORS.get(label, COLORS["parking_space"])


def _display_label(label: str) -> str:
    if label in VEHICLE_LABELS:
        return "vehicle" if label == "car" else label
    if label in {"empty", "occupied", "vacant", "not occupied", "parking_space"}:
        return "parking"
    return label


class LiveDetector:
    def __init__(self):
        self._space_model = None
        self._vehicle_model = None
        self._ready = False
        self.kind = "coco"
        self.path = VEHICLE_WEIGHTS
        self.label = "YOLOv8n 차량만"
        self._lock = threading.Lock()

    def use(self, kind: str, path: str, label: str):
        with self._lock:
            self.kind = kind
            self.path = path
            self.label = label
            self._space_model = None
            self._vehicle_model = None
            self._ready = False

    def ensure(self) -> str:
        if self._ready:
            return self.label
        from ultralytics import YOLO

        self._vehicle_model = YOLO(VEHICLE_WEIGHTS)
        if self.kind == "occupancy":
            self._space_model = YOLO(self.path)
        else:
            self._space_model = None
        self._ready = True
        return self.label

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        with self._lock:
            self.ensure()
            boxes: list[dict] = []
            if self._vehicle_model is not None:
                boxes.extend(self._yolo_boxes(frame_bgr, self._vehicle_model, "coco"))
            if self.kind == "occupancy" and self._space_model is not None:
                boxes.extend(self._yolo_boxes(frame_bgr, self._space_model, "occupancy"))
            return boxes

    def _yolo_boxes(self, frame_bgr: np.ndarray, model, kind: str) -> list[dict]:
        result = model.predict(
            frame_bgr,
            verbose=False,
            imgsz=640,
            conf=0.35,
            device=yolo_device(),
        )[0]
        boxes = []
        if result.boxes is None:
            return boxes
        names = result.names or {}
        mask_points = result.masks.xy if result.masks is not None else None
        for index, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if kind == "occupancy":
                name = str(names.get(cls_id, "")).lower()
                if name in {"occupied", "empty", "vacant"}:
                    label = name
                elif name == "parking_space" or len(names) <= 1:
                    label = "parking_space"
                else:
                    label = "parking_space"
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
        color = _draw_color(label)
        points = box.get("points")
        if points is not None and len(points) >= 3:
            polygon = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, color, 2)
            x1, y1 = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        else:
            x1, y1, x2, y2 = box["xyxy"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{_display_label(label)} {box['conf']:.2f}"
        cv2.putText(canvas, text, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def summarize(boxes: list[dict]) -> str:
    if not boxes:
        return "인식 없음"
    vehicles = sum(1 for box in boxes if box["label"] in VEHICLE_LABELS)
    spaces = sum(1 for box in boxes if box["label"] in SPACE_LABELS)
    parts = []
    if vehicles:
        parts.append(f"차량 {vehicles}")
    if spaces:
        parts.append(f"주차칸 {spaces}")
    return " · ".join(parts) if parts else "인식 없음"
