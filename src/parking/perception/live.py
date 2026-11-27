"""CCTV 프레임 위 주차면/차량 인식."""

from __future__ import annotations

import threading
import json
from pathlib import Path

import cv2
import numpy as np

from parking.app.paths import ROOT
from parking.common.device import yolo_device
from parking.perception.lines import straightened_outline
from parking.perception.tiling import tiled_candidates

MODELS_DIR = ROOT / "models"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_LABELS = set(VEHICLE_CLASSES.values())
SPACE_LABELS = {"parking_space", "empty", "occupied", "vacant", "not occupied"}
# OpenCV BGR — 관제 점유: 빈=초록, 점유=빨강, 차량=시안
COLORS = {
    "parking_space": (255, 128, 0),  # 탐색 중(확정 전) 파랑
    "empty": (80, 200, 80),
    "vacant": (80, 200, 80),
    "occupied": (60, 60, 230),
    "unknown": (160, 160, 160),
    "car": (220, 200, 0),
    "motorcycle": (220, 200, 0),
    "bus": (220, 200, 0),
    "truck": (220, 200, 0),
}
VEHICLE_WEIGHTS = "yolov8n.pt"


def list_models() -> list[dict]:
    """관제에서 고를 수 있는 모델 목록."""
    preferred_path = None
    preference = MODELS_DIR / "preferred_parking_model.json"
    if preference.is_file():
        try:
            preferred_path = (ROOT / json.loads(preference.read_text(encoding="utf-8"))["model"]).resolve()
        except (OSError, ValueError, KeyError, TypeError):
            pass
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
            items.append({"label": rel, "kind": kind, "path": str(path),
                          "preferred": path.resolve() == preferred_path})
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
        # 관제 좌측에서 고른 표시 임계값. 학습 설정이 아니라 추론 필터다.
        self._inference = {"conf": 0.375}

    def use(self, kind: str, path: str, label: str):
        with self._lock:
            self.kind = kind
            self.path = path
            self.label = label
            self._space_model = None
            self._vehicle_model = None
            self._ready = False

    def set_inference(self, *, conf: float | None = None) -> None:
        """관제 좌측 임계값을 다음 추론부터 바로 반영한다."""
        with self._lock:
            if conf is not None:
                self._inference["conf"] = min(0.99, max(0.01, float(conf)))

    def inference_options(self) -> dict:
        with self._lock:
            return dict(self._inference)

    def _ensure_vehicle_unlocked(self) -> None:
        if self._vehicle_model is not None:
            self._ready = True
            return
        from ultralytics import YOLO

        self._vehicle_model = YOLO(VEHICLE_WEIGHTS)
        self._ready = True

    def ensure(self) -> str:
        """관제 루프용: 차량 YOLO만 올린다."""
        with self._lock:
            self._ensure_vehicle_unlocked()
            return self.label

    def ensure_space_model(self) -> str:
        """레이아웃 확정용: 주차칸 모델을 필요할 때만 올린다."""
        from ultralytics import YOLO

        with self._lock:
            if self.kind != "occupancy":
                raise RuntimeError("주차칸 모델을 선택한 뒤 레이아웃을 확정하세요")
            if self._space_model is None:
                self._space_model = YOLO(self.path)
            return self.label

    def unload_space_model(self) -> None:
        """레이아웃 확정 후 주차칸 가중치를 내린다."""
        with self._lock:
            self._space_model = None

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        """관제 루프: 차량만 탐지한다. 슬롯은 고정 레이아웃을 쓴다."""
        with self._lock:
            self._ensure_vehicle_unlocked()
            if self._vehicle_model is None:
                return []
            return self._yolo_boxes(
                frame_bgr,
                self._vehicle_model,
                "coco",
                conf=float(self._inference.get("conf", 0.375)),
            )

    def predict_spaces(self, frame_bgr: np.ndarray) -> list[dict]:
        """선택된 주차칸 모델만 추론한다. 차량 모델은 돌리지 않는다."""
        with self._lock:
            if self.kind != "occupancy":
                return []
            from ultralytics import YOLO

            if self._space_model is None:
                self._space_model = YOLO(self.path)
            return self._yolo_boxes(
                frame_bgr,
                self._space_model,
                "occupancy",
                imgsz=640,
                conf=float(self._inference.get("conf", 0.375)),
            )

    def predict_space_candidates(self, frame_bgr, model, *, imgsz=640, tiled=False):
        predict = lambda image: self._yolo_boxes(image, model, "occupancy", imgsz=imgsz)
        return tiled_candidates(frame_bgr, predict) if tiled else predict(frame_bgr)

    def _yolo_boxes(
        self,
        frame_bgr: np.ndarray,
        model,
        kind: str,
        *,
        imgsz=640,
        conf: float | None = None,
    ) -> list[dict]:
        if conf is None:
            conf = 0.375 if kind == "occupancy" else 0.35
        result = model.predict(
            frame_bgr,
            verbose=False,
            imgsz=imgsz,
            conf=float(conf),
            device=yolo_device(),
        )[0]
        boxes = []
        if result.boxes is None:
            return boxes
        names = result.names or {}
        mask_points = result.masks.xy if result.masks is not None else None
        corner_points = result.keypoints.xy if result.keypoints is not None else None
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
            elif kind == "occupancy" and corner_points is not None and index < len(corner_points):
                candidate = corner_points[index].cpu().numpy()
                if candidate.shape == (4, 2) and np.isfinite(candidate).all():
                    points = candidate.tolist()
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
            outline = straightened_outline(polygon).astype(np.int32) if label in SPACE_LABELS else polygon
            cv2.polylines(canvas, [outline], True, color, 2)
            x1, y1 = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        else:
            x1, y1, x2, y2 = box["xyxy"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{_display_label(label)} {box['conf']:.2f}"
        cv2.putText(canvas, text, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def draw_occupancy(
    frame_bgr: np.ndarray,
    space_states: list[dict],
    vehicle_boxes: list[dict] | None = None,
) -> np.ndarray:
    """고정 슬롯 점유 색 + 차량 박스."""
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    for space in space_states:
        state = str(space.get("state") or "unknown")
        color = COLORS.get(state, COLORS["unknown"])
        points = space.get("points") or []
        if len(points) < 3:
            continue
        polygon = np.asarray(
            [[float(x) * width, float(y) * height] for x, y in points],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
        outline = straightened_outline(polygon).astype(np.int32)
        cv2.polylines(canvas, [outline], True, color, 2)
        x1, y1 = int(polygon[:, 0].min()), int(polygon[:, 1].min())
        label = f"{space.get('space_id', '')} {state}"
        cv2.putText(canvas, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if vehicle_boxes:
        canvas = draw_detections(canvas, vehicle_boxes)
    return canvas


def space_polygons(frame_bgr: np.ndarray, boxes: list[dict]) -> list[dict]:
    """주차칸 마스크를 0~1 폴리곤으로 단순화한다. 꼭짓점은 3개 이상이면 된다."""
    height, width = frame_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return []
    polygons = []
    for box in boxes:
        if box.get("label") not in SPACE_LABELS:
            continue
        raw = box.get("points")
        if not raw or len(raw) < 3:
            x1, y1, x2, y2 = box["xyxy"]
            raw = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        contour = np.asarray(raw, dtype=np.float32).reshape(-1, 1, 2)
        if len(contour) > 8:
            perimeter = cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, max(1.5, 0.012 * perimeter), True)
            if len(simplified) >= 3:
                contour = simplified
        points = []
        for x, y in contour.reshape(-1, 2):
            points.append([
                min(1.0, max(0.0, float(x) / width)),
                min(1.0, max(0.0, float(y) / height)),
            ])
        if len(points) < 3:
            continue
        label = str(box.get("label") or "parking_space")
        if label in {"empty", "vacant", "not occupied"}:
            tags = ["empty"]
        elif label == "occupied":
            tags = ["occupied"]
        else:
            tags = ["unknown"]
        polygons.append(
            {
                "points": points,
                "tags": tags,
                "label": "parking_space",
                "source_name": "model",
            }
        )
    return polygons


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
