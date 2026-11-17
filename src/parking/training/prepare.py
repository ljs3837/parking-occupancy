"""로컬 색인 → CNN crop / YOLO 데이터셋 변환."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2

from parking.app.annotator.io import flat_points, iter_annotation_sources
from parking.app.paths import DATA_DIR, ROOT

RUNS_DIR = ROOT / "outputs" / "train"
MODELS_DIR = ROOT / "models"


def next_model_path(directory: Path, stem: str) -> Path:
    """이전 가중치는 남기고 stem_1.pt, stem_2.pt 순으로 새 경로를 만든다."""
    directory.mkdir(parents=True, exist_ok=True)
    used = []
    for path in directory.glob(f"{stem}_*.pt"):
        tail = path.stem[len(stem) + 1 :]
        if tail.isdigit():
            used.append(int(tail))
    number = max(used, default=0) + 1
    return directory / f"{stem}_{number}.pt"
CNN_DATA = RUNS_DIR / "cnn_crops"
YOLO_DATA = RUNS_DIR / "yolo_parking"

OCC_POS = {"occupied"}
OCC_NEG = {"empty", "not occupied"}


def _norm_points(points) -> list[list[float]]:
    return flat_points(points)


def _bbox_from_points(points: list[list[float]], w: int, h: int) -> tuple[int, int, int, int]:
    xs = [p[0] * w for p in points]
    ys = [p[1] * h for p in points]
    x1, x2 = int(max(0, min(xs))), int(min(w - 1, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(h - 1, max(ys)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def _yolo_box(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    return cx, cy, bw, bh


def _class_id(status: str) -> int | None:
    """CNN 점유 분류용. YOLO 면 발견은 이 값을 쓰지 않는다."""
    s = (status or "").strip().lower()
    if s in OCC_POS or s == "occupied":
        return 1
    if s in OCC_NEG or s in {"empty", "not occupied"}:
        return 0
    return None


def _is_parking_space(polygon: dict) -> bool:
    """가림이 아닌 주차칸 폴리곤만 면 발견 학습에 쓴다. 점유 여부는 보지 않는다."""
    tags = [str(tag).lower() for tag in polygon.get("tags") or []]
    if polygon.get("label") == "occlusion" or "occlusion" in tags:
        return False
    return len(_norm_points(polygon.get("points") or [])) >= 3


def _resolve_image(rel: str) -> tuple[str, Path] | None:
    img = DATA_DIR / rel
    if img.is_file():
        return img.relative_to(DATA_DIR).as_posix(), img
    stem = DATA_DIR / Path(rel).with_suffix("")
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        cand = stem.with_suffix(ext)
        if cand.is_file():
            return cand.relative_to(DATA_DIR).as_posix(), cand
    return None


def _status_of(polygon: dict) -> str:
    if polygon.get("label") == "occlusion" or "occlusion" in polygon.get("tags", []):
        return "unknown"
    for tag in polygon.get("tags") or []:
        name = str(tag).lower()
        if name in OCC_POS:
            return "occupied"
        if name in OCC_NEG:
            return "not occupied"
    return "unknown"


def _item_from_polygons(rel: str, polygons: list[dict]) -> dict | None:
    resolved = _resolve_image(rel)
    if not resolved:
        return None
    rel, img = resolved
    polys = []
    for polygon in polygons:
        if not _is_parking_space(polygon):
            continue
        points = _norm_points(polygon.get("points") or [])
        polys.append(
            {
                "points": points,
                "status": _status_of(polygon),
                "label": polygon.get("label") or "parking_space",
            }
        )
    if not polys:
        return None
    return {"rel": rel, "path": img, "polygons": polys}


def _load_local_annotations() -> list[dict]:
    """우리 색인 JSON → [{path, polygons}]. 공유 폴더는 사진마다 같은 칸을 쓴다."""
    items = []
    for rel, polygons in iter_annotation_sources():
        item = _item_from_polygons(rel, polygons)
        if item:
            items.append(item)
    return items


def collect_samples(log) -> list[dict]:
    local = _load_local_annotations()
    log(f"[데이터] 색인된 사진 전부 사용: {len(local)}장 (허깅페이스 제외)")
    return local


def build_cnn_crops(log) -> Path:
    samples = collect_samples(log)
    if not samples:
        raise RuntimeError("학습용 이미지가 없습니다. data/annotations 색인을 먼저 준비하세요.")

    if CNN_DATA.exists():
        shutil.rmtree(CNN_DATA)
    for split in ("train", "val"):
        for cls in ("empty", "occupied"):
            (CNN_DATA / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.2))
    val_set = set(id(s) for s in samples[:n_val])

    counts = {"train": 0, "val": 0}
    for s in samples:
        split = "val" if id(s) in val_set else "train"
        img = cv2.imread(str(s["path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for i, poly in enumerate(s["polygons"]):
            cid = _class_id(poly["status"])
            if cid is None:
                continue
            cls = "occupied" if cid == 1 else "empty"
            x1, y1, x2, y2 = _bbox_from_points(poly["points"], w, h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out = CNN_DATA / split / cls / f"{Path(s['rel']).stem}_{i}.jpg"
            cv2.imwrite(str(out), crop)
            counts[split] += 1

    log(f"[CNN 데이터] train crops={counts['train']}, val crops={counts['val']}")
    log(f"[CNN 데이터] 경로: {CNN_DATA}")
    if counts["train"] < 10:
        raise RuntimeError("CNN crop이 너무 적습니다. 점유가 표시된 색인을 추가하세요.")
    return CNN_DATA


def build_yolo_dataset(log) -> Path:
    samples = collect_samples(log)
    if not samples:
        raise RuntimeError("학습용 이미지가 없습니다. data/annotations 색인을 먼저 준비하세요.")

    if YOLO_DATA.exists():
        shutil.rmtree(YOLO_DATA)
    for split in ("train", "val"):
        (YOLO_DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.2))
    val_set = set(id(s) for s in samples[:n_val])

    n_img = 0
    for s in samples:
        split = "val" if id(s) in val_set else "train"
        src = Path(s["path"])
        stem = f"{src.stem}_{n_img}"
        dst_img = YOLO_DATA / "images" / split / f"{stem}{src.suffix.lower()}"
        shutil.copy2(src, dst_img)

        lines = []
        for poly in s["polygons"]:
            # 점유(empty/occupied)는 무시하고 주차칸 좌표만 단일 클래스로 학습한다.
            if poly.get("label") == "occlusion" or poly.get("status") == "occlusion":
                continue
            if len(poly["points"]) < 3:
                continue
            coords = []
            for x, y in poly["points"]:
                coords.append(f"{min(1.0, max(0.0, x)):.6f}")
                coords.append(f"{min(1.0, max(0.0, y)):.6f}")
            lines.append("0 " + " ".join(coords))
        if not lines:
            dst_img.unlink(missing_ok=True)
            continue
        (YOLO_DATA / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        n_img += 1

    yaml_path = YOLO_DATA / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {YOLO_DATA.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: parking_space",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[YOLO 데이터] 주차칸 좌표만 사용 · 이미지 {n_img}장 → {YOLO_DATA}")
    return yaml_path
