"""어노테이션·PKLot → CNN crop / YOLO 데이터셋 변환."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2

from parking.app.annotator.io import flat_points, polygons_from_annotation
from parking.app.paths import ANNOT_DIR, DATA_DIR, ROOT

SAMPLES_JSON = DATA_DIR / "huggingface" / "hub" / "Voxel51" / "PKLot" / "samples.json"

RUNS_DIR = ROOT / "outputs" / "train"
MODELS_DIR = ROOT / "models"
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
    s = (status or "").strip().lower()
    if s in OCC_POS or s == "occupied":
        return 1
    if s in OCC_NEG or s in {"empty", "not occupied"}:
        return 0
    return None


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
        points = _norm_points(polygon.get("points") or [])
        if len(points) < 3:
            continue
        polys.append({"points": points, "status": _status_of(polygon)})
    if not polys:
        return None
    return {"rel": rel, "path": img, "polygons": polys}


def _load_local_annotations() -> list[dict]:
    """우리 색인 JSON → [{path, polygons}]"""
    items = []
    if not ANNOT_DIR.exists():
        return items
    for jp in ANNOT_DIR.rglob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        rel = data.get("filepath") or data.get("image") or jp.relative_to(ANNOT_DIR).with_suffix(".jpg").as_posix()
        item = _item_from_polygons(rel, polygons_from_annotation(data))
        if item:
            items.append(item)
    return items


def _sample_budgets(n_local: int, max_images: int) -> tuple[int, int]:
    """로컬 색인과 samples.json이 같이 들어가도록 장수를 나눈다."""
    if max_images <= 0:
        return 0, 0
    if n_local <= 0:
        return 0, max_images
    pklot_slots = max(1, max_images // 2)
    local_slots = max_images - pklot_slots
    if n_local < local_slots:
        pklot_slots += local_slots - n_local
        local_slots = n_local
    return local_slots, pklot_slots


def _load_samples_json(limit: int, skip_paths: set[str], log) -> list[dict]:
    if limit <= 0:
        return []
    if not SAMPLES_JSON.is_file():
        log(f"[데이터] samples.json 없음: {SAMPLES_JSON}")
        return []

    log(f"[데이터] samples.json 로드 중… ({SAMPLES_JSON})")
    raw = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    samples = raw["samples"] if isinstance(raw, dict) else raw
    base = SAMPLES_JSON.parent
    candidates = []
    for sample in samples:
        filepath = sample.get("filepath") or ""
        path = base / filepath
        if not path.is_file():
            path = DATA_DIR / filepath
        if not path.is_file() or str(path) in skip_paths:
            continue
        candidates.append((sample, path))
    random.seed(42)
    if len(candidates) > limit:
        candidates = random.sample(candidates, limit)

    items = []
    for sample, path in candidates:
        try:
            rel = path.relative_to(DATA_DIR).as_posix()
        except ValueError:
            rel = path.name
        item = _item_from_polygons(rel, polygons_from_annotation(sample))
        if item:
            items.append(item)
    log(f"[데이터] samples.json에서 {len(items)}장 사용")
    return items


def collect_samples(max_images: int, log) -> list[dict]:
    local = _load_local_annotations()
    log(f"[데이터] 로컬 색인 이미지: {len(local)}")
    local_slots, pklot_slots = _sample_budgets(len(local), max_images)
    if len(local) > local_slots:
        random.seed(42)
        local = random.sample(local, local_slots)
    skip_paths = {str(item["path"]) for item in local}
    pklot = _load_samples_json(pklot_slots, skip_paths, log)
    log(f"[데이터] 합계 {len(local) + len(pklot)}장 (로컬 {len(local)} + samples.json {len(pklot)})")
    return local + pklot


def build_cnn_crops(max_images: int, log) -> Path:
    samples = collect_samples(max_images, log)
    if not samples:
        raise RuntimeError("학습용 이미지가 없습니다. 색인을 하거나 PKLot를 준비하세요.")

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
        raise RuntimeError("CNN crop이 너무 적습니다. max_images를 늘리거나 색인을 추가하세요.")
    return CNN_DATA


def build_yolo_dataset(max_images: int, log) -> Path:
    samples = collect_samples(max_images, log)
    if not samples:
        raise RuntimeError("학습용 이미지가 없습니다. 색인을 하거나 PKLot를 준비하세요.")

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
            cid = _class_id(poly["status"])
            if cid is None or len(poly["points"]) < 3:
                continue
            coords = []
            for x, y in poly["points"]:
                coords.append(f"{min(1.0, max(0.0, x)):.6f}")
                coords.append(f"{min(1.0, max(0.0, y)):.6f}")
            lines.append(f"{cid} " + " ".join(coords))
        (YOLO_DATA / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
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
                "  0: empty",
                "  1: occupied",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[YOLO 데이터] 이미지 {n_img}장 → {YOLO_DATA}")
    return yaml_path
