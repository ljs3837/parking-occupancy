"""어노테이션·PKLot → CNN crop / YOLO 데이터셋 변환."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2

from parking.app.paths import ANNOT_DIR, DATA_DIR, ROOT

RUNS_DIR = ROOT / "outputs" / "train"
MODELS_DIR = ROOT / "models"
CNN_DATA = RUNS_DIR / "cnn_crops"
YOLO_DATA = RUNS_DIR / "yolo_parking"

OCC_POS = {"occupied"}
OCC_NEG = {"empty", "not occupied"}


def _norm_points(points) -> list[list[float]]:
    if not points:
        return []
    # FiftyOne: [[[x,y],...]] / 우리 어노테이션: [[x,y],...]
    if (
        isinstance(points[0], (list, tuple))
        and points[0]
        and isinstance(points[0][0], (list, tuple))
    ):
        return [[float(x), float(y)] for x, y in points[0]]
    return [[float(x), float(y)] for x, y in points]


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


def _load_local_annotations() -> list[dict]:
    """우리 색인 JSON → [{path, polygons}]"""
    items = []
    if not ANNOT_DIR.exists():
        return items
    for jp in ANNOT_DIR.rglob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        rel = data.get("image") or jp.relative_to(ANNOT_DIR).with_suffix(".jpg").as_posix()
        img = DATA_DIR / rel
        if not img.is_file():
            # 확장자 탐색
            stem = (DATA_DIR / Path(rel).with_suffix("")).as_posix()
            found = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = Path(stem + ext)
                if cand.is_file():
                    found = cand
                    break
            if not found:
                continue
            img = found
            rel = img.relative_to(DATA_DIR).as_posix()
        polys = []
        for p in data.get("polygons") or []:
            tags = p.get("tags") or []
            status = "unknown"
            for t in tags:
                tl = str(t).lower()
                if tl in OCC_POS or tl == "occupied":
                    status = "occupied"
                    break
                if tl in OCC_NEG:
                    status = "not occupied"
                    break
            pts = _norm_points(p.get("points") or [])
            if len(pts) >= 3:
                polys.append({"points": pts, "status": status})
        if polys:
            items.append({"rel": rel, "path": img, "polygons": polys})
    return items


def _load_pklot(max_images: int, log) -> list[dict]:
    """FiftyOne PKLot 또는 samples.json에서 샘플 로드."""
    items: list[dict] = []

    # 1) FiftyOne DB
    try:
        import fiftyone as fo

        names = fo.list_datasets()
        name = "PKLot" if "PKLot" in names else ("Voxel51/PKLot" if "Voxel51/PKLot" in names else None)
        if name:
            log(f"[데이터] FiftyOne 데이터셋 사용: {name}")
            ds = fo.load_dataset(name)
            view = ds.take(max_images) if len(ds) > max_images else ds
            for sample in view:
                path = Path(sample.filepath)
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(DATA_DIR).as_posix()
                except ValueError:
                    rel = path.name
                polys = []
                spaces = sample["parking_spaces"] if "parking_spaces" in sample else None
                if spaces is None:
                    continue
                for pl in spaces.polylines:
                    status = getattr(pl, "occupancy_status", None) or "unknown"
                    pts = _norm_points(pl.points)
                    if len(pts) >= 3:
                        polys.append({"points": pts, "status": status})
                if polys:
                    items.append({"rel": rel, "path": path, "polygons": polys})
            if items:
                return items
    except Exception as e:
        log(f"[데이터] FiftyOne 로드 실패 → samples.json 시도 ({e})")

    # 2) samples.json
    samples_path = DATA_DIR / "huggingface" / "hub" / "Voxel51" / "PKLot" / "samples.json"
    if not samples_path.is_file():
        log("[데이터] PKLot samples.json 없음")
        return items

    log(f"[데이터] samples.json 로드 중… ({samples_path})")
    raw = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = raw["samples"] if isinstance(raw, dict) else raw
    base = samples_path.parent
    random.seed(42)
    chosen = samples if len(samples) <= max_images else random.sample(samples, max_images)
    for s in chosen:
        fp = s.get("filepath") or ""
        path = base / fp
        if not path.is_file():
            path = DATA_DIR / fp
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(DATA_DIR).as_posix()
        except ValueError:
            rel = path.name
        polys = []
        for pl in ((s.get("parking_spaces") or {}).get("polylines") or []):
            status = pl.get("occupancy_status") or "unknown"
            pts = _norm_points(pl.get("points") or [])
            if len(pts) >= 3:
                polys.append({"points": pts, "status": status})
        if polys:
            items.append({"rel": rel, "path": path, "polygons": polys})
    return items


def collect_samples(max_images: int, log) -> list[dict]:
    local = _load_local_annotations()
    log(f"[데이터] 로컬 색인 이미지: {len(local)}")
    if len(local) >= max(10, max_images // 5):
        if len(local) > max_images:
            random.seed(42)
            local = random.sample(local, max_images)
        return local
    pklot = _load_pklot(max_images, log)
    log(f"[데이터] PKLot 이미지: {len(pklot)}")
    merged = local + pklot
    # 중복 경로 제거
    seen = set()
    out = []
    for it in merged:
        key = str(it["path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= max_images:
            break
    return out


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
            if cid is None:
                continue
            cx, cy, bw, bh = _yolo_box(poly["points"])
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
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
