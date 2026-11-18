"""로컬 색인과 허깅페이스 PKLot 전부를 YOLO 데이터셋으로 변환."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from parking.app.annotator.io import flat_points, iter_annotation_sources, polygons_from_annotation
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


SAMPLES_JSON = DATA_DIR / "huggingface" / "hub" / "Voxel51" / "PKLot" / "samples.json"
YOLO_DATA = RUNS_DIR / "yolo_parking"

OCC_POS = {"occupied"}
OCC_NEG = {"empty", "not occupied"}
# data/cctv 인간 색인은 주차장이 제각각이라, 학습 세트에서만 이 횟수만큼 반복한다.
CCTV_TRAIN_REPEAT = 10


def _is_human_cctv(sample: dict) -> bool:
    return str(sample.get("rel") or "").startswith("cctv/")


def _split_rows(samples: list[dict]) -> tuple[list[tuple[str, dict]], int, int]:
    """검증은 원본 1장. cctv 인간 색인만 학습 쪽에서 반복한다."""
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.2))
    rows: list[tuple[str, dict]] = []
    cctv_train = 0
    for index, sample in enumerate(samples):
        if index < n_val:
            rows.append(("val", sample))
            continue
        repeats = CCTV_TRAIN_REPEAT if _is_human_cctv(sample) else 1
        if repeats > 1:
            cctv_train += 1
        rows.extend(("train", sample) for _ in range(repeats))
    return rows, cctv_train, n_val


def _norm_points(points) -> list[list[float]]:
    return flat_points(points)


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


def _load_huggingface(skip_paths: set[str], log) -> list[dict]:
    """PKLot samples.json에 있는 사진을 빠짐없이 읽는다."""
    if not SAMPLES_JSON.is_file():
        log(f"[데이터] 허깅페이스 samples.json 없음: {SAMPLES_JSON}")
        return []

    log(f"[데이터] 허깅페이스 목록 읽는 중…")
    raw = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    samples = raw["samples"] if isinstance(raw, dict) else raw
    base = SAMPLES_JSON.parent
    items = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        if index == 1 or index % 2000 == 0 or index == total:
            log(f"[데이터] 허깅페이스 확인 {index}/{total}")
        filepath = sample.get("filepath") or ""
        path = base / filepath
        if not path.is_file():
            path = DATA_DIR / filepath
        if not path.is_file() or str(path) in skip_paths:
            continue
        try:
            rel = path.relative_to(DATA_DIR).as_posix()
        except ValueError:
            rel = path.as_posix()
        item = _item_from_polygons(rel, polygons_from_annotation(sample))
        if item:
            items.append(item)
    log(f"[데이터] 허깅페이스 {len(items)}장")
    return items


def collect_samples(log) -> list[dict]:
    local = _load_local_annotations()
    log(f"[데이터] 로컬 색인 {len(local)}장")
    huggingface = _load_huggingface({str(item["path"]) for item in local}, log)
    log(f"[데이터] 합계 {len(local) + len(huggingface)}장 (로컬 {len(local)} + 허깅페이스 {len(huggingface)})")
    return local + huggingface


def build_yolo_dataset(log) -> Path:
    samples = collect_samples(log)
    if not samples:
        raise RuntimeError("학습용 이미지가 없습니다. 색인 또는 허깅페이스 PKLot를 확인하세요.")

    if YOLO_DATA.exists():
        shutil.rmtree(YOLO_DATA)
    for split in ("train", "val"):
        (YOLO_DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows, cctv_train, n_val = _split_rows(samples)

    n_img = 0
    for split, s in rows:
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
    log(
        f"[YOLO 데이터] 주차칸 좌표만 사용 · 이미지 {n_img}장"
        f" · cctv 인간 색인 학습 {cctv_train}장 × {CCTV_TRAIN_REPEAT}"
        f" · 검증 원본 {n_val}장 → {YOLO_DATA}"
    )
    return yaml_path
