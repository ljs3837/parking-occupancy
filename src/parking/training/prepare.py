"""로컬 색인, 허깅페이스 PKLot, Kaggle 주차 사진을 YOLO 데이터셋으로 변환."""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from parking.app.annotator.io import flat_points, iter_annotation_sources, polygons_from_annotation
from parking.app.paths import DATA_DIR, ROOT
from parking.training.occlusion import occlude_slots

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
KAGGLE_DIR = DATA_DIR / "kaggle" / "parking-space-detection"
KAGGLE_XML = KAGGLE_DIR / "annotations.xml"
KAGGLE_IMAGES = KAGGLE_DIR / "images"
YOLO_DATA = RUNS_DIR / "yolo_parking"

OCC_POS = {"occupied"}
OCC_NEG = {"empty", "not occupied"}
# data/cctv 인간 색인은 주차장이 제각각이라, 학습 세트에서만 이 횟수만큼 반복한다.
CCTV_TRAIN_REPEAT = 10


def _is_human_cctv(sample: dict) -> bool:
    return str(sample.get("rel") or "").startswith("cctv/")


def _split_rows(samples: list[dict], *, cctv_repeat: int = CCTV_TRAIN_REPEAT) -> tuple[list[tuple[str, dict]], int, int]:
    """검증은 원본 1장. cctv 인간 색인만 학습 쪽에서 반복한다."""
    if cctv_repeat < 1:
        raise ValueError("cctv_repeat must be at least one")
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.2))
    rows: list[tuple[str, dict]] = []
    cctv_train = 0
    for index, sample in enumerate(samples):
        if index < n_val:
            rows.append(("val", sample))
            continue
        repeats = cctv_repeat if _is_human_cctv(sample) else 1
        if _is_human_cctv(sample):
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


def _polygon_points(raw: str, width: float, height: float) -> list[list[float]]:
    points = []
    for pair in raw.split(";"):
        if "," not in pair:
            continue
        x_text, y_text = pair.split(",", 1)
        try:
            x = float(x_text) / width
            y = float(y_text) / height
        except ValueError:
            continue
        points.append([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
    return points


def _load_kaggle_parking(skip_paths: set[str], log) -> list[dict]:
    """원본 사진과 XML 칸 좌표만 읽는다. 선이 그려진 boxes 이미지는 쓰지 않는다."""
    if not KAGGLE_XML.is_file() or not KAGGLE_IMAGES.is_dir():
        log(f"[데이터] Kaggle 주차 데이터 없음: {KAGGLE_DIR}")
        return []

    root = ET.parse(KAGGLE_XML).getroot()
    items = []
    for image in root.iter("image"):
        name = Path(image.get("name") or "").name
        path = KAGGLE_IMAGES / name
        if not name or not path.is_file() or str(path) in skip_paths:
            continue
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        polygons = []
        for polygon in image.findall("polygon"):
            points = _polygon_points(polygon.get("points") or "", width, height)
            if len(points) < 3:
                continue
            polygons.append(
                {
                    "points": points,
                    "status": "unknown",
                    "label": "parking_space",
                }
            )
        if not polygons:
            continue
        try:
            rel = path.relative_to(DATA_DIR).as_posix()
        except ValueError:
            rel = path.as_posix()
        items.append({"rel": rel, "path": path, "polygons": polygons})
    log(f"[데이터] Kaggle 주차 원본 {len(items)}장")
    return items


# annotations는 칸 좌표 파일이고, observe는 관제 영상이다. 학습 사진 폴더가 아니다.
SKIP_DATA_FOLDERS = {"annotations", "observe"}


def list_data_folders() -> list[str]:
    """학습에 넣을 수 있는 data 바로 아래 폴더. 기본 선택은 이 전체다."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in DATA_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in SKIP_DATA_FOLDERS
    )


def _in_selected_folder(rel: str, folders: set[str] | None) -> bool:
    if folders is None:
        return True
    top = rel.split("/", 1)[0] if rel else ""
    return top in folders


def collect_samples(log, folders: set[str] | None = None) -> list[dict]:
    """folders가 None이면 data 아래 폴더를 전부 쓴다."""
    allowed = None if folders is None else set(folders)
    local = [
        item
        for item in _load_local_annotations()
        if _in_selected_folder(str(item.get("rel") or ""), allowed)
    ]
    log(f"[데이터] 로컬 색인 {len(local)}장")
    skip = {str(item["path"]) for item in local}
    use_huggingface = allowed is None or "huggingface" in allowed
    use_kaggle = allowed is None or "kaggle" in allowed
    huggingface = _load_huggingface(skip, log) if use_huggingface else []
    if not use_huggingface:
        log("[데이터] 허깅페이스 폴더 제외")
    kaggle = _load_kaggle_parking(skip | {str(item["path"]) for item in huggingface}, log) if use_kaggle else []
    if not use_kaggle:
        log("[데이터] Kaggle 폴더 제외")
    if allowed is not None:
        huggingface = [item for item in huggingface if _in_selected_folder(str(item.get("rel") or ""), allowed)]
        kaggle = [item for item in kaggle if _in_selected_folder(str(item.get("rel") or ""), allowed)]
    total = len(local) + len(huggingface) + len(kaggle)
    chosen = "전체" if allowed is None else ", ".join(sorted(allowed))
    log(
        f"[데이터] 합계 {total}장 "
        f"(로컬 {len(local)} + 허깅페이스 {len(huggingface)} + Kaggle {len(kaggle)}) "
        f"· 폴더 {chosen}"
    )
    return local + huggingface + kaggle


def build_yolo_dataset(log, *, samples=None, output_dir=None, cctv_repeat=CCTV_TRAIN_REPEAT, folders=None) -> Path:
    samples = list(collect_samples(log, folders) if samples is None else samples)
    data_dir = (Path(output_dir) if output_dir is not None else YOLO_DATA).resolve()
    run_root = RUNS_DIR.resolve()
    if data_dir == run_root or not data_dir.is_relative_to(run_root):
        raise ValueError("Dataset output must be a subdirectory of outputs/train")
    if not samples:
        raise RuntimeError("학습용 이미지가 없습니다. 색인 또는 허깅페이스 PKLot를 확인하세요.")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    for split in ("train", "val"):
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows, cctv_train, n_val = _split_rows(samples, cctv_repeat=cctv_repeat)

    n_img = 0
    n_occluded = 0
    for split, s in rows:
        src = Path(s["path"])
        stem = f"{src.stem}_{n_img}"
        dst_img = data_dir / "images" / split / f"{stem}{src.suffix.lower()}"
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
        (data_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        # Augment train only, after splitting: no augmented validation leakage.
        if split == "train":
            import cv2
            import numpy as np

            image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                augmented, changed = occlude_slots(image, s["polygons"], seed=42 + n_img)
                if changed:
                    augmented_path = dst_img.with_name(f"{stem}_occluded.jpg")
                    ok, encoded = cv2.imencode(".jpg", augmented)
                    if not ok:
                        raise RuntimeError(f"Could not encode {augmented_path}")
                    encoded.tofile(str(augmented_path))
                    (data_dir / "labels" / split / f"{stem}_occluded.txt").write_text(
                        "\n".join(lines) + "\n", encoding="utf-8"
                    )
                    n_occluded += 1
        n_img += 1

    log(f"[YOLO] synthetic partial occlusion: {n_occluded} train images; full labels retained")
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {data_dir.as_posix()}",
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
        f"[YOLO 데이터] 주차칸 좌표 · 이미지 {n_img}장"
        f" · CCTV 학습 원본 {cctv_train}장 × {cctv_repeat}"
        f" · 검증 원본 {n_val}장 → {data_dir}"
    )
    return yaml_path
