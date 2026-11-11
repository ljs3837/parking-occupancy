from __future__ import annotations

import json
from pathlib import Path

from parking.app.paths import ANNOT_DIR, DATA_DIR, IMAGE_EXTS, VIDEO_EXTS


def annot_path(rel: Path) -> Path:
    return ANNOT_DIR / rel.with_suffix(".json")


def list_images() -> list[str]:
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    annot_root = ANNOT_DIR.resolve()
    items: list[str] = []
    for p in DATA_DIR.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            if annot_root in p.resolve().parents:
                continue
        except OSError:
            continue
        items.append(p.relative_to(DATA_DIR).as_posix())
    items.sort()
    return items


def list_media() -> list[str]:
    """Return image and video paths while keeping the existing image API intact."""
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    annot_root = ANNOT_DIR.resolve()
    items: list[str] = []
    for p in DATA_DIR.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        try:
            if annot_root in p.resolve().parents:
                continue
        except OSError:
            continue
        items.append(p.relative_to(DATA_DIR).as_posix())
    items.sort()
    return items


def flat_points(points) -> list[list[float]]:
    """FiftyOne `[[[x,y],...]]` 와 `[[x,y],...]` 를 같은 좌표 목록으로 푼다."""
    if not points:
        return []
    if (
        isinstance(points[0], (list, tuple))
        and points[0]
        and isinstance(points[0][0], (list, tuple))
    ):
        return [[float(x), float(y)] for x, y in points[0]]
    return [[float(x), float(y)] for x, y in points]


def _occupancy_status(polygon: dict) -> tuple[str, str]:
    label = polygon.get("label") or "parking_space"
    tags = [str(tag).lower() for tag in polygon.get("tags") or []]
    if label == "occlusion" or "occlusion" in tags:
        return "occlusion", "unknown"
    if "occupied" in tags:
        return "parking_space", "occupied"
    if "empty" in tags or "not occupied" in tags:
        return "parking_space", "not occupied"
    if "unknown" in tags:
        return "parking_space", "unknown"
    return label, "unknown"


def _polygon_from_polyline(polyline: dict, fallback_id: str) -> dict | None:
    points = flat_points(polyline.get("points") or [])
    if len(points) < 3:
        return None
    label = polyline.get("label") or "parking_space"
    status = (polyline.get("occupancy_status") or "unknown").strip().lower()
    if label == "occlusion":
        tags = ["occlusion"]
    elif status == "occupied":
        tags = ["occupied"]
        label = "parking_space"
    elif status in {"not occupied", "empty"}:
        tags = ["empty"]
        label = "parking_space"
    else:
        tags = ["unknown"]
    return {
        "id": polyline.get("id") or fallback_id,
        "points": points,
        "tags": tags,
        "label": label,
    }


def polygons_from_annotation(data) -> list[dict]:
    """예전 polygons 형식과 samples.json 샘플 형식을 모두 읽는다."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if data.get("polygons"):
        return data["polygons"]
    polylines = ((data.get("parking_spaces") or {}).get("polylines")) or []
    polygons = []
    for index, polyline in enumerate(polylines, start=1):
        polygon = _polygon_from_polyline(polyline, f"p_{index}")
        if polygon:
            polygons.append(polygon)
    return polygons


def load_annotation(rel: str) -> list[dict]:
    ap = annot_path(Path(rel))
    if not ap.exists():
        return []
    data = json.loads(ap.read_text(encoding="utf-8"))
    return polygons_from_annotation(data)


def save_annotation(rel: str, polygons: list[dict]) -> Path:
    """samples.json 한 장 분량과 같은 키로 저장한다."""
    ap = annot_path(Path(rel))
    ap.parent.mkdir(parents=True, exist_ok=True)
    polylines = []
    space_id = 0
    for polygon in polygons:
        label, status = _occupancy_status(polygon)
        space_id += 1
        polylines.append(
            {
                "_cls": "Polyline",
                "attributes": {},
                "tags": [],
                "label": label,
                "points": [[[float(x), float(y)] for x, y in polygon.get("points") or []]],
                "index": space_id,
                "closed": True,
                "filled": True,
                "occupancy_status": status,
                "space_id": space_id,
            }
        )
    payload = {
        "filepath": rel,
        "tags": [],
        "_media_type": "image",
        "parking_spaces": {"_cls": "Polylines", "polylines": polylines},
    }
    ap.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ap


def occupancy_of(tags: list[str]) -> str:
    if "occlusion" in tags:
        return "occlusion"
    if "occupied" in tags:
        return "occupied"
    if "empty" in tags or "not occupied" in tags:
        return "empty"
    if "unknown" in tags:
        return "unknown"
    return "unknown"
