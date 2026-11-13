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
    return _read_image_polygons(rel)


def polygons_from_folder_layout(rel: str, current: list[dict] | None = None) -> list[dict]:
    """같은 폴더에 저장된 칸 모양. 이 사진에 상태가 없으면 차 없음."""
    if not _shares_polygons(rel):
        return []
    layout = _read_layout(rel)
    if not layout:
        return []
    by_id = {polygon.get("id"): polygon for polygon in current or [] if polygon.get("id")}
    applied = []
    for polygon in layout:
        source = by_id.get(polygon["id"])
        if polygon["label"] == "occlusion":
            tags, label = ["occlusion"], "occlusion"
        elif source and source.get("tags"):
            tags = list(source["tags"])
            label = "occlusion" if source.get("label") == "occlusion" or "occlusion" in tags else "parking_space"
            if label == "occlusion":
                tags = ["occlusion"]
        else:
            tags, label = ["empty"], "parking_space"
        applied.append(
            {
                "id": polygon["id"],
                "points": [list(point) for point in polygon["points"]],
                "tags": tags,
                "label": label,
            }
        )
    return applied


def _shares_polygons(rel: str) -> bool:
    return Path(rel).as_posix().startswith("cctv_image/")


def _layout_path(rel: str) -> Path:
    return ANNOT_DIR / Path(rel).parent / "_layout.json"


def _read_image_polygons(rel: str) -> list[dict]:
    path = annot_path(Path(rel))
    if not path.exists():
        return []
    return polygons_from_annotation(json.loads(path.read_text(encoding="utf-8")))


def _read_layout(rel: str) -> list[dict]:
    path = _layout_path(rel)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons = []
    for index, item in enumerate(data.get("polygons") or [], start=1):
        points = [[float(x), float(y)] for x, y in item.get("points") or []]
        if len(points) < 3:
            continue
        label = "occlusion" if item.get("label") == "occlusion" else "parking_space"
        polygons.append(
            {
                "id": item.get("id") or f"p_{index}",
                "points": points,
                "tags": ["occlusion"] if label == "occlusion" else ["empty"],
                "label": label,
            }
        )
    return polygons


def _write_layout(rel: str, polygons: list[dict]) -> None:
    path = _layout_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for polygon in polygons:
        points = [[float(x), float(y)] for x, y in polygon.get("points") or []]
        if len(points) < 3:
            continue
        tags = [str(tag).lower() for tag in polygon.get("tags") or []]
        label = "occlusion" if polygon.get("label") == "occlusion" or "occlusion" in tags else "parking_space"
        items.append({"id": polygon.get("id") or f"p_{len(items) + 1}", "points": points, "label": label})
    payload = {"_cls": "SharedLayout", "polygons": items}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_folder_annotations(rel: str, polygons: list[dict]) -> list[str]:
    """현재 칸을 같은 폴더의 모든 사진에 저장하고 사진별 상태는 유지한다."""
    if not _shares_polygons(rel) or not polygons:
        return []
    folder = DATA_DIR / Path(rel).parent
    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    save_annotation(rel, polygons)
    applied = []
    for path in images:
        target = path.relative_to(DATA_DIR).as_posix()
        if target != rel:
            merged = polygons_from_folder_layout(target, load_annotation(target))
            save_annotation(target, merged, update_layout=False)
        applied.append(target)
    return applied


def save_annotation(rel: str, polygons: list[dict], *, update_layout: bool = True) -> Path:
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
                "id": polygon.get("id") or f"p_{space_id}",
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
    if update_layout and _shares_polygons(rel) and polygons:
        _write_layout(rel, polygons)
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
