from __future__ import annotations

import json
from pathlib import Path

from parking.app.paths import ANNOT_DIR, DATA_DIR, IMAGE_EXTS, VIDEO_EXTS


def shares_folder_annotation(rel: str) -> bool:
    """마지막 폴더의 사진이 같은 CCTV 구도면 색인 JSON 하나를 공유한다."""
    parts = Path(rel).parts
    if len(parts) >= 3 and parts[0] == "cctv_image":
        return True
    return len(parts) >= 2 and parts[0].startswith("camera") and parts[0][len("camera"):].isdigit()


def annot_path(rel: Path) -> Path:
    rel = Path(rel)
    if shares_folder_annotation(rel.as_posix()):
        return ANNOT_DIR / rel.parent / "_layout.json"
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
        points = points[0]
    flat = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        flat.append([float(point[0]), float(point[1])])
    return flat


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


def _polygons_from_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("parking_spaces"):
        return polygons_from_annotation(data)
    polygons = []
    for index, item in enumerate(data.get("polygons") or [], start=1):
        points = flat_points(item.get("points") or [])
        if len(points) < 3:
            continue
        polygon_id = item.get("id") or f"p_{index}"
        status = (item.get("occupancy_status") or "").strip().lower()
        if status or item.get("label") == "occlusion":
            converted = _polygon_from_polyline(
                {
                    "id": polygon_id,
                    "label": item.get("label") or "parking_space",
                    "points": points,
                    "occupancy_status": status or "unknown",
                },
                polygon_id,
            )
            if converted:
                polygons.append(converted)
            continue
        tags = [str(tag).lower() for tag in item.get("tags") or []]
        label = "occlusion" if item.get("label") == "occlusion" or "occlusion" in tags else "parking_space"
        if not tags:
            tags = ["occlusion"] if label == "occlusion" else ["empty"]
        polygons.append({"id": polygon_id, "points": points, "tags": tags, "label": label})
    return polygons


def _write_shared(path: Path, polygons: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for polygon in polygons:
        points = [[float(x), float(y)] for x, y in polygon.get("points") or []]
        if len(points) < 3:
            continue
        label, status = _occupancy_status(polygon)
        items.append(
            {
                "id": polygon.get("id") or f"p_{len(items) + 1}",
                "points": points,
                "label": label,
                "occupancy_status": status,
            }
        )
    payload = {"_cls": "SharedLayout", "polygons": items}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _geom_key(polygons: list[dict]) -> tuple:
    return tuple(
        tuple((round(float(x), 4), round(float(y), 4)) for x, y in polygon.get("points") or [])
        for polygon in polygons
    )


def _copy_occupancy(target: list[dict], donor: list[dict]) -> None:
    if _geom_key(target) != _geom_key(donor):
        return
    for dst, src in zip(target, donor):
        dst["tags"] = list(src.get("tags") or dst.get("tags") or [])
        dst["label"] = src.get("label") or dst.get("label")


_PKLOT_POLYGONS: dict[str, list[dict]] | None = None
_KAGGLE_XML_POLYGONS: dict[str, list[dict]] | None = None


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _parking_polygon(points: list[list[float]], *, tags: list[str] | None = None) -> dict | None:
    if len(points) < 3:
        return None
    return {
        "points": points,
        "tags": list(tags or ["unknown"]),
        "label": "parking_space",
    }


def _pklot_polygons_index() -> dict[str, list[dict]]:
    global _PKLOT_POLYGONS
    if _PKLOT_POLYGONS is not None:
        return _PKLOT_POLYGONS
    index: dict[str, list[dict]] = {}
    path = DATA_DIR / "huggingface" / "hub" / "Voxel51" / "PKLot" / "samples.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        samples = raw["samples"] if isinstance(raw, dict) else raw
        base = path.parent
        for sample in samples:
            filepath = sample.get("filepath") or ""
            if not filepath:
                continue
            try:
                rel = (base / filepath).relative_to(DATA_DIR).as_posix()
            except ValueError:
                continue
            polygons = [
                polygon
                for polygon in polygons_from_annotation(sample)
                if len(polygon.get("points") or []) >= 3
            ]
            if polygons:
                index[rel] = polygons
    _PKLOT_POLYGONS = index
    return index


def _kaggle_xml_index() -> dict[str, list[dict]]:
    global _KAGGLE_XML_POLYGONS
    if _KAGGLE_XML_POLYGONS is not None:
        return _KAGGLE_XML_POLYGONS
    import xml.etree.ElementTree as ET

    index: dict[str, list[dict]] = {}
    path = DATA_DIR / "kaggle" / "parking-space-detection" / "annotations.xml"
    if path.is_file():
        for image in ET.parse(path).getroot().iter("image"):
            name = Path(image.get("name") or "").name
            width = float(image.get("width") or 0)
            height = float(image.get("height") or 0)
            if not name or width <= 0 or height <= 0:
                continue
            polygons = []
            for polygon in image.findall("polygon"):
                points = []
                for pair in (polygon.get("points") or "").split(";"):
                    if "," not in pair:
                        continue
                    x_text, y_text = pair.split(",", 1)
                    try:
                        points.append([_clip01(float(x_text) / width), _clip01(float(y_text) / height)])
                    except ValueError:
                        continue
                item = _parking_polygon(points)
                if item:
                    polygons.append(item)
            if polygons:
                index[name] = polygons
    _KAGGLE_XML_POLYGONS = index
    return index


def _top_view_polygons(image_path: Path) -> list[dict]:
    label_path = image_path.with_suffix(".json")
    if not label_path.is_file():
        return []
    from PIL import Image

    raw = json.loads(label_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        return []
    polygons = []
    for label in raw.get("labels") or []:
        try:
            x1 = _clip01(float(label["x1"]) / width)
            y1 = _clip01(float(label["y1"]) / height)
            x2 = _clip01(float(label["x2"]) / width)
            y2 = _clip01(float(label["y2"]) / height)
        except (KeyError, TypeError, ValueError):
            continue
        name = str(label.get("name") or "").lower()
        tags = ["occupied"] if "occup" in name else ["empty"] if "free" in name else ["unknown"]
        item = _parking_polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], tags=tags)
        if item:
            polygons.append(item)
    return polygons


def _yolo_seg_polygons(text: str) -> list[dict]:
    polygons = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if len(coords) < 6 or len(coords) % 2:
            continue
        points = [
            [_clip01(coords[index]), _clip01(coords[index + 1])]
            for index in range(0, len(coords), 2)
        ]
        item = _parking_polygon(points)
        if item:
            polygons.append(item)
    return polygons


def _mmu_polygons(rel: str) -> list[dict]:
    path = Path(rel)
    label_path = None
    if path.parent.name == "images" and path.parent.parent.name in {"train", "test"}:
        label_path = path.parent.parent / "yolo_seg_labels" / f"{path.stem}.txt"
    elif path.parent.name == "images":
        for split in ("train", "test"):
            candidate = (
                DATA_DIR
                / "kaggle"
                / "mmu-parking-slots"
                / "main_data"
                / split
                / "yolo_seg_labels"
                / f"{path.stem}.txt"
            )
            if candidate.is_file():
                label_path = candidate
                break
    if label_path is None:
        return []
    file_path = label_path if label_path.is_absolute() else DATA_DIR / label_path
    if not file_path.is_file():
        return []
    return _yolo_seg_polygons(file_path.read_text(encoding="utf-8"))


def external_polygons(rel: str) -> list[dict]:
    """허깅페이스 PKLot와 케글 라벨을 색인 화면용 칸으로 읽는다."""
    rel = Path(rel).as_posix()
    if rel.startswith("huggingface/"):
        return list(_pklot_polygons_index().get(rel) or [])
    if rel.startswith("kaggle/parking-space-detection/"):
        return list(_kaggle_xml_index().get(Path(rel).name) or [])
    if rel.startswith("kaggle/parking-space-top-view/"):
        return _top_view_polygons(DATA_DIR / rel)
    if rel.startswith("kaggle/mmu-parking-slots/"):
        return _mmu_polygons(rel)
    return []


def has_source_annotation(rel: str) -> bool:
    """우리 색인 또는 허깅페이스·케글 원본 라벨이 있으면 참."""
    rel_path = Path(rel)
    if annot_path(rel_path).is_file():
        return True
    if shares_folder_annotation(rel) and (ANNOT_DIR / rel_path.with_suffix(".json")).is_file():
        return True
    posix = rel_path.as_posix()
    if posix.startswith("huggingface/"):
        return posix in _pklot_polygons_index()
    if posix.startswith("kaggle/parking-space-detection/"):
        return rel_path.name in _kaggle_xml_index()
    if posix.startswith("kaggle/parking-space-top-view/"):
        return (DATA_DIR / rel_path).with_suffix(".json").is_file()
    if posix.startswith("kaggle/mmu-parking-slots/"):
        return bool(_mmu_polygons(posix))
    return False


def load_annotation(rel: str) -> list[dict]:
    path = annot_path(Path(rel))
    if path.is_file():
        return _polygons_from_file(path)
    if shares_folder_annotation(rel):
        legacy = ANNOT_DIR / Path(rel).with_suffix(".json")
        if legacy.is_file() and legacy != path:
            return _polygons_from_file(legacy)
    return external_polygons(rel)


def polygons_from_folder_layout(rel: str, current: list[dict] | None = None) -> list[dict]:
    """같은 폴더가 공유하는 칸. current는 예전 호출과의 호환용이다."""
    del current
    if not shares_folder_annotation(rel):
        return []
    return load_annotation(rel)


def _shares_polygons(rel: str) -> bool:
    return shares_folder_annotation(rel)


def _legacy_jsons(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.json") if not path.name.startswith("_"))


def apply_folder_annotations(rel: str, polygons: list[dict]) -> list[str]:
    """같은 폴더 사진이 공유하는 색인 파일 하나에 칸을 저장한다."""
    if not shares_folder_annotation(rel) or not polygons:
        return []
    save_annotation(rel, polygons)
    folder = DATA_DIR / Path(rel).parent
    if not folder.is_dir():
        return [rel]
    return [
        path.relative_to(DATA_DIR).as_posix()
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def save_annotation(rel: str, polygons: list[dict], *, update_layout: bool = True) -> Path:
    """공유 폴더는 _layout.json 하나만 쓰고, 그 외에는 사진마다 JSON을 쓴다."""
    del update_layout
    if shares_folder_annotation(rel):
        path = annot_path(Path(rel))
        _write_shared(path, polygons)
        for legacy in _legacy_jsons(path.parent):
            legacy.unlink()
        return path

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
    return ap


def consolidate_cctv_annotations() -> list[str]:
    """cctv_image 하위 폴더의 사진별 JSON을 _layout.json 하나로 합친다."""
    root = ANNOT_DIR / "cctv_image"
    if not root.is_dir():
        return []
    notes = []
    for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
        per_image = _legacy_jsons(folder)
        if not per_image:
            continue
        layout = folder / "_layout.json"
        newest = max(per_image, key=lambda path: (path.stat().st_mtime, path.name))
        source = newest
        if layout.is_file():
            layout_polygons = _polygons_from_file(layout)
            newest_polygons = _polygons_from_file(newest)
            if layout_polygons and (
                not newest_polygons
                or _geom_key(layout_polygons) == _geom_key(newest_polygons)
                or layout.stat().st_mtime >= newest.stat().st_mtime
            ):
                source = layout
        polygons = _polygons_from_file(source)
        if source == layout and polygons:
            source_key = _geom_key(polygons)
            donors = []
            for path in per_image:
                donor = _polygons_from_file(path)
                if donor and _geom_key(donor) == source_key:
                    donors.append((path.stat().st_mtime, donor))
            if donors:
                donors.sort(key=lambda item: item[0])
                _copy_occupancy(polygons, donors[-1][1])
        if not polygons:
            notes.append(f"{folder.relative_to(root).as_posix()}: 칸 없음, 건너뜀")
            continue
        _write_shared(layout, polygons)
        for path in per_image:
            path.unlink()
        notes.append(
            f"{folder.relative_to(root).as_posix()}: {len(per_image)}개 통합, 칸 {len(polygons)}개, 기준 {source.name}"
        )
    return notes


def iter_annotation_sources() -> list[tuple[str, list[dict]]]:
    """학습용 (이미지 상대경로, 칸 목록). 공유 폴더는 사진 수만큼 같은 칸을 쓴다."""
    rows: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    cctv = ANNOT_DIR / "cctv_image"
    if cctv.is_dir():
        for layout in sorted(cctv.rglob("_layout.json")):
            if layout.parent == cctv:
                continue
            image_dir = DATA_DIR / layout.parent.relative_to(ANNOT_DIR)
            if not image_dir.is_dir():
                continue
            polygons = _polygons_from_file(layout)
            if not polygons:
                continue
            for image in sorted(image_dir.iterdir()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel = image.relative_to(DATA_DIR).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                rows.append((rel, polygons))
    if not ANNOT_DIR.is_dir():
        return rows
    for path in sorted(ANNOT_DIR.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            under = path.relative_to(cctv)
        except ValueError:
            under = None
        if under is not None and len(under.parts) >= 2:
            continue
        rel = path.relative_to(ANNOT_DIR).with_suffix(".jpg").as_posix()
        data = json.loads(path.read_text(encoding="utf-8"))
        rel = data.get("filepath") or data.get("image") or rel
        if rel in seen:
            continue
        polygons = _polygons_from_file(path)
        if not polygons:
            continue
        seen.add(rel)
        rows.append((rel, polygons))
    return rows


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
