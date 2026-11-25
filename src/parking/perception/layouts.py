"""관제용 주차칸 레이아웃. 색인(annotations)과 경로를 분리한다."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from parking.app.paths import LAYOUTS_DIR, OBSERVE_DIR


def stream_id_from_video(path: str | Path) -> str:
    """관제 영상 경로 → 레이아웃 파일 stem."""
    path = Path(path)
    try:
        rel = path.resolve().relative_to(OBSERVE_DIR.resolve()).as_posix()
    except ValueError:
        rel = path.name
    stem = Path(rel).with_suffix("").as_posix()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return slug or "stream"


def layout_path(stream_id: str) -> Path:
    return LAYOUTS_DIR / f"{stream_id}.json"


def assign_class_names(spaces: list[dict], stream_id: str) -> list[dict]:
    """카메라별로 1-1, 1-2처럼 붙여 다른 캠의 1번과 겹치지 않게 한다."""
    prefix = stream_id.strip() or "cam"

    def center(space: dict) -> tuple[float, float]:
        points = space.get("points") or []
        if not points:
            return (0.0, 0.0)
        ys = [float(p[1]) for p in points]
        xs = [float(p[0]) for p in points]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    ordered = sorted(spaces, key=center)
    named = []
    for index, space in enumerate(ordered, start=1):
        named.append({**space, "space_id": f"{prefix}-{index}"})
    return named


def delete_layout(stream_id: str) -> bool:
    path = layout_path(stream_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def spaces_from_polygons(polygons: list[dict]) -> list[dict]:
    """색인/모델 폴리곤 → 관제 spaces 목록."""
    spaces = []
    for index, polygon in enumerate(polygons, start=1):
        points = [[float(x), float(y)] for x, y in polygon.get("points") or []]
        if len(points) < 3:
            continue
        space_id = str(polygon.get("id") or polygon.get("space_id") or f"S{index:03d}")
        spaces.append(
            {
                "space_id": space_id,
                "points": points,
                "label": str(polygon.get("label") or "parking_space"),
            }
        )
    return spaces


def save_layout(
    stream_id: str,
    spaces: list[dict],
    *,
    source_video: str | None = None,
    confirmed: bool = True,
) -> Path:
    LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for index, space in enumerate(spaces, start=1):
        points = [[float(x), float(y)] for x, y in space.get("points") or []]
        if len(points) < 3:
            continue
        cleaned.append(
            {
                "space_id": str(space.get("space_id") or f"S{index:03d}"),
                "points": points,
                "label": str(space.get("label") or "parking_space"),
            }
        )
    payload = {
        "stream_id": stream_id,
        "source_video": source_video,
        "confirmed": bool(confirmed),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spaces": cleaned,
    }
    path = layout_path(stream_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_payload(stream_id: str) -> dict | None:
    path = layout_path(stream_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_layout(stream_id: str, *, confirmed_only: bool = False) -> list[dict]:
    payload = _read_payload(stream_id)
    if not payload:
        return []
    if confirmed_only and not payload.get("confirmed"):
        return []
    spaces = payload.get("spaces")
    if not isinstance(spaces, list):
        polygons = payload.get("polygons")
        if isinstance(polygons, list):
            return spaces_from_polygons(polygons)
        return []
    return spaces_from_polygons(spaces)


def is_layout_confirmed(path: str | Path) -> bool:
    stream_id = stream_id_from_video(path)
    payload = _read_payload(stream_id)
    return bool(payload and payload.get("confirmed") and payload.get("spaces"))


def load_layout_for_video(path: str | Path, *, confirmed_only: bool = False) -> tuple[str, list[dict]]:
    stream_id = stream_id_from_video(path)
    return stream_id, load_layout(stream_id, confirmed_only=confirmed_only)
