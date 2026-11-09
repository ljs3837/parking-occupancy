from __future__ import annotations

import json
from pathlib import Path

from parking.app.paths import ANNOT_DIR, DATA_DIR, IMAGE_EXTS


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


def load_annotation(rel: str) -> list[dict]:
    ap = annot_path(Path(rel))
    if not ap.exists():
        return []
    data = json.loads(ap.read_text(encoding="utf-8"))
    return data.get("polygons") or []


def save_annotation(rel: str, polygons: list[dict]) -> Path:
    ap = annot_path(Path(rel))
    ap.parent.mkdir(parents=True, exist_ok=True)
    payload = {"image": rel, "polygons": polygons, "meta": {}}
    ap.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ap


def occupancy_of(tags: list[str]) -> str:
    if "occupied" in tags:
        return "occupied"
    if "empty" in tags or "not occupied" in tags:
        return "empty"
    if "unknown" in tags:
        return "unknown"
    return "unknown"
