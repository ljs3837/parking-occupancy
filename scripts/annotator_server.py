"""주차면 다각형 색인 UI 서버."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANNOT_DIR = ROOT / "data" / "annotations"
STATIC_DIR = ROOT / "frontend" / "annotator"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="Parking Annotator")
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

_IMAGE_CACHE: list[str] | None = None


def _all_image_rels() -> list[str]:
    global _IMAGE_CACHE
    if _IMAGE_CACHE is not None:
        return _IMAGE_CACHE
    items: list[str] = []
    annot_root = ANNOT_DIR.resolve()
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
    _IMAGE_CACHE = items
    return items


def _safe_rel(path: str) -> Path:
    rel = Path(path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(400, "잘못된 경로")
    full = (DATA_DIR / rel).resolve()
    if not str(full).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(400, "경로 이탈")
    return rel


def _annot_path(rel: Path) -> Path:
    # data/foo/bar.jpg -> data/annotations/foo/bar.json
    return ANNOT_DIR / rel.with_suffix(".json")


def _list_images(q: str = "", limit: int = 200, offset: int = 0) -> tuple[list[dict], int]:
    q = (q or "").strip().lower()
    matched = [rel for rel in _all_image_rels() if (not q or q in rel.lower())]
    total = len(matched)
    page = matched[offset : offset + limit]
    items = [
        {
            "path": rel,
            "name": Path(rel).name,
            "annotated": _annot_path(Path(rel)).exists(),
        }
        for rel in page
    ]
    return items, total


@app.post("/api/images/refresh")
def refresh_images():
    global _IMAGE_CACHE
    _IMAGE_CACHE = None
    n = len(_all_image_rels())
    return {"ok": True, "total": n}


class PolygonIn(BaseModel):
    id: str
    points: list[list[float]] = Field(..., description="정규화 [x,y] 0~1")
    tags: list[str] = Field(default_factory=list)
    label: str = "parking_space"


class AnnotationPayload(BaseModel):
    image: str
    polygons: list[PolygonIn] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health():
    return {"ok": True, "data_dir": str(DATA_DIR)}


@app.get("/api/images")
def list_images(
    q: str = "",
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    items, total = _list_images(q=q, limit=limit, offset=offset)
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@app.get("/api/image")
def get_image(path: str):
    rel = _safe_rel(path)
    full = DATA_DIR / rel
    if not full.is_file():
        raise HTTPException(404, "이미지 없음")
    return FileResponse(full)


@app.get("/api/annotation")
def get_annotation(path: str):
    rel = _safe_rel(path)
    ap = _annot_path(rel)
    if not ap.exists():
        return {"image": rel.as_posix(), "polygons": [], "meta": {}}
    data = json.loads(ap.read_text(encoding="utf-8"))
    return data


@app.post("/api/annotation")
def save_annotation(payload: AnnotationPayload):
    rel = _safe_rel(payload.image)
    full = DATA_DIR / rel
    if not full.is_file():
        raise HTTPException(404, "이미지 없음")
    for poly in payload.polygons:
        for x, y in poly.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise HTTPException(400, "좌표는 0~1 정규화여야 합니다")
    ap = _annot_path(rel)
    ap.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "image": rel.as_posix(),
        "polygons": [p.model_dump() for p in payload.polygons],
        "meta": payload.meta,
    }
    ap.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "saved": ap.relative_to(ROOT).as_posix()}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR / "static")), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False)
