"""차량 박스와 고정 슬롯 폴리곤으로 점유를 판정한다."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 슬롯 면적 대비 차량 겹침 비율
OCCUPANCY_RATIO = 0.18
# 같은 raw 상태가 이 횟수 연속이어야 확정 상태 전환
HOLD_HITS = 2


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_poly_to_rect(points: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Sutherland–Hodgman으로 폴리곤을 축정렬 박스에 자른다."""
    x1, y1, x2, y2 = xyxy

    def clip_edge(poly: list[tuple[float, float]], edge: str) -> list[tuple[float, float]]:
        if not poly:
            return []
        out: list[tuple[float, float]] = []
        prev = poly[-1]
        for cur in poly:
            px, py = prev
            cx, cy = cur
            if edge == "left":
                pin, cin = px >= x1, cx >= x1
                ix = x1
                iy = py + (cy - py) * (x1 - px) / (cx - px) if cx != px else py
            elif edge == "right":
                pin, cin = px <= x2, cx <= x2
                ix = x2
                iy = py + (cy - py) * (x2 - px) / (cx - px) if cx != px else py
            elif edge == "top":
                pin, cin = py >= y1, cy >= y1
                iy = y1
                ix = px + (cx - px) * (y1 - py) / (cy - py) if cy != py else px
            else:
                pin, cin = py <= y2, cy <= y2
                iy = y2
                ix = px + (cx - px) * (y2 - py) / (cy - py) if cy != py else px
            if cin:
                if not pin:
                    out.append((ix, iy))
                out.append(cur)
            elif pin:
                out.append((ix, iy))
            prev = cur
        return out

    poly = [(float(x), float(y)) for x, y in points]
    for edge in ("left", "right", "top", "bottom"):
        poly = clip_edge(poly, edge)
        if not poly:
            return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(poly, dtype=np.float64)


def overlap_ratio(space_points_norm: list, vehicle_xyxy: tuple[int, int, int, int], width: int, height: int) -> float:
    """차량 박스 ∩ 슬롯 / 슬롯 면적."""
    if width <= 0 or height <= 0 or len(space_points_norm) < 3:
        return 0.0
    pts = np.asarray(
        [[float(x) * width, float(y) * height] for x, y in space_points_norm],
        dtype=np.float64,
    )
    slot_area = _polygon_area(pts)
    if slot_area <= 1e-6:
        return 0.0
    clipped = _clip_poly_to_rect(pts, tuple(float(v) for v in vehicle_xyxy))
    if len(clipped) < 3:
        return 0.0
    return min(1.0, _polygon_area(clipped) / slot_area)


def assign_occupancy(
    spaces: list[dict],
    vehicle_boxes: list[dict],
    *,
    width: int,
    height: int,
    ratio_threshold: float = OCCUPANCY_RATIO,
) -> list[dict]:
    """칸마다 occupied/vacant. 한 차는 겹침이 가장 큰 칸에만 배정."""
    n = len(spaces)
    best_for_vehicle: list[tuple[int, float, float]] = []
    for box in vehicle_boxes:
        xyxy = box.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        scores = [
            overlap_ratio(space.get("points") or [], tuple(xyxy), width, height)
            for space in spaces
        ]
        if not scores:
            continue
        best_i = int(max(range(len(scores)), key=scores.__getitem__))
        best_for_vehicle.append((best_i, scores[best_i], float(box.get("conf") or 0.0)))

    occupied: dict[int, float] = {}
    for space_i, score, conf in best_for_vehicle:
        if score < ratio_threshold:
            continue
        prev = occupied.get(space_i)
        if prev is None or conf > prev:
            occupied[space_i] = conf

    results = []
    for index, space in enumerate(spaces):
        space_id = str(space.get("space_id") or f"S{index + 1:03d}")
        if index in occupied:
            results.append(
                {
                    "space_id": space_id,
                    "state": "occupied",
                    "confidence": occupied[index],
                    "points": space.get("points") or [],
                }
            )
        else:
            results.append(
                {
                    "space_id": space_id,
                    "state": "vacant",
                    "confidence": 1.0,
                    "points": space.get("points") or [],
                }
            )
    return results


@dataclass
class OccupancyFilter:
    """연속 HOLD_HITS번 같아야 상태 전환."""

    hits_needed: int = HOLD_HITS
    _confirmed: dict[str, str] = field(default_factory=dict)
    _pending: dict[str, tuple[str, int]] = field(default_factory=dict)

    def update(self, observations: list[dict]) -> list[dict]:
        out = []
        for obs in observations:
            space_id = str(obs["space_id"])
            raw = str(obs.get("state") or "unknown")
            confirmed = self._confirmed.get(space_id, "unknown")
            if raw == confirmed:
                self._pending.pop(space_id, None)
                state = confirmed
            else:
                pending_state, count = self._pending.get(space_id, (raw, 0))
                if pending_state != raw:
                    count = 0
                count += 1
                if count >= self.hits_needed:
                    self._confirmed[space_id] = raw
                    self._pending.pop(space_id, None)
                    state = raw
                else:
                    self._pending[space_id] = (raw, count)
                    state = confirmed if confirmed != "unknown" else raw
            out.append({**obs, "state": state})
        return out


def summarize_occupancy(states: list[dict]) -> tuple[int, int, int]:
    """occupied, vacant, unknown 개수."""
    occupied = sum(1 for s in states if s.get("state") == "occupied")
    vacant = sum(1 for s in states if s.get("state") == "vacant")
    unknown = sum(1 for s in states if s.get("state") == "unknown")
    return occupied, vacant, unknown
