"""카메라 관측을 칸별 통합 상태로 유지한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def merge_space_states(by_camera: dict[int, dict[str, str]]) -> dict[str, int]:
    """같은 space_id는 카메라가 달라도 한 칸이다. 한 화면이라도 점유면 점유."""
    buckets: dict[str, set[str]] = {}
    for states in by_camera.values():
        for space_id, state in (states or {}).items():
            key = str(space_id).strip()
            if not key:
                continue
            buckets.setdefault(key, set()).add(str(state))
    occupied = vacant = unknown = 0
    for states in buckets.values():
        if "occupied" in states:
            occupied += 1
        elif "vacant" in states:
            vacant += 1
        else:
            unknown += 1
    return {
        "occupied": occupied,
        "vacant": vacant,
        "unknown": unknown,
        "total": len(buckets),
        "free": vacant,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SpaceFusion:
    """직전 상태를 유지하는 단순 fusion (단일 캠 기준)."""

    spaces: dict[str, dict] = field(default_factory=dict)

    def update(self, camera_id: str, observations: list[dict], *, observed_at: str | None = None) -> list[dict]:
        stamp = observed_at or _utc_now()
        for obs in observations:
            space_id = str(obs.get("space_id") or "")
            if not space_id:
                continue
            state = str(obs.get("state") or "unknown")
            # 가림·미확인은 새 관측으로 확정하지 않고 유지
            if state == "unknown":
                current = self.spaces.get(space_id)
                if current is not None:
                    current["is_held"] = True
                    current["updated_at"] = stamp
                continue
            self.spaces[space_id] = {
                "space_id": space_id,
                "state": state,
                "updated_at": stamp,
                "last_observed_at": stamp,
                "is_held": False,
                "source_cameras": [camera_id],
                "confidence": obs.get("confidence"),
                "points": obs.get("points") or [],
            }
        return self.snapshot()

    def snapshot(self) -> list[dict]:
        return [dict(item) for item in self.spaces.values()]

    def counts(self) -> dict[str, int]:
        occupied = sum(1 for s in self.spaces.values() if s.get("state") == "occupied")
        vacant = sum(1 for s in self.spaces.values() if s.get("state") == "vacant")
        unknown = sum(1 for s in self.spaces.values() if s.get("state") == "unknown")
        return {
            "occupied": occupied,
            "vacant": vacant,
            "unknown": unknown,
            "total": len(self.spaces),
            "free": vacant,
        }
