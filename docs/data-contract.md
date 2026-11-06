# 입출력 규격 초안

시각은 UTC ISO 8601 문자열입니다. 구현 전 팀 합의가 필요합니다.

## 카메라별 관측

```json
{
  "camera_id": "CAM_01",
  "region_id": "A",
  "observed_at": "2026-11-06T00:00:00Z",
  "state": "occupied",
  "visibility": "visible",
  "confidence": 0.95,
  "track_id": "CAM_01:42"
}
```

- state: occupied / vacant / unknown
- visibility: visible / occluded / unavailable
- confidence: 0~1 또는 판정 불가 시 null
- track_id: 카메라 내 추적 ID, 없거나 판정 불가이면 null
- region_id: 해당 카메라 영상의 영역 ID
- 가려진 관측은 unknown으로 전달하고 유지 상태를 새 관측으로 전달하지 않습니다.

## 통합 상태

```json
{
  "space_id": "SPACE_001",
  "state": "occupied",
  "updated_at": "2026-11-06T00:00:05Z",
  "last_observed_at": "2026-11-06T00:00:00Z",
  "is_held": true,
  "source_cameras": ["CAM_01"]
}
```

- updated_at: 통합 계산 시각
- last_observed_at: 실제 확인한 마지막 시각, 이력이 없으면 null
- is_held: 관측하지 못해 직전 상태를 유지하는지 여부
- source_cameras: 마지막 상태를 뒷받침한 카메라 목록
- 초기 상태는 unknown이며 빈자리 수에 넣지 않습니다.
- 대수는 고유 space_id 기준으로 집계합니다.
- 카메라 간 track_id는 같은 차량임을 보장하지 않습니다.

입출차 이벤트 형식과 관측 충돌 정책은 추후 확정합니다.
