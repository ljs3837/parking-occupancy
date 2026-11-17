# 모델

학습된 가중치(`.pt`)는 Git에 포함합니다. `models/_runs`의 학습 로그와 복사본은 제외합니다.

## 앱 학습 결과 (기본 경로)

| 모델 | 저장 위치 | 설명 |
|---|---|---|
| YOLO 주차칸 | `models/yolo_spaces/yolo_*.pt` | 로컬 색인 + PKLot 전체로 학습한 세그멘테이션 |

학습 UI: `python src/main.py` → **학습** 탭
