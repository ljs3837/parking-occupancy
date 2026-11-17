# 모델

학습된 가중치(`.pt`)는 Git에 포함합니다. `models/_runs`의 학습 로그와 복사본은 제외합니다.

## 앱 학습 결과 (기본 경로)

| 모델 | 저장 위치 | 설명 |
|---|---|---|
| CNN 분류 | `models/cnn_occupancy/best.pt` | 칸 crop → empty/occupied |
| YOLO 파인튜닝 | `models/yolo_occupancy/best.pt` | YOLOv8n empty/occupied 탐지 |

학습 UI: `python src/main.py` → **학습** 탭
