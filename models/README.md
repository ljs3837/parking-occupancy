# 모델

가중치는 별도로 공유합니다. 모델 이름·버전·라이선스와 다운로드 방법을 기록하세요. 실제 가중치는 Git에서 제외됩니다.

## 앱 학습 결과 (기본 경로)

| 모델 | 저장 위치 | 설명 |
|---|---|---|
| CNN 분류 | `models/cnn_occupancy/best.pt` | 칸 crop → empty/occupied |
| YOLO 파인튜닝 | `models/yolo_occupancy/best.pt` | YOLOv8n empty/occupied 탐지 |

학습 UI: `python src/main.py` → **학습** 탭
