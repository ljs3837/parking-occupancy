# 모델

학습된 가중치(`.pt`)는 Git에 포함합니다. `models/_runs`의 학습 로그와 복사본은 제외합니다.

## 앱 학습 결과 (기본 경로)

| 모델 | 저장 위치 | 설명 |
|---|---|---|
| YOLO 주차칸 | `models/yolo_spaces/*.pt` | 실험별 데이터·설정은 모델 옆 JSON과 학습 로그 참고 |
| YOLO11s 1차 | `models/yolo_spaces/yolo11s_seg_1.pt` | 공식 사전학습 모델에서 25에포크 실험 |
| YOLO11s 추가 | `models/yolo_spaces/yolo11s_seg_2.pt` | 1차 최적 가중치에서 특징 추출부까지 추가 미세조정 |

학습 UI: `python src/main.py` → **학습** 탭

주차 칸 학습 기본 가중치는 `yolo11s-seg.pt`입니다. 실험 모델의 데이터와 설정은 옆 JSON에 기록합니다. [실험 문서](../docs/parking-line-experiments.md)
