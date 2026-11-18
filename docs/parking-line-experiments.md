# YOLO11s 주차 칸 학습 실험

원본 색인은 보존하고 학습용 복사본에서 좌표와 확대 영역을 변환한다.
주차선 픽셀 자체가 아니라 주차 칸의 전체 영역을 학습한다.

## 데이터

- 학습 원본 25장, 주차 칸 546개. 추가 11개 카메라 그룹의 대표 이미지 포함.
- 확대 영역·전체 프레임·부분 가림 입력 58장과 검토한 배경 2장, 합계 60장.
- 가림 이미지는 전체 칸 정답 좌표를 유지한다. 원본을 새로 색인한 것은 아니다.
- 저해상도 원본 2장과 완전한 네 모서리를 확정하기 어려운 추가 카메라 7개 그룹은 공통 학습 데이터에서 제외했다.
- 검증 원본 4장·76칸은 이전과 동일하다. 검증에 정답 기반 확대나 가림 증강을 적용하지 않는다.
- 검증 목록은 configs/parking_line_split.json, 원본·제외 이유·변형은 outputs/train/line_improvement/dataset_v3/manifest.json에 기록한다.

## 1차 학습

```powershell
$env:PYTHONPATH = "src"
.venv/Scripts/python.exe scripts/train_parking_improved.py --task segment --model yolo11s-seg.pt --epochs 25 --batch 4 --imgsz 640 --freeze 10
```

공식 사전학습 모델에서 25에포크, AdamW 0.0003, 앞 10개 레이어 고정으로 시작했다.
모델은 models/yolo_spaces/yolo11s_seg_1.pt에 저장했다.

## 추가 미세조정

```powershell
.venv/Scripts/python.exe scripts/train_parking_improved.py --task segment --model models/yolo_spaces/yolo11s_seg_1.pt --epochs 10 --batch 4 --imgsz 640 --freeze 0 --lr 0.0001 --warmup-bias-lr 0.001
```

1차 실험의 최적 모델에서 특징 추출부까지 미세조정한다. 옵티마이저는 새로 구성하며, 이전 가중치는 보존한다.
새 모델은 models/yolo_spaces/yolo11s_seg_2.pt로 저장한다. 검증 결과가 더 나은 모델을 앱에 적용한다.

## 비교와 적용

신뢰도 0.35와 원본 해상도 다각형 IoU 0.5 이상에서 정밀도·재현율·F1을 비교한다.
정답 하나를 여러 예측에 중복 매칭하지 않는다. mAP 및 분류 정확도와 구분한다.
640/960 입력, 타일 확대, 네 꼭짓점 근사와 직선 보정을 각각 비교한다.
검증 4장으로 설정까지 선택하므로 최종 일반화 성능으로 해석할 수 없다.

거친 영역을 네 꼭짓점으로 근사하는 enclosing 옵션은 OpenCV approxPolyN을 사용한다.
CCTV 원본에 직각을 강제하지 않는다. 해당 API가 없는 환경에서는 기존 근사 방식으로 대체하므로 결과가 달라질 수 있다.
정확한 실행 환경은 outputs/train/line_improvement/environment.json에 저장한다.

선택된 설정은 모델 옆 JSON의 inference_imgsz, tiled, snap_lines, quad_fit에 기록한다.
앱을 재실행하면 models/preferred_parking_model.json의 모델을 기본 선택한다.
실행 중인 앱에서는 모델을 직접 선택한다.

## 평가 한계

학습에서 제외한 카메라 대표 이미지 3장·136칸은 별도 진단에 사용했다.
1차 모델은 19칸을 맞췄고 오탐 3개였다. 이는 새 주차장 전체에 대한 보장이 아니다.
추가 학습 이후 같은 카메라를 반복 평가하면 진단 비교로 취급하고 새로운 독립 테스트라고 부르지 않는다.
관제 영상 4개의 샘플은 정답이 없으므로 시각적 검토에만 사용한다.

추가 10에포크 실험은 완료했다. 별도 카메라에서 정답 검출 19→16칸, 오탐 3→9개로 악화되어 1차 모델을 기본으로 유지한다. 최종 기록은 outputs/train/line_improvement/report.md와 model_selection.json에 보존한다.
