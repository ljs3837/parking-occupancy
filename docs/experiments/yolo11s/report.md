# YOLO11s 주차 칸 탐지 실험 결과

## 학습

이번 단계 모델: yolo11s_seg_1.pt. 학습 횟수: 25에포크. 시작 가중치: yolo11s-seg.pt.
원본 학습 25장·546칸에서 확대·가림 입력 58장을 만들고, 검토한 배경 2장을 추가했다. 검증 원본은 이전과 같은 4장·76칸이다.
freeze=10, 학습 입력 640, 배치 4, AdamW 학습률 0.0003, CPU. 기존 모델은 보존했다.

## 같은 검증 사진에서 비교

각 행은 confidence 0.35, 원본 해상도 다각형 IoU >= 0.5에서 계산했다. mAP나 일반적인 분류 정확도가 아니다.

| 모델 | 입력 | 타일 | 처리/사각형 피팅 | 정밀도 | 재현율 | F1 | 맞춘 칸/76 | CPU ms/frame |
|---|---:|---|---|---:|---:|---:|---:|---:|
| yolov8_initial_local.pt | 640 | False | raw/conservative | 24.0% | 7.9% | 0.119 | 6/76 | 62 |
| yolov8_initial_local.pt | 640 | False | quad_only/conservative | 30.8% | 5.3% | 0.090 | 4/76 | 62 |
| yolov8_initial_local.pt | 640 | False | refined/conservative | 30.8% | 5.3% | 0.090 | 4/76 | 62 |
| yolov8_initial_local.pt | 960 | False | raw/conservative | 11.1% | 2.6% | 0.043 | 2/76 | 93 |
| yolov8_initial_local.pt | 960 | False | quad_only/conservative | 0.0% | 0.0% | 0.000 | 0/76 | 93 |
| yolov8_initial_local.pt | 960 | False | refined/conservative | 0.0% | 0.0% | 0.000 | 0/76 | 93 |
| yolov8_initial_local.pt | 640 | True | raw/conservative | 17.4% | 10.5% | 0.131 | 8/76 | 194 |
| yolov8_initial_local.pt | 640 | True | quad_only/conservative | 14.3% | 5.3% | 0.077 | 4/76 | 194 |
| yolov8_initial_local.pt | 640 | True | refined/conservative | 14.3% | 5.3% | 0.077 | 4/76 | 194 |
| yolo_5.pt | 640 | False | raw/conservative | 39.3% | 14.5% | 0.212 | 11/76 | 47 |
| yolo_5.pt | 640 | False | quad_only/conservative | 52.9% | 11.8% | 0.194 | 9/76 | 47 |
| yolo_5.pt | 640 | False | refined/conservative | 52.9% | 11.8% | 0.194 | 9/76 | 47 |
| yolo_5.pt | 960 | False | raw/conservative | 50.0% | 19.7% | 0.283 | 15/76 | 86 |
| yolo_5.pt | 960 | False | quad_only/conservative | 61.1% | 14.5% | 0.234 | 11/76 | 86 |
| yolo_5.pt | 960 | False | refined/conservative | 55.6% | 13.2% | 0.213 | 10/76 | 86 |
| yolo_5.pt | 640 | True | raw/conservative | 34.5% | 26.3% | 0.299 | 20/76 | 209 |
| yolo_5.pt | 640 | True | quad_only/conservative | 50.0% | 22.4% | 0.309 | 17/76 | 209 |
| yolo_5.pt | 640 | True | refined/conservative | 50.0% | 22.4% | 0.309 | 17/76 | 209 |
| yolo11s_seg_1.pt | 640 | False | raw/conservative | 60.0% | 23.7% | 0.340 | 18/76 | 107 |
| yolo11s_seg_1.pt | 640 | False | quad_only/conservative | 41.2% | 9.2% | 0.151 | 7/76 | 107 |
| yolo11s_seg_1.pt | 640 | False | refined/conservative | 47.1% | 10.5% | 0.172 | 8/76 | 107 |
| yolo11s_seg_1.pt | 960 | False | raw/conservative | 55.0% | 28.9% | 0.379 | 22/76 | 217 |
| yolo11s_seg_1.pt | 960 | False | quad_only/conservative | 75.0% | 19.7% | 0.312 | 15/76 | 217 |
| yolo11s_seg_1.pt | 960 | False | refined/conservative | 75.0% | 19.7% | 0.312 | 15/76 | 217 |
| yolo11s_seg_1.pt | 640 | True | raw/conservative | 50.0% | 42.1% | 0.457 | 32/76 | 435 |
| yolo11s_seg_1.pt | 640 | True | quad_only/conservative | 47.2% | 22.4% | 0.304 | 17/76 | 435 |
| yolo11s_seg_1.pt | 640 | True | refined/conservative | 50.0% | 23.7% | 0.321 | 18/76 | 435 |
| yolo11s_seg_1.pt | 640 | False | raw/enclosing | 60.0% | 23.7% | 0.340 | 18/76 | 118 |
| yolo11s_seg_1.pt | 640 | False | quad_only/enclosing | 56.0% | 18.4% | 0.277 | 14/76 | 118 |
| yolo11s_seg_1.pt | 640 | False | refined/enclosing | 56.0% | 18.4% | 0.277 | 14/76 | 118 |
| yolo11s_seg_1.pt | 960 | False | raw/enclosing | 55.0% | 28.9% | 0.379 | 22/76 | 234 |
| yolo11s_seg_1.pt | 960 | False | quad_only/enclosing | 64.5% | 26.3% | 0.374 | 20/76 | 234 |
| yolo11s_seg_1.pt | 960 | False | refined/enclosing | 64.5% | 26.3% | 0.374 | 20/76 | 234 |
| yolo11s_seg_1.pt | 640 | True | raw/enclosing | 50.0% | 42.1% | 0.457 | 32/76 | 458 |
| yolo11s_seg_1.pt | 640 | True | quad_only/enclosing | 49.1% | 35.5% | 0.412 | 27/76 | 458 |
| yolo11s_seg_1.pt | 640 | True | refined/enclosing | 49.1% | 35.5% | 0.412 | 27/76 | 458 |
| yolo11s_seg_2.pt | 640 | False | raw/enclosing | 55.1% | 35.5% | 0.432 | 27/76 | 130 |
| yolo11s_seg_2.pt | 640 | False | quad_only/enclosing | 55.3% | 34.2% | 0.423 | 26/76 | 130 |
| yolo11s_seg_2.pt | 640 | False | refined/enclosing | 55.3% | 34.2% | 0.423 | 26/76 | 130 |
| yolo11s_seg_2.pt | 960 | False | raw/enclosing | 45.5% | 39.5% | 0.423 | 30/76 | 242 |
| yolo11s_seg_2.pt | 960 | False | quad_only/enclosing | 45.5% | 32.9% | 0.382 | 25/76 | 242 |
| yolo11s_seg_2.pt | 960 | False | refined/enclosing | 43.6% | 31.6% | 0.366 | 24/76 | 242 |
| yolo11s_seg_2.pt | 640 | True | raw/enclosing | 35.9% | 60.5% | 0.451 | 46/76 | 511 |
| yolo11s_seg_2.pt | 640 | True | quad_only/enclosing | 36.7% | 57.9% | 0.449 | 44/76 | 511 |
| yolo11s_seg_2.pt | 640 | True | refined/enclosing | 35.8% | 56.6% | 0.439 | 43/76 | 511 |

선택 기준: 정밀도 0% 이상인 설정 중 F1 최대. 검증에서 정한 실험 기준이며 실사용 품질 보장이 아니다.

## 적용한 설정

모델: `models/yolo_spaces/yolo11s_seg_1.pt`
입력: 640, 타일: True, 직선 보정: False, 사각형 피팅: enclosing.
모델 옆 JSON에 저장해 앱에서 이 모델을 선택하면 자동 적용된다. 사각형 출력은 유지한다.

## 한계

검증 4장으로 학습 체크포인트와 추론 설정을 선택했다. 독립 테스트 성능이나 새로운 주차장 전체의 정확도로 해석하면 안 된다.
모델 변경, 추가 카메라, 확대, 학습률, 학습 횟수 등이 함께 바뀌었으므로 어느 한 요소의 효과라고 단정할 수 없다.
주차선 픽셀을 직접 색인한 모델이 아니라 주차 칸 영역 분할 모델이다. 완전히 가려진 칸 복원 성능은 별도로 입증되지 않았다.
타일은 정답 없이 화면을 확대 분석하고 원본 좌표로 복원해 중복을 제거한다. CPU에서는 추론 비용이 증가한다.
enclosing은 볼록 윤곽을 네 꼭짓점으로 근사한다. 원본 CCTV 화면에 직각을 강제하지 않는다. OpenCV approxPolyN이 없으면 기존 근사 방식으로 대체된다.
선택 이후 학습에서 제외한 카메라를 별도 평가한다. 해당 결과로 다시 설정을 맞추지 않는다.

비교 이미지에서는 노란색 정답과 AI 원본·사각형 단순화·직선 보정 결과를 함께 확인할 수 있다.

## 추가 학습의 최종 판단

1차 최적 가중치에서 특징 추출부까지 풀어 학습률 0.0001로 10에포크 추가 학습했다.
같은 검증 4장에서는 전체 화면 설정의 F1이 소폭 개선됐지만, 별도 카메라와 실제 영상 검토에서는 개선되지 않았다.
기본 모델은 yolo11s_seg_1.pt로 유지하고 yolo11s_seg_2.pt는 비교용으로 보존한다.

| 별도 카메라 3장·136칸 | 정답 검출 | 오탐 | 누락 | 정밀도 | 재현율 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1차 모델 | 19 | 3 | 117 | 86.4% | 14.0% | 0.241 |
| 추가 학습 모델 | 16 | 9 | 120 | 64.0% | 11.8% | 0.199 |

이 카메라 결과도 최종 모델 유지 판단에 참고했으므로 독립 테스트 성능이라고 부르지 않는다.
관제 영상에는 정답이 없어 정확도를 계산하지 않았다. 차량 윤곽 추종, 인접 칸 병합 및 누락이 남는다.
새 주차장의 주차선을 안정적으로 자동 인식하는 실사용 수준은 아직 아니다.
다음 개선에는 실제 적용 CCTV의 다양한 프레임에 대한 주차 칸·선 정답 보강이 필요하다.
