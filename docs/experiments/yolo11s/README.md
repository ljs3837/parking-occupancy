# YOLO11s 실험 자료

- [실험 결과와 한계](report.md)
- [앱 학습 기록](../training-runs/README.md)
- [1차 모델 CCTV 적용 이미지](cctv_stage1.jpg)
- [추가 학습 모델 CCTV 적용 이미지](cctv_stage2.jpg)
- [설정별 수치](combined_comparison_stage2.json)
- [모델 유지 판단](model_selection.json)
- [원본·분리·제외·이미지 해시](dataset_manifest.json)
- [실행 환경](environment.json)
- [1차 학습 기록](training_stage1.csv), [추가 학습 기록](training_stage2.csv)

기본 모델은 models/yolo_spaces/yolo11s_seg_1.pt이다. 추가 학습은 새 카메라 진단에서 악화되어 기본 적용하지 않았다.
원래 로컬 yolo_4.pt는 main의 다른 yolo_4.pt와 구분해 yolov8_initial_local.pt로 보존했다.

main의 폴더 정리에 맞춰 자체 이미지 20장과 색인은 data/cctv/jongsu 및 data/annotations/cctv/jongsu에 있다.
다른 CNR·NDIS 카메라 원본은 저장소에 포함되지 않는 로컬 데이터이므로 완전히 같은 데이터로 재학습하려면 별도로 준비해야 한다.
전체 생성 데이터·중복 체크포인트는 outputs와 models/_runs에 로컬로 보존한다.
