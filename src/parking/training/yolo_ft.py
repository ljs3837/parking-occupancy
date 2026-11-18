"""주차칸 좌표(단일 클래스) YOLO 파인튜닝. 점유는 차량 YOLO가 따로 판정한다."""

from __future__ import annotations

import shutil
from pathlib import Path

from parking.common.device import device_label, use_amp, yolo_device
from parking.training.prepare import MODELS_DIR, build_yolo_dataset, next_model_path

OUT_DIR = MODELS_DIR / "yolo_spaces"
# 주차 CCTV는 위아래가 있는 사선 시점이라, 매 사진마다 -15~15도만 돌린다.
ROTATION_DEGREES = 15.0


def train_yolo(
    *,
    epochs: int = 5,
    batch: int = 16,
    imgsz: int = 640,
    model_name: str = "yolov8n-seg.pt",
    log=print,
    should_stop=None,
) -> Path:
    device = yolo_device()
    amp = use_amp(device)
    log(f"[YOLO] device={device_label()}  amp={amp}")
    yaml_path = build_yolo_dataset(log)
    log(f"[YOLO] data={yaml_path}")
    log(f"[YOLO] base={model_name}  epochs={epochs}  batch={batch}  degrees=±{ROTATION_DEGREES:g}")

    if should_stop and should_stop():
        raise RuntimeError("학습 시작 전 중지됨")

    from ultralytics import YOLO

    dest = next_model_path(OUT_DIR, "yolo")
    run_name = dest.stem
    log(f"[YOLO] 저장 파일: {dest.name}")
    model = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(MODELS_DIR / "_runs"),
        name=run_name,
        exist_ok=False,
        verbose=True,
        workers=0,
        device=device,
        amp=amp,
        degrees=ROTATION_DEGREES,
        patience=max(3, epochs),
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        cand = MODELS_DIR / "_runs" / run_name / "weights" / "best.pt"
        best = cand if cand.is_file() else best
    if best.is_file():
        shutil.copy2(best, dest)
        log(f"[YOLO] best 저장 → {dest}")
    else:
        raise RuntimeError(f"YOLO best.pt를 찾지 못했습니다: {best}")

    dest.with_suffix(".txt").write_text(
        f"YOLO segmentation ({model_name})\n"
        "labels: polygon parking_space (occupancy ignored)\n"
        f"file: {dest.name}\nsource_run: {best}\n",
        encoding="utf-8",
    )
    log("[YOLO] 완료")
    return dest
