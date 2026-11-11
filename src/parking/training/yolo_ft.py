"""주차칸 empty/occupied YOLO 파인튜닝."""

from __future__ import annotations

import shutil
from pathlib import Path

from parking.training.prepare import MODELS_DIR, build_yolo_dataset

OUT_DIR = MODELS_DIR / "yolo_occupancy"


def train_yolo(
    *,
    epochs: int = 5,
    batch: int = 8,
    max_images: int = 100,
    imgsz: int = 640,
    model_name: str = "yolov8n-seg.pt",
    log=print,
    should_stop=None,
) -> Path:
    yaml_path = build_yolo_dataset(max_images, log)
    log(f"[YOLO] data={yaml_path}")
    log(f"[YOLO] base={model_name}  epochs={epochs}  batch={batch}")

    if should_stop and should_stop():
        raise RuntimeError("학습 시작 전 중지됨")

    from ultralytics import YOLO

    model = YOLO(model_name)
    # Ultralytics는 자체적으로 중단 콜백이 약해서 epoch 단위로만 확인
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(MODELS_DIR / "_runs"),
        name="yolo_occupancy",
        exist_ok=True,
        verbose=True,
        workers=0,
        patience=max(3, epochs),
    )

    # best 가중치 복사
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        # fallback
        cand = MODELS_DIR / "_runs" / "yolo_occupancy" / "weights" / "best.pt"
        best = cand if cand.is_file() else best
    dest = OUT_DIR / "best.pt"
    if best.is_file():
        shutil.copy2(best, dest)
        log(f"[YOLO] best 저장 → {dest}")
    else:
        raise RuntimeError(f"YOLO best.pt를 찾지 못했습니다: {best}")

    (OUT_DIR / "README.txt").write_text(
        f"YOLO segmentation ({model_name})\n"
        "labels: polygon empty/occupied\n"
        f"best: {dest}\nsource_run: {best}\n",
        encoding="utf-8",
    )
    log("[YOLO] 완료")
    return dest
