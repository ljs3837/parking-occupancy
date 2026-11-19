"""주차칸 좌표(단일 클래스) YOLO 파인튜닝. 점유는 차량 YOLO가 따로 판정한다."""

from __future__ import annotations

import shutil
from pathlib import Path

from parking.common.device import device_label, gpu_file_slug, use_amp, yolo_device
from parking.training.prepare import MODELS_DIR, ROOT, build_yolo_dataset, list_data_folders, next_model_path

OUT_DIR = MODELS_DIR / "yolo_spaces"
# 주차 CCTV는 위아래가 있는 사선 시점이라, 매 사진마다 -15~15도만 돌린다.
ROTATION_DEGREES = 15.0


def train_yolo(
    *,
    epochs: int = 5,
    batch: int = 4,
    imgsz: int = 640,
    model_name: str = "yolo11s-seg.pt",
    data_yaml: Path | None = None,
    folders: set[str] | None = None,
    log=print,
    should_stop=None,
) -> Path:
    device = yolo_device()
    amp = use_amp(device)
    log(f"[YOLO] device={device_label()}  amp={amp}")
    from ultralytics import YOLO

    local_weights = ROOT / model_name
    model = YOLO(str(local_weights) if local_weights.is_file() else model_name)
    task = model.task
    yaml_path = Path(data_yaml) if data_yaml is not None else build_yolo_dataset(log, folders=folders, task=task)
    log(f"[YOLO] data={yaml_path}")
    log(f"[YOLO] base={model_name}  epochs={epochs}  batch={batch}  degrees=±{ROTATION_DEGREES:g}")

    if should_stop and should_stop():
        raise RuntimeError("학습 시작 전 중지됨")

    if task == "pose":
        stem = Path(model_name).stem.replace("-", "_")
        base, _, version = stem.rpartition("_")
        dest = next_model_path(MODELS_DIR / "yolo_corners", base if version.isdigit() else stem)
    else:
        dest = next_model_path(OUT_DIR, f"yolo_{gpu_file_slug()}")
    run_name = dest.stem
    log(f"[YOLO] 저장 파일: {dest.name}")
    # Corner identities depend on image orientation.
    augmentation = {"fliplr": 0.0} if task == "pose" else {}
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
        **augmentation,
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

    used_folders = sorted(folders) if folders is not None else list_data_folders()
    dest.with_suffix(".txt").write_text(
        f"YOLO {task} ({model_name})\n"
        f"labels: {'four keypoints' if task == 'pose' else 'polygon'} parking_space (occupancy ignored)\n"
        f"device: {device_label()}\n"
        f"data: {', '.join(used_folders)}\n"
        f"file: {dest.name}\nsource_run: {best}\n",
        encoding="utf-8",
    )
    log("[YOLO] 완료")
    return dest
