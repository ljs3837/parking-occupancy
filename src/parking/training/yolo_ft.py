"""주차칸 좌표(단일 클래스) YOLO 파인튜닝. 점유는 차량 YOLO가 따로 판정한다."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from parking.common.device import device_label, gpu_file_slug, use_amp, yolo_device
from parking.training.prepare import MODELS_DIR, ROOT, build_yolo_dataset, list_data_folders, next_model_path

OUT_DIR = MODELS_DIR / "yolo_spaces"
# 주차 CCTV는 위아래가 있는 사선 시점이라, 매 사진마다 -15~15도만 돌린다.
ROTATION_DEGREES = 15.0
TEST_ROUND_2_AUGMENTATION = {
    "degrees": 5.0,
    "perspective": 0.0,
    "scale": 0.15,
    "translate": 0.03,
    "mosaic": 0.0,
}
# Ultralytics는 MPS/CPU에서 workers를 0으로 강제한다. 아래에서 콜백으로 다시 올린다.
DATALOADER_WORKERS = 4


def _force_dataloader_workers(model, workers: int, log) -> None:
    """MPS/CPU에서도 DataLoader 워커를 강제로 살린다. Mac+GUI에서는 불안정할 수 있다."""

    def on_pretrain_routine_start(trainer):
        if trainer.args.workers == workers:
            return
        log(
            f"[YOLO] Ultralytics가 workers={trainer.args.workers}로 막아서 "
            f"{workers}로 강제 복구 (불안정하면 0으로 되돌리세요)"
        )
        trainer.args.workers = workers

    model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분"
    if minutes:
        return f"{minutes}분 {secs:02d}초"
    return f"{secs}초"


def _metric(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _attach_progress(model, log) -> None:
    """에포크·배치·남은 시간을 학습 탭 로그로 보낸다."""
    state = {"train_start": None, "epoch_start": None, "batch": 0, "bucket": -1}

    def on_train_start(trainer):
        state["train_start"] = time.time()
        batches = len(trainer.train_loader)
        log(
            f"[YOLO] 학습 루프 시작 · 에포크 {trainer.start_epoch + 1}→{trainer.epochs}"
            f" · 에포크당 배치 {batches}"
        )

    def on_train_epoch_start(trainer):
        state["epoch_start"] = time.time()
        state["batch"] = 0
        state["bucket"] = -1
        log(f"[YOLO] 에포크 {trainer.epoch + 1}/{trainer.epochs} 학습 시작")

    def on_train_batch_end(trainer):
        state["batch"] += 1
        total = max(len(trainer.train_loader), 1)
        done = state["batch"]
        bucket = int(done * 20 / total)
        if done not in (1, total) and bucket == state["bucket"]:
            return
        state["bucket"] = bucket
        now = time.time()
        epoch_elapsed = now - (state["epoch_start"] or now)
        epoch_left = epoch_elapsed / done * (total - done)
        span = max(trainer.epochs - trainer.start_epoch, 1)
        finished = (trainer.epoch - trainer.start_epoch) + done / total
        overall_left = (now - state["train_start"]) / finished * (span - finished) if finished else 0
        log(
            f"[YOLO] 에포크 {trainer.epoch + 1}/{trainer.epochs}"
            f" · 배치 {done}/{total} ({done * 100 // total}%)"
            f" · 이번 에포크 남은 시간 {_format_duration(epoch_left)}"
            f" · 전체 남은 시간 {_format_duration(overall_left)}"
        )

    def on_train_epoch_end(trainer):
        elapsed = time.time() - (state["epoch_start"] or time.time())
        log(
            f"[YOLO] 에포크 {trainer.epoch + 1}/{trainer.epochs} 학습 끝"
            f" · 소요 {_format_duration(elapsed)} · 검증 시작"
        )

    def on_fit_epoch_end(trainer):
        metrics = trainer.metrics or {}
        score = _metric(metrics, "metrics/mAP50(M)", "metrics/mAP50(B)")
        score_text = f" · mAP50 {score:.3f}" if score is not None else ""
        elapsed = trainer.epoch_time or 0
        remain = max(trainer.epochs - (trainer.epoch + 1), 0) * elapsed
        log(
            f"[YOLO] 에포크 {trainer.epoch + 1}/{trainer.epochs} 완료"
            f" · 학습+검증 {_format_duration(elapsed)}{score_text}"
            f" · 남은 에포크 예상 {_format_duration(remain)}"
        )

    for name, callback in (
        ("on_train_start", on_train_start),
        ("on_train_epoch_start", on_train_epoch_start),
        ("on_train_batch_end", on_train_batch_end),
        ("on_train_epoch_end", on_train_epoch_end),
        ("on_fit_epoch_end", on_fit_epoch_end),
    ):
        model.add_callback(name, callback)


def train_yolo(
    *,
    epochs: int = 5,
    batch: int = 4,
    imgsz: int = 640,
    model_name: str = "yolo11s-seg.pt",
    data_yaml: Path | None = None,
    folders: set[str] | None = None,
    test_round_2: bool = False,
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
    augmentation = {"degrees": ROTATION_DEGREES}
    if test_round_2:
        augmentation.update(TEST_ROUND_2_AUGMENTATION)
    if task == "pose":
        # Corner identities depend on image orientation.
        augmentation["fliplr"] = 0.0
    yaml_path = Path(data_yaml) if data_yaml is not None else build_yolo_dataset(log, folders=folders, task=task)
    log(f"[YOLO] data={yaml_path}")
    log(f"[YOLO] base={model_name}  epochs={epochs}  batch={batch}")
    augmentation_text = ", ".join(f"{key}={value:g}" for key, value in augmentation.items())
    log(f"[YOLO] 증강: {augmentation_text}")

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
    _attach_progress(model, log)
    _force_dataloader_workers(model, DATALOADER_WORKERS, log)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(MODELS_DIR / "_runs"),
        name=run_name,
        exist_ok=False,
        verbose=True,
        workers=DATALOADER_WORKERS,
        device=device,
        amp=amp,
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
        f"labels: "
        f"{'four keypoints' if task == 'pose' else 'bbox' if task == 'detect' else 'polygon'}"
        f" parking_space (occupancy ignored)\n"
        f"device: {device_label()}\n"
        f"data: {', '.join(used_folders)}\n"
        f"augmentation: {augmentation_text}\n"
        f"file: {dest.name}\nsource_run: {best}\n",
        encoding="utf-8",
    )
    log("[YOLO] 완료")
    return dest
