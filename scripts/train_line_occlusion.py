"""Train on local CCTV annotations with train-only partial occlusion."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    import torch
    from parking.training.prepare import _load_local_annotations, build_yolo_dataset
    from parking.training.yolo_ft import train_yolo
    torch.set_num_threads(4)
    samples = sorted((s for s in _load_local_annotations() if s["rel"].startswith("cctv/")), key=lambda s: s["rel"])
    if len(samples) < 5:
        raise RuntimeError("At least five annotated CCTV images are required")
    run_dir = ROOT / "outputs" / "train" / ("line_occlusion_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    data = build_yolo_dataset(print, samples=samples, output_dir=run_dir / "dataset")
    manifest = {
        "task": "parking slot segmentation with partial occlusion",
        "source_images": len(samples),
        "source_slots": sum(len(s["polygons"]) for s in samples),
        "epochs": args.epochs,
        "split_seed": 42,
        "validation_scope": "image holdout; not a new-site or independent occlusion benchmark",
        "sources": [s["rel"] for s in samples],
        "dataset": str(data),
        "train_images": len(list((data.parent / "images/train").glob("*"))),
        "val_images": len(list((data.parent / "images/val").glob("*"))),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    model = train_yolo(epochs=args.epochs, batch=4, imgsz=640,
                       model_name=str(ROOT / "yolov8n-seg.pt"), data_yaml=data)
    manifest["model"] = str(model)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRAINING_COMPLETE " + str(model), flush=True)


if __name__ == "__main__":
    main()
