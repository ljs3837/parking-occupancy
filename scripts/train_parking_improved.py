"""Reproducible ROI segmentation versus four-corner training experiment."""
from pathlib import Path
import sys,os,json,argparse,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",choices=["segment","pose"],required=True)
    parser.add_argument("--epochs",type=int,default=25)
    parser.add_argument("--model",default=None)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--freeze",type=int,default=10)
    parser.add_argument("--lr",type=float,default=None)
    parser.add_argument("--warmup-bias-lr",type=float,default=.1)
    args=parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS","4")
    os.environ.setdefault("MKL_NUM_THREADS","4")
    import torch
    from ultralytics import YOLO
    from parking.training.improved_data import build_improved_datasets
    from parking.training.prepare import next_model_path
    torch.set_num_threads(4)
    dataset=ROOT/"outputs/train/line_improvement/dataset_v3"
    baseline=ROOT/"outputs/train/line_occlusion_20261118_120235/dataset"
    if dataset.exists() and not (dataset/"manifest.json").is_file():
        raise RuntimeError("Incomplete dataset directory; inspect it before reusing")
    if not dataset.exists():
        manifest=build_improved_datasets(dataset,baseline)
        print(json.dumps(manifest["image_counts"]),flush=True)
    add_reviewed_negatives(dataset,args.task)
    dest_dir=ROOT/"models"/("yolo_spaces" if args.task=="segment" else "yolo_corners")
    model_source=args.model or ("yolo11s-seg.pt" if args.task=="segment" else "yolov8n-pose.yaml")
    stem="yolo11s_seg" if Path(model_source).stem.startswith(("yolo11s-seg","yolo11s_seg")) else "yolo"
    dest=next_model_path(dest_dir,stem)
    name=("roi_" if args.task=="segment" else "corners_")+dest.stem
    if args.task=="segment":
        model=YOLO(model_source)
    else:
        # Use existing local pretrained backbone; no external weight download.
        model=YOLO("yolov8n-pose.yaml").load(str(ROOT/"models/yolo_spaces/yolov8_initial_local.pt"))
    learning_rate=args.lr if args.lr is not None else (.0003 if args.task=="segment" else .001)
    result=model.train(data=str(dataset/args.task/"data.yaml"),epochs=args.epochs,batch=args.batch,
                       imgsz=args.imgsz,device="cpu",workers=0,amp=False,optimizer="AdamW",lr0=learning_rate,lrf=.1,
                       cos_lr=True,warmup_epochs=2,warmup_bias_lr=args.warmup_bias_lr,mosaic=0.,mixup=0.,scale=.15,translate=.03,
                       degrees=3.,hsv_s=.3,hsv_v=.25,fliplr=.5,freeze=args.freeze,seed=42,
                       project=str(ROOT/"models/_runs"),name=name,exist_ok=False,
                       patience=args.epochs,plots=True,save=True)
    best=Path(result.save_dir)/"weights/best.pt"
    shutil.copy2(best,dest)
    metadata={"task":args.task,"source_run":str(result.save_dir),"dataset":str(dataset/args.task/"data.yaml"),
              "inference_imgsz":args.imgsz,"training_imgsz":args.imgsz,"tiled":False,"epochs":args.epochs,
              "base_model":model_source,"freeze":args.freeze,"batch":args.batch,
              "learning_rate":learning_rate,"warmup_bias_lr":args.warmup_bias_lr,
              "train_images":len(list((dataset/args.task/"images/train").glob("*.png"))),
              "notes":"Experimental ROI/extra-camera model; selection pending identical validation comparison"}
    dest.with_suffix(".json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    dest.with_suffix(".txt").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("TRAINING_COMPLETE "+str(dest),flush=True)


def add_reviewed_negatives(dataset,task="pose"):
    import cv2,numpy as np
    from parking.training.prepare import _load_local_annotations
    samples=[s for s in _load_local_annotations() if s["rel"].startswith("cctv/")]
    # These exact crop previews were visually reviewed; do not infer background
    # merely because an arbitrary area lacks annotations.
    manifest=json.loads((dataset/"manifest.json").read_text(encoding="utf-8"))
    train_sources={r["source"] for r in manifest["records"] if r["split"]=="train" and not r.get("excluded")}
    negatives=[]
    for suffix,region in [("17",(0.,0.,1.,.38)),("11",(0.,.86,1.,1.))]:
        source=next(s for s in samples if Path(s["rel"]).stem==chr(0x3163)+suffix)
        if source["rel"] not in train_sources:
            raise RuntimeError("Reviewed background source is not in the training split")
        image=cv2.imdecode(np.fromfile(str(source["path"]),np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2];x,y,x2,y2=region
        crop=image[round(y*h):round(y2*h),round(x*w):round(x2*w)]
        stem="reviewed_background_"+suffix
        ok,encoded=cv2.imencode(".png",crop)
        if not ok:raise RuntimeError("Background image encoding failed")
        encoded.tofile(str(dataset/task/"images/train"/(stem+".png")))
        (dataset/task/"labels/train"/(stem+".txt")).write_text("",encoding="utf-8")
        negatives.append({"source":source["rel"],"normalized_crop":region,"reason":"visually checked, no parking slot"})
    (dataset/task/"reviewed_backgrounds.json").write_text(json.dumps(negatives,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
