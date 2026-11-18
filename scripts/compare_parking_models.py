"""Compare all models on the same uncropped original validation images."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",nargs="+",required=True)
    p.add_argument("--output",default="outputs/train/line_improvement/comparison")
    p.add_argument("--camera-test",action="store_true")
    p.add_argument("--quad-fit",choices=["conservative","enclosing"],default="conservative")
    p.add_argument("--use-model-settings",action="store_true")
    args=p.parse_args()
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from parking.perception.live import LiveDetector,draw_detections
    from parking.perception.lines import refine_spaces
    from parking.training.polygon_metrics import score_polygons,summarize_scores
    torch.set_num_threads(4)
    dataset=ROOT/"outputs/train/line_improvement/dataset_v3/segment"
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    samples=[]
    for path in sorted((dataset/"images/val").glob("*.png")):
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2]
        targets=[np.array([float(v) for v in line.split()[1:]]).reshape(-1,2)*[w,h]
                 for line in (dataset/"labels/val"/(path.stem+".txt")).read_text().splitlines()]
        samples.append((path,image,targets))
    if args.camera_test:
        from parking.training.prepare import _load_local_annotations
        manifest=json.loads((dataset.parent/"manifest.json").read_text(encoding="utf-8"))
        config=json.loads((ROOT/"configs/parking_line_split.json").read_text(encoding="utf-8"))
        names={r["source"] for r in manifest["records"] if str(Path(r["source"]).parent).replace("\\","/") in config["heldout_camera_groups"]}
        trained={r["source"] for r in manifest["records"] if r["split"]=="train" and not r.get("excluded")}
        assert not names & trained
        samples=[]
        for source in _load_local_annotations():
            if source["rel"] not in names:continue
            image=cv2.imdecode(np.fromfile(str(source["path"]),np.uint8),cv2.IMREAD_COLOR)
            h,w=image.shape[:2]
            targets=[np.asarray(poly["points"])*[w,h] for poly in source["polygons"]]
            path=Path("_".join(Path(source["rel"]).parts[-3:])).with_suffix(".png")
            samples.append((path,image,targets))
        if len(samples)!=3:raise RuntimeError("Expected three camera test representatives")
    detector=LiveDetector();rows=[]
    for model_path in args.models:
        model=YOLO(str(ROOT/model_path))
        metadata_path=(ROOT/model_path).with_suffix(".json")
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        settings=[(metadata.get("inference_imgsz",640),metadata.get("tiled",False))] if args.use_model_settings else [(640,False),(960,False),(640,True)]
        fit=metadata.get("quad_fit",args.quad_fit) if args.use_model_settings else args.quad_fit
        for size,tiled in settings:
            name=Path(model_path).parent.name+"_"+Path(model_path).stem+f"_{size}"+("_tiles" if tiled else "_full")
            # Warm up outside the latency measurement (no labels used).
            detector.predict_space_candidates(samples[0][1],model,imgsz=size,tiled=tiled)
            scores={"raw":[],"quad_only":[],"refined":[]};times=[]
            for path,image,targets in samples:
                t=time.perf_counter()
                raw=detector.predict_space_candidates(image,model,imgsz=size,tiled=tiled)
                quad_only=refine_spaces(image,raw,snap_lines=False,quad_fit=fit)
                refined=refine_spaces(image,raw,quad_fit=fit)
                times.append((time.perf_counter()-t)*1000)
                for mode,pred in (("raw",raw),("quad_only",quad_only),("refined",refined)):
                    scores[mode].append(score_polygons(pred,targets,image.shape[1],image.shape[0]))
                gt_canvas=image.copy()
                for q in targets:
                    cv2.polylines(gt_canvas,[np.rint(q).astype(np.int32)],True,(0,255,255),2)
                panels=[gt_canvas,draw_detections(image,raw),draw_detections(image,quad_only),draw_detections(image,refined)]
                for panel,title in zip(panels,("Labels (evaluation only)","AI candidates","Quadrilaterals","Line refinement")):
                    cv2.rectangle(panel,(0,0),(panel.shape[1],30),(0,0,0),-1)
                    cv2.putText(panel,title,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                cv2.imencode(".jpg",np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))[1].tofile(str(output/(name+"_"+path.stem+".jpg")))
            row={"model":model_path,"imgsz":size,"tiled":tiled,"confidence":.35,"quad_fit":fit,
                 "mean_ms":float(np.mean(times)),"raw":summarize_scores(scores["raw"]),"quad_only":summarize_scores(scores["quad_only"]),"refined":summarize_scores(scores["refined"])}
            rows.append(row);print(json.dumps(row),flush=True)
            (output/"metrics.json").write_text(json.dumps({"evaluation":("3 heldout camera representatives; no retuning; not necessarily unseen parking sites" if args.camera_test else "4 fixed original images, 76 slots; validation, not independent test")+"; native-resolution raster polygon IoU >= 0.5; confidence 0.35", "rows":rows},indent=2),encoding="utf-8")
    print("COMPARISON_COMPLETE",flush=True)

if __name__=="__main__":main()
