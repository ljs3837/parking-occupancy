"""Qualitative video samples; no accuracy claim without video annotations."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="models/yolo_spaces/yolo11s_seg_1.pt")
    p.add_argument("--output",default="outputs/train/line_improvement/video_previews")
    args=p.parse_args()
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from parking.perception.live import LiveDetector,draw_detections
    from parking.perception.lines import refine_spaces
    torch.set_num_threads(4)
    model_path=ROOT/args.model
    meta=json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    model=YOLO(str(model_path));detector=LiveDetector()
    output=ROOT/args.output
    output.mkdir(exist_ok=True)
    records=[];contact=[]
    for source in sorted((ROOT/"data/observe").glob("*.mp4"))[:4]:
        cap=cv2.VideoCapture(str(source));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fraction in (.25,.5):
            frame_index=round(total*fraction)
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
            ok,image=cap.read()
            if not ok:raise RuntimeError(f"Could not read {source} at {frame_index}")
            raw=detector.predict_space_candidates(image,model,imgsz=meta["inference_imgsz"],tiled=meta["tiled"])
            detected=refine_spaces(image,raw,snap_lines=meta.get("snap_lines",True),quad_fit=meta.get("quad_fit","conservative"))
            canvas=draw_detections(image,detected)
            cv2.rectangle(canvas,(0,0),(canvas.shape[1],30),(0,0,0),-1)
            cv2.putText(canvas,f"YOLO11s | {source.name} | {len(detected)} slot candidates",(6,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            path=output/f"video_{source.stem}_frame_{frame_index}.jpg"
            cv2.imencode(".jpg",canvas)[1].tofile(str(path))
            if fraction==.5:contact.append(cv2.resize(canvas,(640,360)))
            records.append({"source":str(source),"frame":frame_index,"raw_candidates":len(raw),"displayed_candidates":len(detected),"preview":str(path)})
        cap.release()
    if len(contact)==4:
        cv2.imencode(".jpg",np.vstack([np.hstack(contact[:2]),np.hstack(contact[2:])]))[1].tofile(str(output/"contact_sheet.jpg"))
    (output/"summary.json").write_text(json.dumps({"scope":"qualitative, no ground truth or accuracy measurement","records":records},indent=2),encoding="utf-8")
    print(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
