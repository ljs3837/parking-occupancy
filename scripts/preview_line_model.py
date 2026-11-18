"""Save held-out image comparisons before and after geometric refinement."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from parking.perception.live import LiveDetector, draw_detections
    from parking.perception.lines import refine_spaces
    dataset = Path(args.dataset)
    output = dataset.parent / "previews"
    output.mkdir(exist_ok=True)
    model = YOLO(args.model)
    detector = LiveDetector()
    summary = []
    for source in sorted((dataset / "images/val").glob("*")):
        image = cv2.imdecode(np.fromfile(str(source),dtype=np.uint8), cv2.IMREAD_COLOR)
        raw = detector._yolo_boxes(image, model, "occupancy")
        refined = refine_spaces(image, raw)
        panels = [draw_detections(image,raw), draw_detections(image,refined)]
        for panel, title in zip(panels,("AI prediction", "Quadrilateral + line refinement")):
            cv2.rectangle(panel,(0,0),(panel.shape[1],32),(0,0,0),-1)
            cv2.putText(panel,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        preview = np.hstack(panels)
        if preview.shape[1] > 1800:
            preview = cv2.resize(preview,(1800,round(preview.shape[0]*1800/preview.shape[1])))
        path = output / (source.stem + "_comparison.jpg")
        ok, encoded = cv2.imencode(".jpg",preview)
        if not ok: raise RuntimeError("Preview encoding failed")
        encoded.tofile(str(path))
        summary.append({"image":source.name,"model_candidates":len(raw),"quadrilateral_candidates":len(refined),
                        "line_refined":sum(d.get("line_refined",False) for d in refined),"preview":str(path)})
    (output / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
