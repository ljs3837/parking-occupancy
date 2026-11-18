"""Audited ROI/corner datasets with the previous validation images held fixed."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from parking.training.occlusion import occlude_slots
from parking.training.prepare import _load_local_annotations


def ordered_quad(points):
    q = np.asarray(points, np.float32).reshape(-1, 2)
    if len(q) == 5 and np.allclose(q[0], q[-1]):
        q = q[:-1]
    if len(q) != 4 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        return None
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 1e-6:
        return None
    # Image-relative clockwise order, starting at the top-left-most corner.
    center = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:,1]-center[1], q[:,0]-center[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)


def roi_crop(image, polygons, padding=.10):
    h,w = image.shape[:2]
    all_points = np.vstack([p["points"] for p in polygons]) * [w,h]
    low,high = all_points.min(0),all_points.max(0)
    margin = np.maximum((high-low)*padding, 8)
    low = np.maximum(np.floor(low-margin),0).astype(int)
    high = np.minimum(np.ceil(high+margin),(w,h)).astype(int)
    output = []
    for p in polygons:
        xy = (np.asarray(p["points"])*[w,h]-low)/(high-low)
        output.append({**p,"points":xy.tolist()})
    return image[low[1]:high[1],low[0]:high[0]].copy(),output,[*low.tolist(),*high.tolist()]


def label_rows(polygons, task):
    rows=[]
    for p in polygons:
        q=np.asarray(p["points"],float)
        if task == "segment":
            values=q.ravel()
        else:
            low,high=q.min(0),q.max(0)
            values=np.r_[(low+high)/2,high-low,np.c_[q,np.full(4,2)].ravel()]
        rows.append("0 " + " ".join(f"{v:.7f}" for v in values))
    return "\n".join(rows) + ("\n" if rows else "")


def select_sources(baseline=None):
    sources=_load_local_annotations()
    config_path=Path(__file__).resolve().parents[3]/"configs/parking_line_split.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    val_stems={Path(p).stem for p in config["validation_images"]}
    if baseline is not None and (Path(baseline)/"images/val").is_dir():
        previous={p.stem.rsplit("_",1)[0] for p in (Path(baseline)/"images/val").glob("*")}
        if val_stems != previous:raise RuntimeError("Validation split differs from the baseline")
    local=sorted((s for s in sources if s["rel"].startswith("cctv/")),key=lambda s:s["rel"])
    grouped=defaultdict(list)
    for s in sources:
        if s["rel"].startswith("cctv_image/"):
            grouped[str(Path(s["rel"]).parent)].append(s)
    extras=[]
    for group,items in sorted(grouped.items()):
        items=sorted(items,key=lambda s:s["rel"])
        extras.append(items[len(items)//2])
    return local+extras,val_stems


def build_improved_datasets(root, baseline):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    sources,val_stems=select_sources(baseline)
    records=[]
    hashes={"train":set(),"val":set()}
    for index,s in enumerate(sources):
        image=cv2.imdecode(np.fromfile(str(s["path"]),np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(f"Unreadable image: {s['path']}")
        is_val=s["rel"].startswith("cctv/") and Path(s["rel"]).stem in val_stems
        split="val" if is_val else "train"
        record={"source":s["rel"],"split":split,"source_sha256":hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest()}
        if not is_val and min(image.shape[:2]) < 224:
            record["excluded"]="low-resolution image; fewer than 224 pixels on the shorter side"
            records.append(record)
            continue
        invalid=[i for i,p in enumerate(s["polygons"]) if ordered_quad(p["points"]) is None]
        if invalid and not is_val:
            record["excluded"]="non-quadrilateral/clipped geometry requiring manual review"
            record["polygon_indices"]=invalid
            records.append(record)
            continue
        polys=[]
        for i,p in enumerate(s["polygons"]):
            quad=ordered_quad(p["points"])
            if quad is None:
                raise RuntimeError(f"Needs manual geometry review: {s['rel']}, polygon {i}")
            polys.append({**p,"points":quad.tolist()})
        hashes[split].add(record["source_sha256"])
        stem=f"source_{index:03d}"
        variants=[(stem,image,polys)] if is_val else []
        if not is_val:
            roi,roi_polys,bounds=roi_crop(image,polys)
            record["roi_bounds"]=bounds
            variants.append((stem+"_roi",roi,roi_polys))
            # Extra camera frames have shared layouts; retain their full context too.
            if s["rel"].startswith("cctv_image/"):
                variants.append((stem+"_full",image,polys))
            occluded,count=occlude_slots(roi,roi_polys,seed=142+index)
            if count:
                variants.append((stem+"_occluded",occluded,roi_polys))
        record["variants"]=[v[0] for v in variants]
        record["slots"]=len(polys)
        records.append(record)
        for name,img,labels in variants:
            ok,encoded=cv2.imencode(".png",img)
            if not ok: raise RuntimeError("Could not encode image")
            for task in ("segment","pose"):
                directory=root/task
                ip=directory/"images"/split
                lp=directory/"labels"/split
                ip.mkdir(parents=True,exist_ok=True)
                lp.mkdir(parents=True,exist_ok=True)
                encoded.tofile(str(ip/(name+".png")))
                (lp/(name+".txt")).write_text(label_rows(labels,task),encoding="utf-8")
    if hashes["train"] & hashes["val"]: raise RuntimeError("Duplicate image across splits")
    if len(hashes["val"])!=4: raise RuntimeError("The previous four-image validation split must stay fixed")
    for task in ("segment","pose"):
        directory=root/task
        text=f"path: {directory.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: parking_space\n"
        if task=="pose":text+="kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n"
        (directory/"data.yaml").write_text(text,encoding="utf-8")
    manifest={"records":records,"validation":"same four original images as yolo_4; no label-guided validation crop",
              "training":"ROI crops + full extra camera frames + partial occlusion; no new manual line labels",
              "image_counts":{split:len(list((root/"segment/images"/split).glob("*.png"))) for split in ("train","val")},
              "extra_camera_groups":len({str(Path(r["source"]).parent) for r in records if r["source"].startswith("cctv_image/") and not r.get("excluded")})}
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest
