"""Common raster polygon IoU metric for segmentation and corner predictions."""
import cv2
import numpy as np


def polygon_mask(points,width,height,resolution=None):
    mask_width=width if resolution is None else resolution
    mask_height=height if resolution is None else resolution
    mask=np.zeros((mask_height,mask_width),np.uint8)
    if points is None or len(points)<3:return mask.astype(bool)
    q=np.asarray(points,float)*[mask_width/width,mask_height/height]
    if not np.isfinite(q).all():return mask.astype(bool)
    cv2.fillPoly(mask,[np.rint(q).astype(np.int32)],1)
    return mask.astype(bool)


def score_polygons(predictions,targets,width,height,iou_threshold=.5):
    masks=[polygon_mask(q,width,height) for q in targets]
    areas=np.asarray([m.sum() for m in masks])
    matched=set();ious=[]
    for p in sorted(predictions,key=lambda p:p["conf"],reverse=True):
        mask=polygon_mask(p.get("points"),width,height)
        intersections=np.asarray([np.count_nonzero(mask & gt) for gt in masks])
        overlaps=intersections/np.maximum(1,mask.sum()+areas-intersections)
        for j in matched:overlaps[j]=-1
        if len(overlaps) and overlaps.max()>=iou_threshold:
            j=int(overlaps.argmax());matched.add(j);ious.append(float(overlaps[j]))
    return {"tp":len(matched),"fp":len(predictions)-len(matched),"fn":len(targets)-len(matched),"matched_ious":ious}


def summarize_scores(items):
    tp=sum(s["tp"] for s in items);fp=sum(s["fp"] for s in items);fn=sum(s["fn"] for s in items)
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return {"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,
            "f1":2*precision*recall/max(1e-9,precision+recall)}
