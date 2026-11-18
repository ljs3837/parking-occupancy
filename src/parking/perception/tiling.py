"""Image-only overlapping crops; never uses annotation coordinates at inference."""
from __future__ import annotations
import cv2
import numpy as np


def polygon_iou(a,b):
    if a is None or b is None or len(a)<3 or len(b)<3:
        return 0.
    a=cv2.convexHull(np.asarray(a,np.float32).reshape(-1,2))
    b=cv2.convexHull(np.asarray(b,np.float32).reshape(-1,2))
    area_a,area_b=abs(cv2.contourArea(a)),abs(cv2.contourArea(b))
    if min(area_a,area_b)<1:return 0.
    intersection,_=cv2.intersectConvexConvex(a,b)
    return float(intersection/max(1e-6,area_a+area_b-intersection))


def merge_candidates(detections, threshold=.55):
    kept=[]
    for item in sorted(detections,key=lambda d:d["conf"],reverse=True):
        if any(polygon_iou(item.get("points"),other.get("points"))>threshold for other in kept):
            continue
        kept.append(item)
    return kept


def tiled_candidates(frame, predict):
    """Full frame plus four 65%-size overlapping crops, projected back to pixels."""
    h,w=frame.shape[:2]
    output=list(predict(frame))
    if min(h,w)<128:
        return output
    cw,ch=int(round(w*.65)),int(round(h*.65))
    for x,y in ((0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch)):
        crop=frame[y:y+ch,x:x+cw]
        for detection in predict(crop):
            x1,y1,x2,y2=detection["xyxy"]
            # Do not add truncated tile-border objects as complete slots.
            if (x>0 and x1<3) or (y>0 and y1<3) or (x+cw<w and x2>cw-3) or (y+ch<h and y2>ch-3):
                continue
            item=dict(detection)
            item["xyxy"]=(x1+x,y1+y,x2+x,y2+y)
            if item.get("points") is not None:
                item["points"]=(np.asarray(item["points"])+[x,y]).tolist()
            output.append(item)
    return merge_candidates(output)
