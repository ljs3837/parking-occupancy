"""Small, date-held-out parking-corner experiment; never edits UI annotations."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parking.common.device import use_amp, yolo_device

import cv2
import numpy as np
import torch
from ultralytics import YOLO

OUT = ROOT / 'outputs' / 'slot_pose_trial'
DATA = ROOT / 'data/huggingface/hub/Voxel51/PKLot'


def corners(points):
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (4, 2):
        return None
    p = p[np.argsort(np.arctan2(*(p - p.mean(0))[:, ::-1].T))]
    return np.roll(p, -np.argmin(p.sum(1)), axis=0)


def prepare():
    samples = json.loads((DATA / 'samples.json').read_text(encoding='utf-8'))['samples']
    local = [s for s in samples if (DATA / s['filepath']).is_file()]
    selected = {}
    for split, date, limit in [('train', '2012-09-21', 32), ('val', '2012-09-19', 18)]:
        selected[split] = [s for s in local if date in str(s['date'])][:limit]
        for kind in ['images', 'labels']:
            (OUT / 'dataset' / kind / split).mkdir(parents=True, exist_ok=True)
        for s in selected[split]:
            src = DATA / s['filepath']
            shutil.copyfile(src, OUT / 'dataset/images' / split / src.name)
            rows = []
            for poly in s['parking_spaces']['polylines']:
                p = corners(poly['points'][0])
                if p is None:
                    continue
                p = np.clip(p, 0, 1)
                lo, hi = p.min(0), p.max(0)
                values = [0, *((lo + hi) / 2), *(hi - lo)]
                for x, y in p:
                    values.extend([float(x), float(y), 2])
                rows.append(' '.join(map(str, values)))
            (OUT / 'dataset/labels' / split / (src.stem + '.txt')).write_text('\n'.join(rows))
    yaml = f'path: {OUT.joinpath("dataset").as_posix()}\ntrain: images/train\nval: images/val\nnames: {{0: parking_space}}\nkpt_shape: [4, 3]\n'
    (OUT / 'dataset.yaml').write_text(yaml)
    (OUT / 'split.json').write_text(json.dumps({k: [s['filepath'] for s in v] for k, v in selected.items()}, indent=2))
    return selected


def iou(a, b):
    a, b = cv2.convexHull(a.astype(np.float32)), cv2.convexHull(b.astype(np.float32))
    inter = cv2.intersectConvexConvex(a, b)[0]
    return float(inter / max(cv2.contourArea(a) + cv2.contourArea(b) - inter, 1e-8))


def evaluate(selected):
    model = YOLO(str(OUT / 'training/weights/best.pt'))
    cases = [(DATA / s['filepath'], s) for s in selected['val']]
    for video in sorted((ROOT.parent / '개인자료/CHAD_점유실험').glob('*.mp4')):
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .25))
        ok, frame = cap.read()
        cap.release()
        if ok:
            target = OUT / f'CHAD_{video.stem}.jpg'
            cv2.imwrite(str(target), frame)
            cases.append((target, None))
    report, panels = [], []
    totals = {'tp': 0, 'predictions': 0, 'ground_truth': 0}
    for index, (path, sample) in enumerate(cases):
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        result = model.predict(im, imgsz=512, conf=.15, iou=.5, max_det=300, device=yolo_device(), verbose=False)[0]
        pred = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else np.empty((0, 4, 2))
        conf = result.boxes.conf.cpu().tolist()
        gt = [] if sample is None else [p for x in sample['parking_spaces']['polylines'] if (p := corners(x['points'][0])) is not None]
        gt = [p * [w, h] for p in gt]
        pairs = sorted([(iou(a, b), i, j) for i, a in enumerate(pred) for j, b in enumerate(gt)], reverse=True)
        used_p, used_g = set(), set()
        for score, i, j in pairs:
            if score >= .5 and i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
        if sample is not None:
            totals['tp'] += len(used_p)
            totals['predictions'] += len(pred)
            totals['ground_truth'] += len(gt)
        polygons = [{'id': f'proposal-{i+1}', 'points': p.tolist(), 'confidence': c, 'tags': ['unreviewed']} for i, (p, c) in enumerate(zip(pred, conf))]
        (OUT / f'{path.stem}_proposals.json').write_text(json.dumps({'image': str(path), 'coordinate_system': 'original_pixels', 'polygons': polygons}, indent=2))
        report.append({'image': str(path), 'predicted': len(pred), 'gt': len(gt) if sample else None, 'matched_iou_0.5': len(used_p) if sample else None})
        for p in gt:
            cv2.polylines(im, [p.astype(np.int32)], True, (255, 230, 0), 2)
        for p in pred:
            cv2.polylines(im, [p.astype(np.int32)], True, (0, 140, 255), 2)
        cv2.rectangle(im, (0, 0), (w, 58), (20, 20, 20), -1)
        title = f'{"PKLot held-out date" if sample else "CHAD unseen camera"} | proposals {len(pred)}'
        cv2.putText(im, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        cv2.putText(im, 'CYAN: reference   ORANGE: model proposal', (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f'{path.stem}_result.jpg'), im)
        if index in [0, 5, 10, 17] or sample is None:
            panels.append(cv2.resize(im, (768, 432)))
    totals['precision_iou_0.5'] = totals['tp'] / max(totals['predictions'], 1)
    totals['recall_iou_0.5'] = totals['tp'] / max(totals['ground_truth'], 1)
    payload = {'training': '32 PKLot images, Sep 21; 20 epochs; detector backbone transferred to custom 4-keypoint pose', 'evaluation': '18 PKLot images Sep 19, same camera/layout; 2 CHAD frames without GT', 'threshold': .15, 'totals': totals, 'cases': report}
    (OUT / 'report.json').write_text(json.dumps(payload, indent=2))
    cv2.imwrite(str(OUT / 'comparison.jpg'), np.vstack([np.hstack(panels[i:i+2]) for i in range(0, len(panels), 2)]))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = prepare()
    if not (OUT / 'training/weights/best.pt').exists():
        model = YOLO('yolo11n-pose.yaml').load(str(ROOT / 'yolo11n.pt'))
        device = yolo_device()
        model.train(data=str(OUT / 'dataset.yaml'), epochs=20, imgsz=512, batch=4, workers=0, device=device, freeze=10, project=str(OUT), name='training', exist_ok=True, patience=0, plots=False, cache=False, amp=use_amp(device), mosaic=0, fliplr=0, flipud=0, degrees=0, translate=.05, scale=.1, seed=42)
    evaluate(selected)
