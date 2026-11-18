"""Select requested model inference settings on validation and write a report."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="models/yolo_spaces/yolo11s_seg_1.pt")
    p.add_argument("--metrics",default="outputs/train/line_improvement/yolo11_comparison/metrics.json")
    p.add_argument("--no-preference",action="store_true")
    p.add_argument("--min-precision",type=float,default=0.)
    p.add_argument("--report",default="outputs/train/line_improvement/report.md")
    args=p.parse_args()
    data=json.loads((ROOT/args.metrics).read_text(encoding="utf-8"))
    rows=data["rows"]
    candidates=[]
    for r in rows:
        if r["model"]==args.model:
            for stage in ("quad_only","refined"):
                if r[stage]["precision"] >= args.min_precision:
                    candidates.append((r,stage))
    if not candidates:raise RuntimeError("No requested model setting meets the precision requirement")
    selected,stage=max(candidates,key=lambda pair:(pair[0][pair[1]]["f1"],pair[0][pair[1]]["precision"],
                                                -pair[0]["imgsz"],not pair[0]["tiled"],pair[1]=="quad_only"))
    path=(ROOT/args.model).with_suffix(".json")
    meta=json.loads(path.read_text(encoding="utf-8"))
    meta.setdefault("training_imgsz",640)
    meta.update({"inference_imgsz":selected["imgsz"],"tiled":selected["tiled"],"snap_lines":stage=="refined",
                 "selected_stage":stage,"quad_fit":selected.get("quad_fit","conservative"),"validation_polygon_metrics":selected[stage],
                 "selection_min_precision":args.min_precision,
                 "selection_scope":"validation settings selection, not an independent test"})
    path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    if not args.no_preference:
        (ROOT/"models/preferred_parking_model.json").write_text(json.dumps({"model":args.model,"reason":"User requested YOLO11s; experimental validation settings"},indent=2),encoding="utf-8")
    text=["# YOLO11s 주차 칸 탐지 실험 결과", "", "## 학습", "",
          f"이번 단계 모델: {Path(args.model).name}. 학습 횟수: {meta['epochs']}에포크. 시작 가중치: {meta['base_model']}.",
          "원본 학습 25장·546칸에서 확대·가림 입력 58장을 만들고, 검토한 배경 2장을 추가했다. 검증 원본은 이전과 같은 4장·76칸이다.",
          f"freeze={meta['freeze']}, 학습 입력 {meta['training_imgsz']}, 배치 {meta['batch']}, AdamW 학습률 {meta.get('learning_rate',.0003)}, CPU. 기존 모델은 보존했다.", "", "## 같은 검증 사진에서 비교", "",
          "각 행은 confidence 0.35, 원본 해상도 다각형 IoU >= 0.5에서 계산했다. mAP나 일반적인 분류 정확도가 아니다.", "",
          "| 모델 | 입력 | 타일 | 처리/사각형 피팅 | 정밀도 | 재현율 | F1 | 맞춘 칸/76 | CPU ms/frame |",
          "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        for mode in ("raw","quad_only","refined"):
            m=r[mode]
            text.append(f"| {Path(r['model']).name} | {r['imgsz']} | {r['tiled']} | {mode}/{r.get('quad_fit','conservative')} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.3f} | {m['tp']}/76 | {r['mean_ms']:.0f} |")
    text += ["", f"선택 기준: 정밀도 {args.min_precision:.0%} 이상인 설정 중 F1 최대. 검증에서 정한 실험 기준이며 실사용 품질 보장이 아니다.", "", "## 적용한 설정", "",f"모델: `{args.model}`",f"입력: {selected['imgsz']}, 타일: {selected['tiled']}, 직선 보정: {stage=='refined'}, 사각형 피팅: {selected.get('quad_fit','conservative')}.",
             "모델 옆 JSON에 저장해 앱에서 이 모델을 선택하면 자동 적용된다. 사각형 출력은 유지한다.",
             "", "## 한계", "", "검증 4장으로 학습 체크포인트와 추론 설정을 선택했다. 독립 테스트 성능이나 새로운 주차장 전체의 정확도로 해석하면 안 된다.",
             "모델 변경, 추가 카메라, 확대, 학습률, 학습 횟수 등이 함께 바뀌었으므로 어느 한 요소의 효과라고 단정할 수 없다.",
             "주차선 픽셀을 직접 색인한 모델이 아니라 주차 칸 영역 분할 모델이다. 완전히 가려진 칸 복원 성능은 별도로 입증되지 않았다.",
             "타일은 정답 없이 화면을 확대 분석하고 원본 좌표로 복원해 중복을 제거한다. CPU에서는 추론 비용이 증가한다.",
             "enclosing은 볼록 윤곽을 네 꼭짓점으로 근사한다. 원본 CCTV 화면에 직각을 강제하지 않는다. OpenCV approxPolyN이 없으면 기존 근사 방식으로 대체된다.",
             "선택 이후 학습에서 제외한 카메라를 별도 평가한다. 해당 결과로 다시 설정을 맞추지 않는다.",
             "", "비교 이미지에서는 노란색 정답과 AI 원본·사각형 단순화·직선 보정 결과를 함께 확인할 수 있다."]
    output=ROOT/args.report
    output.write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps({"model":args.model,"imgsz":selected["imgsz"],"tiled":selected["tiled"],"stage":stage,"metrics":selected[stage],"report":str(output)},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
