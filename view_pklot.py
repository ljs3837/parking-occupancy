from pathlib import Path

import fiftyone as fo

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "huggingface" / "hub" / "Voxel51" / "PKLot"
NAME = "PKLot"

if not DATASET_DIR.exists():
    raise SystemExit(f"데이터 폴더 없음: {DATASET_DIR}")

# 디스크 파일 ≠ DB 등록. 폴더에서 import 필요.
if NAME in fo.list_datasets():
    dataset = fo.load_dataset(NAME)
else:
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(DATASET_DIR),
        dataset_type=fo.types.FiftyOneDataset,
        name=NAME,
        persistent=True,
    )

print(f"샘플 수: {len(dataset)}")
print("App 실행 중... 종료하려면 Ctrl+C")
session = fo.launch_app(dataset)
session.wait()
