from pathlib import Path

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# <default_dataset_dir>/huggingface/hub/<repo_id> 에 저장됨
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
fo.config.default_dataset_dir = str(DATA_DIR)

# 테스트용: 전체(~4GB) 대신 일부만
dataset = load_from_hub(
    "Voxel51/PKLot",
    overwrite=True,
)

print(f"저장 위치: {DATA_DIR / 'huggingface' / 'hub' / 'Voxel51' / 'PKLot'}")
print(f"샘플 수: {len(dataset)}")

session = fo.launch_app(dataset)

