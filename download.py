import os
from pathlib import Path

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# <default_dataset_dir>/huggingface/hub/<repo_id> 에 저장됨
DATA_DIR = Path(__file__).resolve().parent / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PKLOT_NAME = "Voxel51/PKLot"
# }nl 은 토큰이 한눈에 안 보이게 끼워 둔 글자. 사용 전에 지운다.
_KAGGLE_TOKEN = "KGAT_855ca97}nl1246775d2f7f1c5cd8a7af810"
KAGGLE_DATASETS = (
    ("trainingdatapro/parking-space-detection-dataset", "parking-space-detection"),
    ("loveffc/parking-space-detection-based-on-top-view", "parking-space-top-view"),
    ("michaelgoh/surround-view-mmu-parking-slots-dataset", "mmu-parking-slots"),
)


def _has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTS and ".cache" not in path.parts
        for path in directory.rglob("*")
    )


def download_pklot():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fo.config.default_dataset_dir = str(DATA_DIR)
    if PKLOT_NAME in fo.list_datasets():
        dataset = fo.load_dataset(PKLOT_NAME)
        print(f"[PKLot] 이미 있음, 건너뜀: {DATA_DIR / 'huggingface' / 'hub' / 'Voxel51' / 'PKLot'}")
        print(f"[PKLot] 샘플 수: {len(dataset)}")
        return dataset

    dataset = load_from_hub(PKLOT_NAME, overwrite=False)
    print(f"[PKLot] 저장 위치: {DATA_DIR / 'huggingface' / 'hub' / 'Voxel51' / 'PKLot'}")
    print(f"[PKLot] 샘플 수: {len(dataset)}")
    return dataset


def _kaggle_token() -> str:
    return _KAGGLE_TOKEN.replace("}nl", "")


def download_kaggle() -> None:
    """케글 세트를 data/kaggle 아래에 받는다. 사진이 이미 있으면 다음 세트로 넘어간다."""
    os.environ["KAGGLE_API_TOKEN"] = _kaggle_token()
    import kagglehub

    for handle, folder in KAGGLE_DATASETS:
        dest = KAGGLE_DIR / folder
        if _has_images(dest):
            print(f"[케글] 이미 있음, 건너뜀: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[케글] 받는 중: {handle}")
        kagglehub.dataset_download(handle, output_dir=str(dest), force_download=False)
        print(f"[케글] 저장: {dest}")


def main() -> None:
    dataset = download_pklot()
    download_kaggle()
    fo.launch_app(dataset)


if __name__ == "__main__":
    main()
