import os
import shutil
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


def _partial_archive(dest: Path) -> Path | None:
    if not dest.is_dir():
        return None
    archives = [path for path in dest.glob("*.archive") if path.is_file() and path.stat().st_size > 0]
    if len(archives) == 1:
        return archives[0]
    return None


def _resume_archive(handle: str, dest: Path, archive: Path) -> None:
    """미완성 압축이 있으면 Range로 이어서 받고 푼다. 버전이 다르면 예외를 낸다."""
    from kagglehub.cache import Cache, _get_override_archive_name
    from kagglehub.clients import build_kaggle_client, download_file
    from kagglehub.exceptions import handle_call
    from kagglehub.handle import parse_dataset_handle
    from kagglehub.http_resolver import (
        _build_dataset_download_request,
        _extract_archive,
        _get_current_version,
    )

    dataset = parse_dataset_handle(handle)
    with build_kaggle_client() as api_client:
        if not dataset.is_versioned():
            dataset = dataset.with_version(_get_current_version(api_client, dataset))
        expected = dest / _get_override_archive_name(dataset)
        if archive.resolve() != expected.resolve():
            raise FileNotFoundError(f"압축 버전이 다름: {archive.name}")
        request = _build_dataset_download_request(dataset, None)
        response = handle_call(
            lambda: api_client.datasets.dataset_api_client.download_dataset(request),
            dataset,
        )
        print(f"[케글] 이어서 받는 중: {archive} ({archive.stat().st_size} bytes)")
        download_file(response, str(archive), dataset)
        _extract_archive(str(archive), str(dest))
        archive.unlink(missing_ok=True)
        Cache(override_dir=str(dest)).mark_as_complete(dataset, None)


def download_kaggle() -> None:
    """케글 세트를 data/kaggle 아래에 받는다. 사진이 이미 있으면 다음 세트로 넘어간다."""
    os.environ["KAGGLE_API_TOKEN"] = _kaggle_token()
    import kagglehub

    for handle, folder in KAGGLE_DATASETS:
        dest = KAGGLE_DIR / folder
        if _has_images(dest):
            print(f"[케글] 이미 있음, 건너뜀: {dest}")
            continue
        archive = _partial_archive(dest)
        if archive is not None:
            try:
                _resume_archive(handle, dest, archive)
                print(f"[케글] 저장: {dest}")
                continue
            except Exception as exc:
                print(f"[케글] 이어받기 실패, 지우고 다시 받음: {exc}")
                shutil.rmtree(dest, ignore_errors=True)
        elif dest.exists():
            print(f"[케글] 받은 사진이 없는 폴더라 지우고 다시 받음: {dest}")
            shutil.rmtree(dest, ignore_errors=True)
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
