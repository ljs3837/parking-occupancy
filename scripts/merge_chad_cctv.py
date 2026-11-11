"""CHAD_Videos에서 앞번호 1~4 영상을 번호별·파일명순으로 하나씩 합친다.

원본은 삭제하지 않는다. C 드라이브 용량이 부족하므로 결과 파일은 D:\\parking-cctv 에 쓰고,
data/cctv/1.mp4 같은 바로가기로 연결한다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

SRC = Path(r"c:\Users\user1\soo\project\parking-occupancy\data\cctv\CHAD_Videos")
OUT_DIR = Path(r"D:\parking-cctv")
LINK_DIR = Path(r"c:\Users\user1\soo\project\parking-occupancy\data\cctv")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BATCH = 8


def sources(camera: str) -> list[Path]:
    """카메라별 001~010만. 원본 파일명은 1_001_0.mp4 형태."""
    files = []
    for path in SRC.glob(f"{camera}_*.mp4"):
        parts = path.stem.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        if int(parts[1]) <= 10:
            files.append(path)
    return sorted(files, key=lambda path: path.name)


def encode_batch(files: list[Path], dest: Path) -> None:
    filters = []
    for index in range(len(files)):
        filters.append(
            f"[{index}:v]scale=1280:720,fps=30,setsar=1,format=yuv420p[v{index}]"
        )
    filters.append(
        "".join(f"[v{index}]" for index in range(len(files)))
        + f"concat=n={len(files)}:v=1:a=0[v]"
    )
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for path in files:
        cmd += ["-i", str(path)]
    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-an",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def concat_parts(parts: list[Path], dest: Path) -> None:
    listing = dest.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{part.as_posix()}'\n" for part in parts),
        encoding="utf-8",
    )
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(dest),
        ],
        check=True,
    )
    listing.unlink(missing_ok=True)


def link_into_project(camera: str, target: Path) -> None:
    """파일 심볼릭 링크는 권한이 필요해서, 폴더 연결(merged)만 사용한다."""
    link_dir = LINK_DIR / "merged"
    if link_dir.exists():
        return
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_dir), str(target.parent)],
        check=False,
    )


def merge_camera(camera: str) -> None:
    final = OUT_DIR / f"{camera}.mp4"
    if final.is_file() and final.stat().st_size > 0:
        print(f"[CAM {camera}] 이미 있음 {final}", flush=True)
        link_into_project(camera, final)
        return
    files = sources(camera)
    print(f"[CAM {camera}] {len(files)}개", flush=True)
    work = OUT_DIR / f"_{camera}"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, len(files), BATCH):
        chunk = files[start : start + BATCH]
        part = work / f"part_{start:04d}.mp4"
        print(f"[CAM {camera}] {chunk[0].name} ~ {chunk[-1].name}", flush=True)
        encode_batch(chunk, part)
        parts.append(part)
    concat_parts(parts, final)
    for part in parts:
        part.unlink(missing_ok=True)
    link_into_project(camera, final)
    print(f"[CAM {camera}] 완료 {final}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for camera in ("1", "2", "3", "4"):
        merge_camera(camera)


if __name__ == "__main__":
    main()
