"""CLI 공통 프레임 이름 규칙과 안전한 JSON 입출력.

프레임 번호는 원본 영상의 프레임 번호다. 숫자 순으로 정렬해야
긴 영상에서도 랠리의 시간 순서가 보존된다.
"""

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

FRAME_NAME = re.compile(r"frame_(\d+)\.jpg", re.IGNORECASE)
METADATA_FILES = (
    "selection_metadata.json",
    "rally_metadata.json",
    "frames_metadata.json",
)


def frame_number(path: str | Path) -> int:
    """frame_000123.jpg에서 원본 프레임 번호 123을 얻는다."""
    match = FRAME_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"프레임 이름은 frame_<번호>.jpg 형식이어야 합니다: {path}")
    return int(match.group(1))


def list_frames(frame_dir: str | Path) -> list[Path]:
    """입력 폴더의 JPG를 검증하고 프레임 번호순으로 반환한다."""
    folder = Path(frame_dir)
    if not folder.is_dir():
        raise ValueError(f"프레임 폴더가 존재하지 않습니다: {folder}")
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
    numbered = [(frame_number(path), path) for path in paths]
    # 자릿수가 다른 같은 번호의 파일은 하나의 주석 키를 공유하므로 거부한다.
    if len({number for number, _ in numbered}) != len(numbered):
        raise ValueError(f"같은 프레임 번호의 이미지가 중복되어 있습니다: {folder}")
    return [path for _, path in sorted(numbered, key=lambda item: item[0])]


def positive_finite(value: str | float) -> float:
    """FPS 등 양수 입력을 검사한다. nan/inf도 유효한 값으로 받지 않는다."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("0보다 큰 유한한 숫자를 입력해야 합니다.")
    return number


def write_json_atomic(path: str | Path, data: Any) -> None:
    """임시 파일을 완성한 뒤 교체하여 저장 실패 시 기존 JSON을 보존한다."""
    destination = Path(path)
    # 파일명만 주어져도 parent는 '.'이므로 현재 폴더에 정상 저장된다.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(data, temporary, indent=4, ensure_ascii=False, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_frame_metadata(frame_dir: str | Path) -> dict:
    """가장 최근 처리 단계의 메타데이터를 읽는다. 과거 데이터는 {}를 반환한다."""
    for filename in METADATA_FILES:
        path = Path(frame_dir) / filename
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError(f"메타데이터는 JSON 객체여야 합니다: {path}")
            return data
    return {}
