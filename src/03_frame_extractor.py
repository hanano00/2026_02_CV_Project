"""03단계: 원본 프레임 번호를 유지하면서 지정 FPS로 JPG를 추출한다.

사용 예: python src/03_frame_extractor.py video.mp4 --fps 2 --output frames
출력에는 이미지와 원본 영상/FPS를 기록한 frames_metadata.json이 생성된다.
기존 결과와 섞이지 않도록 출력 폴더는 새 폴더 또는 빈 폴더만 허용한다.
"""

import argparse
import math
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm

from common import positive_finite, write_json_atomic


def require_empty_output(output_dir: Path) -> None:
    """기존 파일을 덮어쓰거나 이전 추출 결과를 섞지 않도록 검사한다."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"출력은 새 폴더 또는 빈 폴더를 지정하세요: {output_dir}")


def extract_frames(video_path, output_dir, target_fps):
    """임시 폴더에 추출한 후, 모든 저장이 성공한 결과만 출력 폴더로 옮긴다."""
    target_fps = positive_finite(target_fps)
    output_dir = Path(output_dir).resolve()
    require_empty_output(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    progress = None
    try:
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")
        try:
            original_fps = positive_finite(cap.get(cv2.CAP_PROP_FPS))
        except ValueError as exc:
            raise ValueError("영상의 FPS 정보를 읽을 수 없습니다.") from exc
        if target_fps > original_fps:
            raise ValueError("샘플링 FPS가 원본 FPS보다 클 수 없습니다.")
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count <= 0:
            raise ValueError("영상의 전체 프레임 수를 읽을 수 없습니다.")
        total_frames = int(count)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"영상       : {video_path}")
        print(f"해상도     : {width} x {height}")
        print(f"원본 FPS   : {original_fps:.3f}")
        print(f"전체 프레임: {total_frames}")
        print(f"영상 길이  : {total_frames / original_fps:.2f}초")
        print(f"샘플링 FPS : {target_fps:g}")

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        progress = tqdm(total=total_frames, desc="Extracting")
        saved_count = 0
        decoded_count = 0
        # 같은 파일시스템의 임시 폴더를 써야 완료 시 디렉터리를 한 번에 옮길 수 있다.
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}_extract_", dir=output_dir.parent
        ) as temporary:
            staging = Path(temporary)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index = decoded_count
                # 원본 60 FPS / 목표 2 FPS이면 원본 번호 0, 30, 60...을 저장한다.
                # 누적 덧셈 대신 곱셈으로 비교해 장시간 영상의 부동소수점 오차를 줄인다.
                if frame_index * target_fps >= saved_count * original_fps:
                    path = staging / f"frame_{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        raise OSError(f"프레임 이미지 저장에 실패했습니다: {path.name}")
                    saved_count += 1
                decoded_count += 1
                progress.update(1)

            if decoded_count < total_frames:
                raise ValueError(
                    f"영상을 끝까지 읽지 못했습니다: {decoded_count}/{total_frames} 프레임. "
                    "영상 파일과 프레임 수 정보를 확인하세요."
                )
            metadata = {
                "schema_version": 1,
                "source_video": str(Path(video_path).resolve()),
                "source_fps": original_fps,
                "sampling_fps": target_fps,
                "source_frame_count": total_frames,
                "frame_width": width,
                "frame_height": height,
                "frame_count": saved_count,
            }
            write_json_atomic(staging / "frames_metadata.json", metadata)
            # 실행 중 다른 작업이 출력 폴더를 사용했어도 덮어쓰지 않는다.
            require_empty_output(output_dir)
            staging.replace(output_dir)

        print(f"\n추출 완료: {saved_count}개 프레임")
        print(f"저장 위치: {output_dir}")
    finally:
        if progress is not None:
            progress.close()
        cap.release()


def main():
    """CLI의 FPS와 출력 경로를 전달하고 실패 원인을 사용자에게 알린다."""
    parser = argparse.ArgumentParser(
        description="영상에서 지정한 FPS로 프레임을 추출합니다."
    )
    parser.add_argument("video", help="입력 영상 파일 경로")
    parser.add_argument(
        "--fps", type=positive_finite, default=2, help="샘플링 FPS (기본값: 2)"
    )
    parser.add_argument("--output", default="frames", help="새 폴더 또는 빈 출력 폴더")
    args = parser.parse_args()
    try:
        extract_frames(args.video, args.output, args.fps)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
