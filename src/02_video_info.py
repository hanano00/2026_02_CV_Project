"""02단계: 해상도, FPS, 프레임 수와 재생시간을 확인한다.

사용 예: python src/02_video_info.py datasets/videos/match01.mp4
"""

import argparse
import math

import cv2

from common import positive_finite


def get_video_info(video_path):
    """메타데이터를 출력한다. 조회에 실패해도 영상 핸들은 반드시 닫는다."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")

        # FPS가 0 또는 NaN이면 길이를 계산할 수 없으므로 나누기 전에 검사한다.
        try:
            fps = positive_finite(cap.get(cv2.CAP_PROP_FPS))
        except ValueError as exc:
            raise ValueError("영상의 FPS 정보를 읽을 수 없습니다.") from exc
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count <= 0:
            raise ValueError("영상의 전체 프레임 수를 읽을 수 없습니다.")
        frame_count = int(count)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        seconds = int(frame_count / fps)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("\n===== Video Information =====")
        print(f"파일       : {video_path}")
        print(f"해상도     : {width} x {height}")
        print(f"FPS        : {fps:.3f}")
        print(f"전체 프레임 : {frame_count}")
        print(f"재생시간   : {hours:02d}:{minutes:02d}:{seconds:02d}")
    finally:
        cap.release()


def main():
    """CLI 인자를 받아 조회하고, 잘못된 입력은 설명과 함께 종료한다."""
    parser = argparse.ArgumentParser(description="영상 정보를 확인합니다.")
    parser.add_argument("video", help="영상 파일 경로")
    args = parser.parse_args()
    try:
        get_video_info(args.video)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
