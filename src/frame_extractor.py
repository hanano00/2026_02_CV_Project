import cv2
import os
import argparse
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = total_frames / original_fps

    print(f"영상       : {video_path}")
    print(f"해상도     : {width} x {height}")
    print(f"원본 FPS   : {original_fps:.2f}")
    print(f"전체 프레임: {total_frames}")
    print(f"영상 길이  : {duration:.2f}초")
    print(f"샘플링 FPS : {target_fps}")

    if target_fps <= 0:
        raise ValueError("target_fps는 0보다 커야 합니다.")

    if target_fps > original_fps:
        raise ValueError(
            "샘플링 FPS가 원본 FPS보다 클 수 없습니다."
        )

    os.makedirs(output_dir, exist_ok=True)

    # 이미 추출된 프레임이 있는지 확인
    existing_files = os.listdir(output_dir)

    if existing_files:
        print(f"\n경고: {output_dir}에 파일이 있습니다.")

        answer = input("계속 진행할까요? [y/N]: ")

        if answer.lower() != "y":
            print("취소했습니다.")
            cap.release()
            return

    # 예:
    # 원본 60 FPS
    # target 2 FPS
    # → 30 프레임마다 저장
    interval = original_fps / target_fps

    next_frame_to_save = 0
    saved_count = 0

    progress = tqdm(
        total=total_frames,
        desc="Extracting"
    )

    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index >= next_frame_to_save:

            filename = os.path.join(
                output_dir,
                f"frame_{frame_index:06d}.jpg"
            )

            success = cv2.imwrite(filename, frame)

            if success:
                saved_count += 1

            next_frame_to_save += interval

        frame_index += 1
        progress.update(1)

    progress.close()
    cap.release()

    print("\n추출 완료")
    print(f"추출된 프레임: {saved_count}")
    print(f"저장 위치: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="영상에서 지정한 FPS로 프레임을 추출합니다."
    )

    parser.add_argument(
        "video",
        help="입력 영상 파일 경로"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2,
        help="샘플링 FPS (기본값: 2)"
    )

    parser.add_argument(
        "--output",
        default="frames",
        help="프레임 저장 폴더 (기본값: frames)"
    )

    args = parser.parse_args()

    extract_frames(
        args.video,
        args.output,
        args.fps
    )


if __name__ == "__main__":
    main()

# Bash Command to run the script:
# python src/frame_extractor.py <video_path> --fps <target_fps> --output <output_directory>
