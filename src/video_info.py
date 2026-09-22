import argparse
import cv2


def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = frame_count / fps

    cap.release()

    print()
    print("===== Video Information =====")
    print(f"파일       : {video_path}")
    print(f"해상도     : {width} x {height}")
    print(f"FPS        : {fps:.3f}")
    print(f"전체 프레임 : {frame_count}")
    print(f"재생시간   : {duration:.2f}초")
    print(f"재생시간   : {duration / 60:.2f}분")


def main():

    parser = argparse.ArgumentParser(
        description="영상 정보를 확인합니다."
    )

    parser.add_argument(
        "video",
        help="영상 파일 경로"
    )

    args = parser.parse_args()

    get_video_info(args.video)


if __name__ == "__main__":
    main()
