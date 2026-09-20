import argparse
import os

from yt_dlp import YoutubeDL


def download_video(url, output_path):

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": output_path,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():

    parser = argparse.ArgumentParser(
        description="YouTube 영상을 다운로드합니다."
    )

    parser.add_argument(
        "url",
        help="YouTube 영상 URL"
    )

    parser.add_argument(
        "--output",
        default="videos/video.mp4",
        help="저장할 파일 경로"
    )

    args = parser.parse_args()

    print(f"URL    : {args.url}")
    print(f"저장 위치: {args.output}")

    download_video(
        args.url,
        args.output
    )

    print("다운로드 완료")


if __name__ == "__main__":
    main()
