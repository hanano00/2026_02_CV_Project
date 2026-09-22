import argparse
import os

from yt_dlp import YoutubeDL

# uv run yt-dlp -F "URL" : 다운로드 가능한 영상정보 확인


def download_video(url, output_path, fps):
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    format_selector = (
        f"bestvideo[height<=1080][fps={fps}]"
        "+bestaudio/"
        f"best[height<=1080][fps={fps}]"
    )

    ydl_opts = {
        "format": format_selector,
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

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="원하는 FPS (기본값: 25)"
    )

    args = parser.parse_args()

    print(f"URL      : {args.url}")
    print(f"저장 위치 : {args.output}")
    print(f"요청 FPS  : {args.fps}")

    download_video(
        args.url,
        args.output,
        args.fps
    )

    print("다운로드 완료")


if __name__ == "__main__":
    main()
