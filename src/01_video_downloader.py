"""01단계: 지정한 FPS의 YouTube 영상 소스를 최대 1080p로 다운로드한다.

FPS는 변환 옵션이 아니라 제공되는 영상 소스의 선택 조건이다.
먼저 'uv run yt-dlp -F URL'로 해상도와 FPS를 확인할 수 있다.
사용 예: python src/01_video_downloader.py URL --fps 25 --output videos/match01.mp4
영상/오디오를 합치는 다운로드는 별도로 설치된 ffmpeg가 필요하다.
"""

import argparse
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from common import positive_finite


def download_video(url, output_path, fps):
    """조건에 맞는 영상과 오디오를 선택해 하나의 영상으로 저장한다."""
    fps = positive_finite(fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 별도 영상+오디오를 우선 선택하고, 없으면 소리가 포함된 단일 소스를 사용한다.
    # 29.97 같은 FPS도 지정할 수 있다. 조건에 맞는 소스가 없으면 다운로드가 실패한다.
    requested_fps = f"{fps:g}"
    format_selector = (
        f"bestvideo[height<=1080][fps={requested_fps}]+bestaudio/"
        f"best[height<=1080][fps={requested_fps}]"
    )
    ydl_opts = {
        "format": format_selector,
        "merge_output_format": "mp4",
        "outtmpl": str(output_path),
        # 재생목록 정보가 붙은 URL이어도 지정한 영상 하나만 받는다.
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    """다운로드 조건과 저장 위치를 CLI에서 받아 실행한다."""
    parser = argparse.ArgumentParser(description="YouTube 영상을 다운로드합니다.")
    parser.add_argument("url", help="YouTube 영상 URL")
    parser.add_argument("--output", default="videos/video.mp4", help="저장할 파일 경로")
    parser.add_argument(
        "--fps",
        type=positive_finite,
        default=25,
        help="원본 소스의 FPS 조건 (기본값: 25)",
    )
    args = parser.parse_args()
    print(f"URL      : {args.url}")
    print(f"저장 위치 : {args.output}")
    print(f"요청 FPS  : {args.fps:g}")
    try:
        download_video(args.url, args.output, args.fps)
    except (ValueError, OSError, DownloadError) as exc:
        parser.exit(1, f"오류: {exc}\n")
    print("다운로드 완료")


if __name__ == "__main__":
    main()
