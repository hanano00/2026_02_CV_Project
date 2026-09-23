"""추출된 프레임에서 랠리의 시작과 끝을 지정해 별도 폴더로 복사한다.

A/D는 이동, S/E는 구간 지정, N은 랠리 저장이다. 원본 프레임 번호와
알려진 영상 메타데이터를 보존하며, 기존 랠리 폴더에 덮어쓰지 않는다.
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import cv2

from common import frame_number, list_frames, load_frame_metadata, write_json_atomic


class RallySplitter:
    """시간순으로 프레임을 탐색하고 지정한 양 끝을 포함한 구간을 저장한다."""

    def __init__(self, frame_dir, output_dir):
        self.frame_dir = Path(frame_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.frame_paths = list_frames(self.frame_dir)
        self.source_metadata = load_frame_metadata(self.frame_dir)
        self.current_index = 0
        self.start_index = None
        self.end_index = None
        self.step = 1
        self._cached_index = None
        self._cached_frame = None

    def get_frame_number(self, index):
        """탐색 인덱스를 원본 영상의 프레임 번호로 바꾼다."""
        return frame_number(self.frame_paths[index])

    def draw(self, frame):
        """원본을 변경하지 않도록 호출자가 전달한 화면 사본에 안내를 그린다."""
        lines = [
            (f"Frame: {self.get_frame_number(self.current_index)}", (255, 255, 255)),
            ("A/D: Move   [ / ]: Change Step", (255, 255, 255)),
            ("S: Start   E: End   N: Save Rally   Q: Quit", (255, 255, 255)),
            (f"Step: {self.step}", (255, 255, 255)),
        ]
        if self.start_index is not None:
            lines.append(
                (f"Start: {self.get_frame_number(self.start_index)}", (0, 255, 0))
            )
        if self.end_index is not None:
            lines.append((f"End: {self.get_frame_number(self.end_index)}", (0, 255, 0)))
        for row, (text, color) in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (20, 35 + row * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
        return frame

    def _next_rally_number(self):
        """삭제된 번호를 재사용하지 않도록 기존 숫자 디렉터리의 최댓값을 찾는다."""
        numbers = []
        for path in self.output_dir.iterdir():
            match = re.fullmatch(r"rally_(\d+)", path.name)
            if match and path.is_dir():
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def save_rally(self):
        """완성된 구간과 메타데이터를 충돌하지 않는 새 랠리 폴더에 저장한다."""
        if self.start_index is None or self.end_index is None:
            print("시작과 끝을 모두 지정해야 합니다.")
            return None

        start, end = sorted((self.start_index, self.end_index))
        if not 0 <= start <= end < len(self.frame_paths):
            raise ValueError("랠리 범위가 입력 프레임 범위를 벗어났습니다.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        rally_number = self._next_rally_number()
        metadata = dict(self.source_metadata)
        metadata.update(
            source_frame_dir=str(self.frame_dir),
            start_frame=self.get_frame_number(start),
            end_frame=self.get_frame_number(end),
            frame_count=end - start + 1,
        )
        # 복사에 실패한 구간이 정상 랠리처럼 보이지 않도록 숨겨진 임시 폴더에서 준비한다.
        with tempfile.TemporaryDirectory(prefix=".rally-", dir=self.output_dir) as temp:
            staging_dir = Path(temp)
            for source in self.frame_paths[start : end + 1]:
                shutil.copy2(source, staging_dir / source.name)
            write_json_atomic(staging_dir / "rally_metadata.json", metadata)

            # 다른 실행이 같은 번호를 먼저 사용해도 기존 데이터에 합쳐 쓰지 않는다.
            while True:
                rally_dir = self.output_dir / f"rally_{rally_number:04d}"
                try:
                    rally_dir.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    rally_number += 1
            try:
                staging_dir.replace(rally_dir)
            except BaseException:
                # 이 실행이 확보한 빈 폴더만 정리한다. 내용이 생겼다면 건드리지 않는다.
                try:
                    rally_dir.rmdir()
                except OSError:
                    pass
                raise

        print(f"Rally {rally_number:04d} 저장")
        print(f"Frame: {metadata['start_frame']} ~ {metadata['end_frame']}")
        print(f"저장 위치: {rally_dir}")
        self.start_index = None
        self.end_index = None
        return rally_dir

    def handle_key(self, key):
        """키 입력을 처리하고, 종료 요청이면 False를 반환한다."""
        if key == ord("a"):
            self.current_index = max(0, self.current_index - self.step)
        elif key == ord("d"):
            self.current_index = min(
                len(self.frame_paths) - 1, self.current_index + self.step
            )
        elif key == ord("["):
            self.step = max(1, self.step - 1)
        elif key == ord("]"):
            self.step += 1
        elif key == ord("s"):
            self.start_index = self.current_index
            print("Start:", self.get_frame_number(self.current_index))
        elif key == ord("e"):
            self.end_index = self.current_index
            print("End:", self.get_frame_number(self.current_index))
        elif key == ord("n"):
            self.save_rally()
        elif key == ord("q"):
            return False
        return True

    def _current_frame(self):
        """위치가 달라질 때만 이미지를 다시 읽어 같은 프레임의 반복 디코딩을 피한다."""
        if self._cached_index != self.current_index:
            path = self.frame_paths[self.current_index]
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"프레임을 읽을 수 없습니다: {path}")
            self._cached_frame = frame
            self._cached_index = self.current_index
        return self._cached_frame

    def run(self):
        """GUI 탐색을 실행하고 정상 종료와 예외 모두에서 창을 정리한다."""
        if not self.frame_paths:
            print(f"프레임이 없습니다: {self.frame_dir}")
            return
        window_name = "Rally Splitter"
        try:
            cv2.namedWindow(window_name)
            while True:
                cv2.imshow(window_name, self.draw(self._current_frame().copy()))
                key = cv2.waitKey(30) & 0xFF
                try:
                    visible = (
                        cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
                    )
                except cv2.error:
                    visible = False
                if not visible or not self.handle_key(key):
                    break
        finally:
            cv2.destroyAllWindows()
        if self.start_index is not None or self.end_index is not None:
            print("N으로 저장하지 않은 랠리 구간은 반영되지 않았습니다.")


def main():
    """CLI 경로를 받아 랠리 분할기를 실행한다."""
    parser = argparse.ArgumentParser(description="프레임을 랠리 단위로 분리합니다.")
    parser.add_argument("--frames", required=True, help="1차 추출 프레임 폴더")
    parser.add_argument("--output", required=True, help="랠리 저장 폴더")
    args = parser.parse_args()
    try:
        RallySplitter(args.frames, args.output).run()
    except (ValueError, OSError) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    main()
