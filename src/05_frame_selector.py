"""랠리 프레임 중 학습에 사용할 이미지를 선택하고 원본 정보를 함께 보존한다.

S는 선택/해제, W는 중간 저장, Q 또는 창 닫기는 저장 후 종료다.
이전 결과는 출처와 파일 내용을 확인한 뒤 불러오며 원본 이미지는 삭제하지 않는다.
"""

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import cv2

from common import frame_number, list_frames, load_frame_metadata, write_json_atomic


def _same_contents(first, second):
    """파일 크기나 수정 시각이 같아도 다른 영상일 수 있어 실제 내용을 비교한다."""
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        return (
            hashlib.file_digest(left, "sha256").digest()
            == hashlib.file_digest(right, "sha256").digest()
        )


class FrameSelector:
    """원본과 검증된 출력 사본을 구분하여 선택 상태를 관리한다."""

    def __init__(self, frame_dir, output_dir):
        self.frame_dir = Path(frame_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self._check_distinct_directories()
        self.frame_paths = list_frames(self.frame_dir)
        self._sources = {path.name: path for path in self.frame_paths}
        self.source_metadata = load_frame_metadata(self.frame_dir)
        self.current_index = 0
        self.step = 5
        self.selected = set()
        self._managed_names = set()
        self._saved_selection = set()
        self.dirty = False
        self._cached_index = None
        self._cached_frame = None
        self.load_selected()

    def _check_distinct_directories(self):
        """심볼릭 링크를 포함해 입력과 출력이 실제로 같은 디렉터리인지 확인한다."""
        if self.frame_dir.resolve() == self.output_dir.resolve():
            raise ValueError("입력 프레임 폴더와 선택 결과 폴더는 달라야 합니다.")

    def _validated_output_names(self):
        """다른 출처의 이미지가 섞인 출력을 거부하고 검증한 사본 이름만 반환한다."""
        self._check_distinct_directories()
        if not self.output_dir.exists():
            return set()
        if not self.output_dir.is_dir():
            raise ValueError(f"출력 경로가 폴더가 아닙니다: {self.output_dir}")

        metadata = load_frame_metadata(self.output_dir)
        metadata_path = self.output_dir / "selection_metadata.json"
        if metadata_path.exists():
            source_dir = metadata.get("source_frame_dir")
            if (
                not isinstance(source_dir, str)
                or Path(source_dir).resolve() != self.frame_dir
            ):
                raise ValueError(
                    "기존 선택 결과의 원본 폴더가 다릅니다. 새 출력 폴더를 지정하세요."
                )
        elif any(
            (self.output_dir / name).exists()
            for name in ("rally_metadata.json", "frames_metadata.json")
        ):
            raise ValueError(
                "프레임 추출/랠리 폴더를 선택 결과 폴더로 재사용할 수 없습니다."
            )

        names = set()
        for destination in list_frames(self.output_dir):
            source = self._sources.get(destination.name)
            if (
                source is None
                or destination.is_symlink()
                or not _same_contents(source, destination)
            ):
                raise ValueError(
                    f"입력과 일치하지 않는 기존 출력 이미지입니다: {destination}. "
                    "다른 출력 폴더를 지정하세요."
                )
            names.add(destination.name)
        return names

    def load_selected(self):
        """메타데이터가 없는 과거 결과도 원본과 내용이 같은 사본에 한해 이어서 연다."""
        self.selected = self._validated_output_names()
        self._managed_names = set(self.selected)
        self._saved_selection = set(self.selected)
        self.dirty = False

    def get_filename(self):
        """현재 화면의 프레임 파일명을 반환한다."""
        return self.frame_paths[self.current_index].name

    def get_frame_number(self):
        """현재 프레임의 원본 영상 내 번호를 반환한다."""
        return frame_number(self.frame_paths[self.current_index])

    def toggle_selection(self):
        """선택 상태만 바꾸며 저장 시점까지 디스크의 파일은 그대로 둔다."""
        filename = self.get_filename()
        if filename in self.selected:
            self.selected.remove(filename)
            print(f"선택 해제: {filename}")
        else:
            self.selected.add(filename)
            print(f"선택: {filename}")
        self.dirty = self.selected != self._saved_selection

    def _copy_frame(self, source, destination):
        """이미지를 임시 파일에 복사한 뒤 기존 파일을 덮어쓰지 않고 공개한다."""
        with tempfile.NamedTemporaryFile(
            prefix=".frame-", dir=self.output_dir, delete=False
        ) as f:
            temporary = Path(f.name)
        try:
            shutil.copy2(source, temporary)
            # 같은 파일 시스템의 하드 링크 생성은 완성된 파일을 한 번에 공개하고,
            # 목적지가 이미 있다면 실패하므로 검사 이후의 충돌도 덮어쓰지 않는다.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save_selected(self):
        """검증한 사본만 추가/삭제하고 선택 목록과 영상 정보를 원자적으로 기록한다."""
        if not self.selected <= self._sources.keys():
            raise ValueError("입력 폴더에 없는 프레임이 선택되어 있습니다.")
        existing = self._validated_output_names()
        # 실행 도중 추가된 출력은 이 세션이 관리하지 않으므로 지우거나 덮어쓰지 않는다.
        if existing - self._managed_names:
            raise ValueError(
                "작업 도중 출력 이미지가 추가되었습니다. 다시 열어 확인하세요."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in sorted(self.selected, key=frame_number):
            if filename not in existing:
                self._copy_frame(self._sources[filename], self.output_dir / filename)
                self._managed_names.add(filename)
        for filename in self._managed_names - self.selected:
            (self.output_dir / filename).unlink(missing_ok=True)

        metadata = dict(self.source_metadata)
        metadata.update(
            source_frame_dir=str(self.frame_dir),
            selected_frames=sorted(frame_number(name) for name in self.selected),
            frame_count=len(self.selected),
        )
        write_json_atomic(self.output_dir / "selection_metadata.json", metadata)
        self._managed_names = set(self.selected)
        self._saved_selection = set(self.selected)
        self.dirty = False
        print(f"선택된 프레임: {len(self.selected)}개 (저장 완료)")

    def draw(self, frame):
        """선택 상태와 아직 저장되지 않은 변경 여부를 화면에 표시한다."""
        selected = self.get_filename() in self.selected
        white = (255, 255, 255)
        lines = [
            (f"Frame: {self.get_frame_number()}", white),
            (
                "SELECTED" if selected else "NOT SELECTED",
                (0, 255, 0) if selected else (0, 0, 255),
            ),
            (f"A/D: Move   Step: {self.step}   [ / ]: Change Step", white),
            ("S: Select   W: Save   Q: Save and Quit", white),
            (
                f"Selected: {len(self.selected)}   {'UNSAVED' if self.dirty else 'SAVED'}",
                white,
            ),
        ]
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

    def handle_key(self, key):
        """기존 탐색 키와 S 토글을 유지하고 W 중간 저장을 제공한다."""
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
            self.toggle_selection()
        elif key == ord("w"):
            self.save_selected()
        elif key == ord("q"):
            self.save_selected()
            return False
        return True

    def _current_frame(self):
        """화면 위치가 바뀔 때만 이미지를 읽는다."""
        if self._cached_index != self.current_index:
            path = self.frame_paths[self.current_index]
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"프레임을 읽을 수 없습니다: {path}")
            self._cached_frame = frame
            self._cached_index = self.current_index
        return self._cached_frame

    def run(self):
        """GUI에서 선택하고 Q 또는 창 닫기로 종료하면 선택 결과를 저장한다."""
        if not self.frame_paths:
            print(f"프레임이 없습니다: {self.frame_dir}")
            return
        window_name = "Frame Selector"
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
                if not visible:
                    self.save_selected()
                    break
                if not self.handle_key(key):
                    break
        finally:
            cv2.destroyAllWindows()


def main():
    """독립 CLI로 입력 랠리와 선택 결과 폴더를 연결한다."""
    parser = argparse.ArgumentParser(description="학습에 사용할 프레임을 선택합니다.")
    parser.add_argument("--frames", required=True, help="입력 프레임 폴더")
    parser.add_argument("--output", required=True, help="선택된 프레임 저장 폴더")
    args = parser.parse_args()
    try:
        FrameSelector(args.frames, args.output).run()
    except (ValueError, OSError) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    main()
