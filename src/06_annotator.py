"""선택한 프레임에 선수 또는 공 bbox를 그려 클래스별 JSON으로 저장한다.

마우스 드래그로 박스를 추가하고 A/D로 이동한다. 완성한 편집은 이동과
종료 시 자동 저장하며, S는 객체가 없는 프레임도 검수 완료 상태로 저장한다.
R은 현재 박스를 지우고 Q는 저장 후 종료한다. 좌표는 원본 이미지 기준이다.
"""

import argparse
import json
import math
from pathlib import Path

import cv2

from common import frame_number, list_frames, write_json_atomic


class Annotator:
    """프레임 탐색, 마우스 편집 상태, JSON 저장을 관리한다."""

    def __init__(self, frame_dir, annotation_path, object_type):
        if object_type not in {"player", "ball"}:
            raise ValueError("객체 종류는 player 또는 ball이어야 합니다.")
        self.frame_dir = Path(frame_dir)
        self.annotation_path = Path(annotation_path)
        self.object_type = object_type
        self.frame_paths = list_frames(frame_dir)
        self.current_index = 0
        self.boxes = []
        self.dirty = False

        # 드래그 중인 박스는 완성된 boxes와 분리하여 이동 시 잘못 저장하지 않는다.
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_box = None
        # 화면을 다시 그릴 때마다 JPEG를 읽지 않고, 프레임이 바뀔 때만 읽는다.
        self._cached_index = None
        self._cached_frame = None
        self.annotations = self.load_annotations()

    def load_annotations(self):
        """기존 라벨을 검사해 클래스 혼용이나 손상된 bbox의 덮어쓰기를 막는다."""
        if not self.annotation_path.exists():
            return {}
        try:
            with self.annotation_path.open(encoding="utf-8") as file:
                annotations = json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                f"라벨 JSON을 읽을 수 없습니다: {self.annotation_path}: {error}"
            ) from error
        if not isinstance(annotations, dict):
            raise TypeError("라벨 JSON은 프레임 번호를 키로 갖는 객체여야 합니다.")

        for number, annotation in annotations.items():
            if (
                not number.isascii()
                or not number.isdecimal()
                or str(int(number)) != number
            ):
                raise ValueError(f"잘못된 라벨 프레임 번호: {number!r}")
            if not isinstance(annotation, dict) or not isinstance(
                annotation.get("objects"), list
            ):
                raise TypeError(f"프레임 {number}: objects 배열이 필요합니다.")
            for obj in annotation["objects"]:
                if not isinstance(obj, dict):
                    raise TypeError(
                        f"프레임 {number}: 객체 라벨은 JSON 객체여야 합니다."
                    )
                if obj.get("class") != self.object_type:
                    raise ValueError(
                        f"프레임 {number}: 기존 클래스 {obj.get('class')!r}와 "
                        f"--type {self.object_type}이 다릅니다. 기존 클래스에 맞는 "
                        "--type을 사용하거나 --output에 별도 JSON 파일을 지정하세요."
                    )
                coordinates = [obj.get(name) for name in ("x1", "y1", "x2", "y2")]
                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value == int(value)
                    for value in coordinates
                ):
                    raise ValueError(
                        f"프레임 {number}: bbox 좌표는 유한한 정수여야 합니다."
                    )
                x1, y1, x2, y2 = map(int, coordinates)
                if not (0 <= x1 < x2 and 0 <= y1 < y2):
                    raise ValueError(
                        f"프레임 {number}: bbox는 음수 좌표 없이 양의 너비·높이를 가져야 합니다."
                    )
        return annotations

    def save_annotations(self, annotations=None):
        """임시 파일을 교체하여 저장 도중 기존 JSON이 깨지는 것을 막는다."""
        write_json_atomic(
            self.annotation_path,
            self.annotations if annotations is None else annotations,
        )

    def get_frame_number(self):
        """배열 인덱스가 아닌 파일명에 보존된 원본 영상 프레임 번호를 반환한다."""
        return frame_number(self.frame_paths[self.current_index])

    def load_current_annotation(self):
        """현재 프레임의 저장된 박스를 불러오고 새 편집 상태로 전환한다."""
        annotation = self.annotations.get(str(self.get_frame_number()), {"objects": []})
        self.boxes = [
            tuple(int(obj[name]) for name in ("x1", "y1", "x2", "y2"))
            for obj in annotation["objects"]
        ]
        self.dirty = False
        self.cancel_drag()

    def save_current_annotation(self):
        """현재 박스를 저장한다. 실패하면 메모리의 라벨과 미저장 상태를 유지한다."""
        number = self.get_frame_number()
        candidate = dict(self.annotations)
        candidate[str(number)] = {
            "objects": [
                {
                    "class": self.object_type,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                for x1, y1, x2, y2 in self.boxes
            ]
        }
        try:
            self.save_annotations(candidate)
        except (OSError, ValueError, TypeError) as error:
            # S로 빈 프레임을 검수 완료 처리하려던 경우에도 재시도할 상태를 남긴다.
            self.dirty = True
            print(f"저장 실패: {error}. 현재 편집을 유지합니다. S로 다시 저장하세요.")
            return False

        # 디스크 쓰기에 성공한 뒤에만 저장 완료로 표시한다.
        self.annotations = candidate
        self.dirty = False
        print(f"Saved: frame {number}")
        return True

    def get_frame(self, index=None):
        """프레임을 한 번 읽어 캐시한다. 읽기 실패 시 이전 캐시는 보존한다."""
        if index is None:
            index = self.current_index
        if index != self._cached_index:
            path = self.frame_paths[index]
            frame = cv2.imread(str(path))
            if frame is None:
                raise OSError(f"프레임을 읽을 수 없습니다: {path}")
            self._cached_frame = frame
            self._cached_index = index
        return self._cached_frame

    def cancel_drag(self):
        """완성되지 않은 드래그만 취소하고 이미 추가한 박스는 유지한다."""
        self.drawing = False
        self.current_box = None

    def clamp_point(self, x, y):
        """이미지 밖에서 마우스를 놓아도 좌표가 원본 이미지 경계를 넘지 않게 한다."""
        height, width = self.get_frame().shape[:2]
        return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))

    def mouse_callback(self, event, x, y, flags, param):
        """드래그 방향을 정규화하고 면적이 있는 박스만 편집 목록에 추가한다."""
        if event not in {
            cv2.EVENT_LBUTTONDOWN,
            cv2.EVENT_MOUSEMOVE,
            cv2.EVENT_LBUTTONUP,
        }:
            return
        if event != cv2.EVENT_LBUTTONDOWN and not self.drawing:
            return
        x, y = self.clamp_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_box = None
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current_box = (self.start_x, self.start_y, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            x1, x2 = sorted((self.start_x, x))
            y1, y2 = sorted((self.start_y, y))
            self.cancel_drag()
            # 클릭만 했거나 한 축으로만 움직인 경우에는 학습에 쓸 수 없는 bbox이다.
            if x1 < x2 and y1 < y2:
                self.boxes.append((x1, y1, x2, y2))
                self.dirty = True

    def move_frame(self, offset):
        """완성된 편집을 저장한 뒤 이동한다. 저장·읽기 실패 시 현재 화면에 머문다."""
        self.cancel_drag()
        target = self.current_index + offset
        if not 0 <= target < len(self.frame_paths):
            return False
        if self.dirty and not self.save_current_annotation():
            return False
        try:
            self.get_frame(target)
        except OSError as error:
            print(error)
            return False
        self.current_index = target
        self.load_current_annotation()
        return True

    def request_exit(self):
        """종료 전 변경한 프레임만 저장한다. 보기만 한 프레임은 빈 라벨로 만들지 않는다."""
        self.cancel_drag()
        return not self.dirty or self.save_current_annotation()

    def handle_key(self, key):
        """키 입력을 처리하고 계속 실행할지 반환한다. 대문자 키도 허용한다."""
        if key in (ord("a"), ord("A")):
            self.move_frame(-1)
        elif key in (ord("d"), ord("D")):
            self.move_frame(1)
        elif key in (ord("s"), ord("S")):
            # 빈 boxes도 명시적으로 저장하므로 '미검수'와 '객체 없음'을 구분한다.
            self.save_current_annotation()
        elif key in (ord("r"), ord("R")):
            self.cancel_drag()
            if self.boxes:
                self.boxes = []
                self.dirty = True
        elif key in (ord("q"), ord("Q")):
            return not self.request_exit()
        return True

    def draw(self, frame):
        """원본의 복사본에 완성 박스·드래그 미리보기·저장 상태·조작법을 표시한다."""
        for index, (x1, y1, x2, y2) in enumerate(self.boxes, start=1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{self.object_type} {index}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        number = self.get_frame_number()
        if self.dirty:
            status = "UNSAVED"
        elif str(number) in self.annotations:
            status = "SAVED"
        else:
            status = "UNREVIEWED"
        cv2.putText(
            frame,
            f"{self.object_type.upper()} | Frame: {number} | {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "A: Previous  D: Next  S: Save  R: Reset  Q: Quit | Auto-save on move/quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame

    def run(self):
        """GUI 이벤트 루프를 실행하고 종료·오류 상황에서도 창을 정리한다."""
        if not self.frame_paths:
            print(f"프레임이 없습니다: {self.frame_dir}")
            return
        window_name = f"Tennis Annotator - {self.object_type}"

        def open_window():
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)

        try:
            self.get_frame()
            self.load_current_annotation()
            open_window()
            while True:
                cv2.imshow(window_name, self.draw(self.get_frame().copy()))
                if not self.handle_key(cv2.waitKey(30) & 0xFF):
                    break
                # 창의 닫기 버튼도 Q와 같은 저장 정책을 적용한다.
                try:
                    visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    visible = 0
                if visible < 1:
                    if self.request_exit():
                        break
                    # 저장 실패 시 창을 복원해 메모리의 편집을 유지하고 재시도하게 한다.
                    open_window()
        except OSError, cv2.error:
            if self.dirty:
                self.save_current_annotation()
            raise
        finally:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass  # 창을 닫은 뒤에는 정리할 창이 없을 수 있다.


def main():
    """CLI 인자를 받아 클래스별 주석 작업을 시작한다."""
    parser = argparse.ArgumentParser(description="Tennis Bounding Box Annotator")
    parser.add_argument(
        "--type",
        choices=["player", "ball"],
        required=True,
        help="annotation 대상: player 또는 ball (클래스별 JSON 사용)",
    )
    parser.add_argument("--frames", required=True, help="프레임 폴더")
    parser.add_argument("--output", required=True, help="annotation JSON 파일")
    args = parser.parse_args()
    try:
        annotator = Annotator(args.frames, args.output, args.type)
        print(f"Object type : {args.type}")
        print(f"Frames      : {len(annotator.frame_paths)}")
        annotator.run()
    except (OSError, ValueError, TypeError, cv2.error) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    main()

# 예: python src/06_annotator.py --type player --frames data/selected --output annotations/player.json
