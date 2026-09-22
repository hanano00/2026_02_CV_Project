import argparse
import glob
import json
import os

import cv2


class Annotator:

    def __init__(self, frame_dir, annotation_path, object_type):
        self.frame_dir = frame_dir
        self.annotation_path = annotation_path
        self.object_type = object_type

        self.frame_paths = sorted(
            glob.glob(
                os.path.join(frame_dir, "*.jpg")
            )
        )

        self.current_index = 0

        # 현재 프레임의 Box
        self.boxes = []

        # 마우스로 Box를 그리고 있는 중인지
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_box = None

        self.annotations = self.load_annotations()

    def load_annotations(self):

        if not os.path.exists(self.annotation_path):
            return {}

        with open(
            self.annotation_path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    def save_annotations(self):

        os.makedirs(
            os.path.dirname(self.annotation_path),
            exist_ok=True
        )

        with open(
            self.annotation_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.annotations,
                f,
                indent=4,
                ensure_ascii=False
            )

    def get_frame_number(self):

        filename = os.path.basename(
            self.frame_paths[self.current_index]
        )

        return int(
            filename.split("_")[1].split(".")[0]
        )

    def load_current_annotation(self):

        frame_number = self.get_frame_number()

        annotation = self.annotations.get(
            str(frame_number)
        )

        if annotation is None:
            self.boxes = []

        else:
            self.boxes = [
                (
                    obj["x1"],
                    obj["y1"],
                    obj["x2"],
                    obj["y2"]
                )
                for obj in annotation["objects"]
            ]

    def save_current_annotation(self):

        frame_number = self.get_frame_number()

        self.annotations[str(frame_number)] = {

            "objects": [

                {
                    "class": self.object_type,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }

                for x1, y1, x2, y2 in self.boxes
            ]
        }

        self.save_annotations()

        print(
            f"Saved: frame {frame_number}"
        )

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.drawing = True

            self.start_x = x
            self.start_y = y

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.drawing:

                self.current_box = (
                    self.start_x,
                    self.start_y,
                    x,
                    y
                )

        elif event == cv2.EVENT_LBUTTONUP:

            self.drawing = False

            x1 = min(self.start_x, x)
            y1 = min(self.start_y, y)

            x2 = max(self.start_x, x)
            y2 = max(self.start_y, y)

            self.boxes.append(
                (x1, y1, x2, y2)
            )

            self.current_box = None

    def draw(self, frame):

        for i, box in enumerate(self.boxes):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{self.object_type} {i + 1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if self.current_box is not None:

            x1, y1, x2, y2 = self.current_box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

        frame_number = self.get_frame_number()

        cv2.putText(
            frame,
            f"{self.object_type.upper()} | "
            f"Frame: {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "A: Previous  D: Next  "
            "S: Save  R: Reset  Q: Quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return frame

    def run(self):

        window_name = (
            f"Tennis Annotator - {self.object_type}"
        )

        cv2.namedWindow(window_name)

        cv2.setMouseCallback(
            window_name,
            self.mouse_callback
        )

        self.load_current_annotation()

        while True:

            frame = cv2.imread(
                self.frame_paths[self.current_index]
            )

            if frame is None:
                print("프레임을 읽을 수 없습니다.")
                break

            display = self.draw(
                frame.copy()
            )

            cv2.imshow(
                window_name,
                display
            )

            key = cv2.waitKey(30) & 0xFF

            # 이전 프레임
            if key == ord("a"):

                if self.current_index > 0:

                    self.current_index -= 1

                    self.load_current_annotation()

            # 다음 프레임
            elif key == ord("d"):

                if (
                    self.current_index
                    < len(self.frame_paths) - 1
                ):

                    self.current_index += 1

                    self.load_current_annotation()

            # 저장
            elif key == ord("s"):

                self.save_current_annotation()

            # 현재 Box 삭제
            elif key == ord("r"):

                self.boxes = []
                self.current_box = None

            # 종료
            elif key == ord("q"):

                self.save_annotations()
                break

        cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(
        description="Tennis Bounding Box Annotator"
    )

    parser.add_argument(
        "--type",
        choices=["player", "ball"],
        required=True,
        help="annotation 대상: player 또는 ball"
    )

    parser.add_argument(
        "--frames",
        required=True,
        help="프레임 폴더"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="annotation JSON 파일"
    )

    args = parser.parse_args()

    annotator = Annotator(
        frame_dir=args.frames,
        annotation_path=args.output,
        object_type=args.type
    )

    if not annotator.frame_paths:
        print(
            f"프레임이 없습니다: {args.frames}"
        )
        return

    print(
        f"Object type : {args.type}"
    )

    print(
        f"Frames      : "
        f"{len(annotator.frame_paths)}"
    )

    annotator.run()


if __name__ == "__main__":
    main()

# Command to run the annotator:
# python src/annotator.py --type player --frames path/to/frames --output path/to/annotations.json
