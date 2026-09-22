import os
import glob
import shutil
import argparse
import cv2


class FrameSelector:

    def __init__(self, frame_dir, output_dir):

        self.frame_dir = frame_dir
        self.output_dir = output_dir

        self.frame_paths = sorted(
            glob.glob(
                os.path.join(frame_dir, "*.jpg")
            )
        )

        self.current_index = 0

        # 한 번에 이동할 프레임 수
        self.step = 5

        # 선택된 프레임
        self.selected = set()

        self.load_selected()

    def load_selected(self):

        if not os.path.exists(self.output_dir):
            return

        selected_paths = glob.glob(
            os.path.join(
                self.output_dir,
                "*.jpg"
            )
        )

        for path in selected_paths:

            self.selected.add(
                os.path.basename(path)
            )

    def get_filename(self):

        return os.path.basename(
            self.frame_paths[self.current_index]
        )

    def get_frame_number(self):

        filename = self.get_filename()

        return int(
            filename.split("_")[1].split(".")[0]
        )

    def toggle_selection(self):

        filename = self.get_filename()

        if filename in self.selected:

            self.selected.remove(filename)

            print(
                f"선택 해제: {filename}"
            )

        else:

            self.selected.add(filename)

            print(
                f"선택: {filename}"
            )

    def save_selected(self):

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        for frame_path in self.frame_paths:

            filename = os.path.basename(
                frame_path
            )

            destination = os.path.join(
                self.output_dir,
                filename
            )

            if filename in self.selected:

                if not os.path.exists(destination):

                    shutil.copy2(
                        frame_path,
                        destination
                    )

            else:

                if os.path.exists(destination):

                    os.remove(destination)

        print(
            f"선택된 프레임: "
            f"{len(self.selected)}개"
        )

    def draw(self, frame):

        frame_number = self.get_frame_number()
        filename = self.get_filename()

        if filename in self.selected:

            status = "SELECTED"
            status_color = (0, 255, 0)

        else:

            status = "NOT SELECTED"
            status_color = (0, 0, 255)

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"A/D: Move   Step: {self.step}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "[ / ]: Change Step",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "S: Select   Q: Quit",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Selected: {len(self.selected)}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return frame

    def run(self):

        if not self.frame_paths:

            print(
                f"프레임이 없습니다: "
                f"{self.frame_dir}"
            )

            return

        window_name = "Frame Selector"

        cv2.namedWindow(
            window_name
        )

        while True:

            frame = cv2.imread(
                self.frame_paths[self.current_index]
            )

            if frame is None:

                print(
                    "프레임을 읽을 수 없습니다."
                )

                break

            display = self.draw(
                frame.copy()
            )

            cv2.imshow(
                window_name,
                display
            )

            key = cv2.waitKey(30) & 0xFF

            # 이전
            if key == ord("a"):

                self.current_index = max(
                    0,
                    self.current_index - self.step
                )

            # 다음
            elif key == ord("d"):

                self.current_index = min(
                    len(self.frame_paths) - 1,
                    self.current_index + self.step
                )

            # Step 감소
            elif key == ord("["):

                self.step = max(
                    1,
                    self.step - 1
                )

                print(
                    f"Step: {self.step}"
                )

            # Step 증가
            elif key == ord("]"):

                self.step += 1

                print(
                    f"Step: {self.step}"
                )

            # 선택 / 선택 해제
            elif key == ord("s"):

                self.toggle_selection()

            # 종료
            elif key == ord("q"):

                self.save_selected()

                break

        cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(
        description="학습에 사용할 프레임을 선택합니다."
    )

    parser.add_argument(
        "--frames",
        required=True,
        help="입력 프레임 폴더"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="선택된 프레임 저장 폴더"
    )

    args = parser.parse_args()

    selector = FrameSelector(
        frame_dir=args.frames,
        output_dir=args.output
    )

    selector.run()


if __name__ == "__main__":
    main()
