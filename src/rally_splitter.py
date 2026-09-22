import argparse
import glob
import os
import shutil

import cv2


class RallySplitter:

    def __init__(self, frame_dir, output_dir):

        self.frame_dir = frame_dir
        self.output_dir = output_dir

        self.frame_paths = sorted(
            glob.glob(
                os.path.join(frame_dir, "*.jpg")
            )
        )

        self.current_index = 0

        # 현재 지정한 랠리의 시작/끝
        self.start_index = None
        self.end_index = None

        # 기본 프레임 이동 폭
        self.step = 1

    def get_frame_number(self, index):

        filename = os.path.basename(
            self.frame_paths[index]
        )

        return int(
            filename.split("_")[1].split(".")[0]
        )

    def draw(self, frame):

        frame_number = self.get_frame_number(
            self.current_index
        )

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
            "A: Previous  D: Next",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "S: Start  E: End  N: Save Rally  Q: Quit",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if self.start_index is not None:

            start_frame = self.get_frame_number(
                self.start_index
            )

            cv2.putText(
                frame,
                f"Start: {start_frame}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if self.end_index is not None:

            end_frame = self.get_frame_number(
                self.end_index
            )

            cv2.putText(
                frame,
                f"End: {end_frame}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame

    def save_rally(self):

        if (
            self.start_index is None
            or self.end_index is None
        ):
            print("시작과 끝을 모두 지정해야 합니다.")
            return

        start = min(
            self.start_index,
            self.end_index
        )

        end = max(
            self.start_index,
            self.end_index
        )

        rally_number = len(
            glob.glob(
                os.path.join(
                    self.output_dir,
                    "rally_*"
                )
            )
        ) + 1

        rally_dir = os.path.join(
            self.output_dir,
            f"rally_{rally_number:04d}"
        )

        os.makedirs(
            rally_dir,
            exist_ok=True
        )

        for i in range(start, end + 1):

            source = self.frame_paths[i]

            destination = os.path.join(
                rally_dir,
                os.path.basename(source)
            )

            shutil.copy2(
                source,
                destination
            )

        start_frame = self.get_frame_number(start)
        end_frame = self.get_frame_number(end)

        print(
            f"Rally {rally_number:04d} 저장"
        )

        print(
            f"Frame: {start_frame} ~ {end_frame}"
        )

        print(
            f"저장 위치: {rally_dir}"
        )

        self.start_index = None
        self.end_index = None

    def run(self):

        if not self.frame_paths:

            print(
                f"프레임이 없습니다: {self.frame_dir}"
            )

            return

        window_name = "Rally Splitter"

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

            # 이전 프레임
            if key == ord("a"):

                if self.current_index > 0:

                    self.current_index -= 1

            # 다음 프레임
            elif key == ord("d"):

                if (
                    self.current_index
                    < len(self.frame_paths) - 1
                ):

                    self.current_index += 1

            # 이동 폭 감소
            elif key == ord("["):

                self.step = max(
                    1,
                    self.step - 1
                )

                print(f"이동 폭: {self.step}")

            # 이동 폭 증가
            elif key == ord("]"):

                self.step += 1

                print(f"이동 폭: {self.step}")

            # 시작 지정
            elif key == ord("s"):

                self.start_index = (
                    self.current_index
                )

                print(
                    "Start:",
                    self.get_frame_number(
                        self.current_index
                    )
                )

            # 끝 지정
            elif key == ord("e"):

                self.end_index = (
                    self.current_index
                )

                print(
                    "End:",
                    self.get_frame_number(
                        self.current_index
                    )
                )

            # 랠리 저장
            elif key == ord("n"):

                self.save_rally()

            # 종료
            elif key == ord("q"):

                break

        cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(
        description="프레임을 랠리 단위로 분리합니다."
    )

    parser.add_argument(
        "--frames",
        required=True,
        help="1차 추출 프레임 폴더"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="랠리 저장 폴더"
    )

    args = parser.parse_args()

    splitter = RallySplitter(
        frame_dir=args.frames,
        output_dir=args.output
    )

    splitter.run()


if __name__ == "__main__":
    main()
