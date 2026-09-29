"""09단계: 자체 모델로 이미지 한 장을 추론하고 원본 좌표 bbox와 시각화를 저장한다."""

import argparse
from pathlib import Path

import cv2
import torch

from common import write_json_atomic
from detection.boxes import decode_predictions, restore_boxes
from detection.data import preprocess_image
from detection.runtime import choose_device, load_detector, new_output_directory


def main():
    """체크포인트에 저장된 전처리 크기를 사용해 학습/추론 입력 조건을 일치시킨다."""
    parser = argparse.ArgumentParser(
        description="직접 학습한 모델로 이미지의 선수·공을 검출합니다."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="새 추론 출력 폴더")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("confidence/nms-iou는 0~1이어야 합니다.")
    image = cv2.imread(args.image)
    if image is None:
        raise OSError(f"이미지를 읽지 못했습니다: {args.image}")
    device = choose_device(args.device)
    model, checkpoint = load_detector(args.weights, device)
    tensor, transform = preprocess_image(image, checkpoint["image_size"])
    with torch.inference_mode():
        prediction = decode_predictions(
            model(tensor.unsqueeze(0).to(device)), args.confidence, args.nms_iou
        )[0]
    boxes = restore_boxes(prediction["boxes"], transform)
    objects = []
    for box, score, label in zip(boxes, prediction["scores"], prediction["labels"]):
        x1, y1, x2, y2 = box.tolist()
        # 패딩 영역에만 있던 박스는 원본 이미지의 객체가 아니므로 제외한다.
        if x2 <= x1 or y2 <= y1:
            continue
        name = checkpoint["classes"][int(label)]
        objects.append(
            {
                "class": name,
                "score": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        cv2.rectangle(
            image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2
        )
        cv2.putText(
            image,
            f"{name} {float(score):.2f}",
            (round(x1), max(15, round(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    output = new_output_directory(args.output)
    if not cv2.imwrite(str(output / "prediction.jpg"), image):
        raise OSError("검출 결과 이미지 저장에 실패했습니다.")
    write_json_atomic(
        output / "prediction.json",
        {
            "image": str(Path(args.image).resolve()),
            "weights": str(Path(args.weights).resolve()),
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "objects": objects,
        },
    )
    print(f"검출 객체: {len(objects)}개\n저장 위치: {output}")


if __name__ == "__main__":
    main()
