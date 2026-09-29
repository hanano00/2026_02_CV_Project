"""08단계: 학습된 자체 모델을 val/test split에서 평가한다.

개발 중에는 val을 사용하고, 최종 성능 보고에는 따로 보관한 test를 사용한다.
소프트웨어 테스트 실행 도구가 아니라 검출 모델의 성능 측정 도구다.
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from common import write_json_atomic
from detection.data import DetectionDataset, collate_detection
from detection.loss import DetectionLoss
from detection.runtime import (
    choose_device,
    evaluate,
    load_detector,
    new_output_directory,
)


def main():
    """체크포인트의 클래스 순서·입력 크기를 유지한 채 평가 결과를 JSON으로 저장한다."""
    parser = argparse.ArgumentParser(
        description="직접 학습한 검출 모델의 클래스별 AP50을 평가합니다."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True, help="새 평가 출력 폴더")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--score-floor",
        type=float,
        default=0.01,
        help="AP 계산에 포함할 최소 예측 점수",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Precision/Recall 보고 임계값"
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size는 양수, workers는 0 이상이어야 합니다.")
    if not 0 <= args.score_floor <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("0 ≤ score-floor ≤ confidence ≤ 1, 0 ≤ nms-iou ≤ 1이어야 합니다.")
    device = choose_device(args.device)
    model, checkpoint = load_detector(args.weights, device)
    dataset = DetectionDataset(args.config, args.split, checkpoint["image_size"])
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("데이터와 체크포인트의 클래스 이름·순서가 다릅니다.")
    output = new_output_directory(args.output)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_detection,
        shuffle=False,
    )
    result = evaluate(
        model,
        loader,
        DetectionLoss(),
        device,
        dataset.classes,
        args.score_floor,
        args.confidence,
        args.nms_iou,
    )
    result.update(
        weights=str(Path(args.weights).resolve()),
        config=str(Path(args.config).resolve()),
        split=args.split,
        images=len(dataset),
        image_size=checkpoint["image_size"],
    )
    write_json_atomic(output / "metrics.json", result)
    for name, metrics in result["per_class"].items():
        print(f"{name}: {metrics}")
    print(f"mAP50: {result['mAP50']}\n저장 위치: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
