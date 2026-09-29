"""07단계: 직접 정의한 CNN 검출기를 무작위 초기화부터 학습한다.

사전학습 가중치나 YOLO/torchvision 모델을 불러오지 않는다.
예: python src/07_train_detector.py --config configs/detector.example.json --output runs/baseline01
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import positive_finite, write_json_atomic
from detection.data import DetectionDataset, collate_detection
from detection.loss import DetectionLoss
from detection.model import TennisGridDetector
from detection.runtime import (
    choose_device,
    evaluate,
    new_output_directory,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


def main():
    """데이터·설정을 준비하고 학습/검증을 반복하며 best/last 체크포인트를 남긴다."""
    parser = argparse.ArgumentParser(
        description="PyTorch CNN 객체검출기를 처음부터 학습합니다."
    )
    parser.add_argument(
        "--config", required=True, help="데이터 분할과 JSON 라벨 경로 설정"
    )
    parser.add_argument("--output", required=True, help="새 실험 출력 폴더")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512, help="32 이상, 8의 배수")
    parser.add_argument("--lr", type=positive_finite, default=1e-3)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps, cuda, cuda:0 등"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=0, help="검증 성능 미개선 종료 횟수, 0은 비활성"
    )
    parser.add_argument("--flip-probability", type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 0:
        parser.error(
            "epochs/batch-size는 양수, workers/patience는 0 이상이어야 합니다."
        )
    if not 0 <= args.seed < 2**32:
        parser.error("seed는 0 이상 2**32 미만이어야 합니다.")
    seed_everything(args.seed)
    device = choose_device(args.device)
    training = DetectionDataset(
        args.config, "train", args.image_size, args.flip_probability
    )
    validation = DetectionDataset(args.config, "val", args.image_size)
    missing_classes = [
        name for name, count in training.object_counts.items() if not count
    ]
    if missing_classes:
        raise ValueError(f"학습 정답이 없는 클래스입니다: {missing_classes}")
    if training.image_paths & validation.image_paths:
        raise ValueError(
            "train과 val에 같은 이미지 경로가 있습니다. 경기/랠리별로 분리하세요."
        )
    output = new_output_directory(args.output)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": collate_detection,
    }
    train_loader = DataLoader(
        training, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(validation, shuffle=False, **loader_options)
    model = TennisGridDetector(len(training.classes)).to(device)
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    settings = {
        **vars(args),
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "classes": training.classes,
        "architecture": "TennisGridDetector",
        "pretrained": False,
        "torch_version": str(torch.__version__),
        "train_objects": training.object_counts,
        "val_objects": validation.object_counts,
    }
    write_json_atomic(output / "run_config.json", settings)
    print(f"장치: {device} | train: {len(training)} | val: {len(validation)}")
    print(
        f"학습 파라미터 수: {sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    history = []
    best_ap, best_loss = -1.0, float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        training_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_metrics = evaluate(
            model, val_loader, criterion, device, training.classes
        )
        current_ap = validation_metrics["mAP50"]
        comparable_ap = -1.0 if current_ap is None else current_ap
        # AP50을 우선하고 AP가 같은 경우 검증 손실로 비교한다.
        improved = comparable_ap > best_ap or (
            comparable_ap == best_ap and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_ap, best_loss = comparable_ap, validation_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": training_metrics,
                "val": validation_metrics,
            }
        )
        scheduler.step()
        payload = {
            "format_version": 1,
            "architecture": "TennisGridDetector",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "classes": training.classes,
            "image_size": args.image_size,
            "settings": settings,
            "val_metrics": validation_metrics,
        }
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        write_json_atomic(output / "history.json", history)
        print(f"검증 loss={validation_metrics['loss']:.4f}, mAP50={current_ap}")
        if args.patience and stale_epochs >= args.patience:
            print("검증 성능이 개선되지 않아 조기 종료합니다.")
            break
    print(f"학습 결과: {output}\n최적 가중치: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
