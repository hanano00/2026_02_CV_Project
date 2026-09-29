"""클래스별 Precision/Recall과 IoU 0.5 기준 AP를 직접 계산한다.

AP50은 예측을 점수순으로 정렬하고, 한 정답에 예측 하나만 연결하여 계산한다.
COCO의 전체 mAP50-95 평가를 대체하는 구현은 아니다.
"""

import torch

from .boxes import box_iou


def detection_metrics(records, classes, confidence=0.25):
    """records는 이미지별 (예측 dict, 정답 dict) 목록이다."""
    per_class = {}
    for label, name in enumerate(classes):
        ground_truth = {}
        predictions = []
        for image_index, (prediction, target) in enumerate(records):
            ground_truth[image_index] = target["boxes"][target["labels"] == label].cpu()
            selected = prediction["labels"] == label
            for box, score in zip(
                prediction["boxes"][selected], prediction["scores"][selected]
            ):
                predictions.append((float(score), image_index, box))
        predictions.sort(key=lambda item: item[0], reverse=True)
        matched = {
            key: torch.zeros(len(value), dtype=torch.bool)
            for key, value in ground_truth.items()
        }
        total_gt = sum(len(value) for value in ground_truth.values())
        true_positive = []
        scores = []
        for score, image_index, box in predictions:
            available = torch.where(~matched[image_index])[0]
            correct = False
            if available.numel():
                overlaps = box_iou(
                    box.reshape(1, 4), ground_truth[image_index][available]
                )[0]
                best = int(overlaps.argmax())
                if float(overlaps[best]) >= 0.5:
                    matched[image_index][available[best]] = True
                    correct = True
            true_positive.append(float(correct))
            scores.append(score)
        tp = torch.tensor(true_positive)
        fp = 1 - tp
        cumulative_tp = tp.cumsum(0)
        cumulative_fp = fp.cumsum(0)
        ap = None
        if total_gt:
            recall_curve = cumulative_tp / total_gt
            precision_curve = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(
                1
            )
            # Precision envelope의 면적: 점수 순위에 따른 PR 곡선을 적분한다.
            recall_curve = torch.cat(
                (torch.tensor([0.0]), recall_curve, torch.tensor([1.0]))
            )
            precision_curve = torch.cat(
                (torch.tensor([0.0]), precision_curve, torch.tensor([0.0]))
            )
            precision_curve = precision_curve.flip(0).cummax(0).values.flip(0)
            ap = float(
                ((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:]).sum()
            )
        accepted = torch.tensor(scores) >= confidence
        tp_at_threshold = int(tp[accepted].sum())
        fp_at_threshold = int(fp[accepted].sum())
        per_class[name] = {
            "ground_truth": total_gt,
            "AP50": ap,
            "precision": tp_at_threshold / max(tp_at_threshold + fp_at_threshold, 1),
            "recall": tp_at_threshold / total_gt if total_gt else None,
            "true_positive": tp_at_threshold,
            "false_positive": fp_at_threshold,
            "false_negative": total_gt - tp_at_threshold,
        }
    available_ap = [
        row["AP50"] for row in per_class.values() if row["AP50"] is not None
    ]
    return {
        "mAP50": sum(available_ap) / len(available_ap) if available_ap else None,
        "iou_threshold": 0.5,
        "precision_recall_confidence": confidence,
        "classes_with_ground_truth": len(available_ap),
        "per_class": per_class,
    }
