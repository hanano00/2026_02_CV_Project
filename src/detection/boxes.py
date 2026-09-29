"""격자 예측을 bbox로 해석하고 중복 검출을 제거한다. torchvision 모델은 사용하지 않는다."""

import torch


def box_iou(first, second):
    """[N,4], [M,4]의 xyxy 박스 간 IoU를 계산한다."""
    intersection_min = torch.maximum(first[:, None, :2], second[None, :, :2])
    intersection_max = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(-1)
    area_first = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(-1)
    area_second = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_first[:, None] + area_second - intersection).clamp_min(
        1e-8
    )


def nms(boxes, scores, threshold, limit):
    """높은 점수부터 선택하고 겹치는 같은 클래스의 박스를 제거한다."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() and len(keep) < limit:
        current = order[0]
        keep.append(int(current))
        remaining = order[1:]
        if not remaining.numel():
            break
        overlaps = box_iou(boxes[current].reshape(1, 4), boxes[remaining])[0]
        order = remaining[overlaps <= threshold]
    return torch.tensor(keep, dtype=torch.long)


def decode_predictions(
    raw, score_threshold=0.01, nms_threshold=0.5, max_detections=100
):
    """활성화·좌표 변환·클래스별 NMS를 적용해 정규화 xyxy 결과를 반환한다.

    CPU에서 후처리하므로 CUDA/MPS 전용 NMS 연산에 의존하지 않는다.
    속도와 결과 크기를 제한하기 위해 NMS 전 클래스별 상위 300개만 고려한다.
    """
    if not 0 <= score_threshold <= 1 or not 0 <= nms_threshold <= 1:
        raise ValueError("점수와 NMS 임계값은 0~1이어야 합니다.")
    raw = raw.detach().float().cpu()
    _, classes, _, height, width = raw.shape
    results = []
    for image in raw:
        image_boxes, image_scores, image_labels = [], [], []
        for label in range(classes):
            scores = image[label, 0].sigmoid().flatten()
            indices = torch.where(scores >= score_threshold)[0]
            if indices.numel() == 0:
                continue
            indices = indices[scores[indices].argsort(descending=True)[:300]]
            values = image[label, 1:].reshape(4, -1)[:, indices].T
            cell_x = (indices % width).float()
            cell_y = (indices // width).float()
            center = torch.stack(
                (
                    (cell_x + values[:, 0].sigmoid()) / width,
                    (cell_y + values[:, 1].sigmoid()) / height,
                ),
                dim=-1,
            )
            # 네트워크 출력의 매우 큰 값으로 인한 exp overflow를 막는다.
            size = values[:, 2:].clamp(-16, 0).exp()
            boxes = torch.cat((center - size / 2, center + size / 2), dim=-1).clamp(
                0, 1
            )
            chosen = nms(boxes, scores[indices], nms_threshold, max_detections)
            image_boxes.append(boxes[chosen])
            image_scores.append(scores[indices][chosen])
            image_labels.append(torch.full((len(chosen),), label, dtype=torch.long))
        if image_boxes:
            boxes = torch.cat(image_boxes)
            scores = torch.cat(image_scores)
            labels = torch.cat(image_labels)
            order = scores.argsort(descending=True)[:max_detections]
            results.append(
                {
                    "boxes": boxes[order],
                    "scores": scores[order],
                    "labels": labels[order],
                }
            )
        else:
            results.append(
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty(0),
                    "labels": torch.empty(0, dtype=torch.long),
                }
            )
    return results


def restore_boxes(boxes, transform):
    """letterbox 기준 정규화 좌표를 원본 이미지의 픽셀 좌표로 되돌린다."""
    boxes = boxes.clone() * transform["input_size"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform["pad_left"]) / transform["scale_x"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform["pad_top"]) / transform["scale_y"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, transform["original_width"])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, transform["original_height"])
    return boxes
