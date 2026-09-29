"""정답의 중심 격자를 지정하고 객체 점수·중심·크기 손실을 계산한다."""

import torch
from torch import nn
from torch.nn import functional as F


def encode_targets(targets, num_classes, height, width):
    """클래스별 중심 위치 하나에 정답을 배정한다. 표현할 수 없는 충돌은 알린다."""
    presence = torch.zeros(len(targets), num_classes, height, width)
    geometry = torch.zeros(len(targets), num_classes, 4, height, width)
    for batch_index, target in enumerate(targets):
        for box, label in zip(target["boxes"], target["labels"]):
            center = (box[:2] + box[2:]) / 2
            size = box[2:] - box[:2]
            gx, gy = float(center[0]) * width, float(center[1]) * height
            x, y = min(int(gx), width - 1), min(int(gy), height - 1)
            label = int(label)
            if presence[batch_index, label, y, x]:
                raise ValueError(
                    f"같은 클래스의 중심이 같은 격자에 겹칩니다: {target['image_path']}. "
                    "입력 해상도를 높이거나 모델의 출력 구조를 확장해야 합니다."
                )
            presence[batch_index, label, y, x] = 1
            geometry[batch_index, label, :, y, x] = torch.tensor(
                [
                    gx - x,
                    gy - y,
                    float(size[0].clamp_min(1e-8).log()),
                    float(size[1].clamp_min(1e-8).log()),
                ]
            )
    return presence, geometry


class DetectionLoss(nn.Module):
    """배경 불균형에는 focal loss, 정답 격자의 bbox에는 Smooth L1을 적용한다.

    클래스마다 객체 점수를 따로 예측하므로 클래스 구분도 presence 손실이 담당한다.
    bbox 손실을 배경에 적용하지 않아 아무 객체도 없는 이미지도 학습할 수 있다.
    """

    def forward(self, raw, targets):
        _, classes, _, height, width = raw.shape
        presence, geometry = encode_targets(targets, classes, height, width)
        presence = presence.to(raw.device)
        geometry = geometry.to(raw.device)
        logits = raw[:, :, 0]
        probability = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
        pt = probability * presence + (1 - probability) * (1 - presence)
        alpha = 0.25 * presence + 0.75 * (1 - presence)
        normalizer = presence.sum().clamp_min(1)
        object_loss = (alpha * (1 - pt).square() * ce).sum() / normalizer
        positive = presence.bool()
        predicted = raw[:, :, 1:].permute(0, 1, 3, 4, 2)[positive]
        expected = geometry.permute(0, 1, 3, 4, 2)[positive]
        if predicted.numel():
            center_loss = (
                F.smooth_l1_loss(
                    predicted[:, :2].sigmoid(), expected[:, :2], reduction="sum"
                )
                / normalizer
            )
            size_loss = (
                F.smooth_l1_loss(predicted[:, 2:], expected[:, 2:], reduction="sum")
                / normalizer
            )
        else:
            center_loss = raw.sum() * 0
            size_loss = raw.sum() * 0
        total = object_loss + 5 * center_loss + size_loss
        return total, {
            "loss": float(total.detach()),
            "object_loss": float(object_loss.detach()),
            "center_loss": float(center_loss.detach()),
            "size_loss": float(size_loss.detach()),
        }
