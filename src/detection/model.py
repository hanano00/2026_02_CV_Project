"""CNN과 격자 출력부를 직접 정의한다. 외부 모델·사전학습 가중치를 사용하지 않는다."""

import math

import torch
from torch import nn


def conv_block(in_channels, out_channels, stride=1, dilation=1):
    """작은 배치에서도 동작하도록 BatchNorm 대신 GroupNorm을 사용한다."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        ),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class TennisGridDetector(nn.Module):
    """클래스별로 [객체 점수, 중심 x/y 오프셋, log 너비/높이]를 출력한다.

    출력: [배치, 클래스 수, 5, 입력 높이/8, 입력 너비/8].
    클래스별 출력이라 선수와 공의 중심이 같은 격자에 있어도 표현 가능하다.
    같은 클래스의 중심 두 개가 같은 격자에 있으면 표현할 수 없는 첫 기준 모델이다.
    """

    stride = 8

    def __init__(self, num_classes):
        super().__init__()
        if num_classes < 1:
            raise ValueError("클래스는 하나 이상이어야 합니다.")
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            conv_block(3, 24, stride=2),
            conv_block(24, 48, stride=2),
            conv_block(48, 96, stride=2),
            conv_block(96, 128),
            conv_block(128, 128, dilation=2),
        )
        self.head = nn.Conv2d(128, num_classes * 5, 1)
        # 모든 가중치는 새로 초기화한다. 다운로드·전이학습 과정은 없다.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        with torch.no_grad():
            bias = self.head.bias.view(num_classes, 5)
            # 대부분의 격자가 배경이므로 초기 객체 확률을 약 1%로 둔다.
            bias[:, 0] = math.log(0.01 / 0.99)
            bias[:, 3:5] = math.log(0.1)

    def forward(self, images):
        """픽셀값이 [0,1]인 RGB 텐서를 받아 활성화 전 예측값을 반환한다."""
        raw = self.head(self.backbone(images))
        batch, _, height, width = raw.shape
        return raw.reshape(batch, self.num_classes, 5, height, width)
