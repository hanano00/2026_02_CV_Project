"""장치 선택, 학습/검증 루프, 프로젝트 자체 체크포인트 입출력."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .boxes import decode_predictions
from .metrics import detection_metrics
from .model import TennisGridDetector


def choose_device(name="auto"):
    """명시한 장치를 우선하며 auto는 CUDA → MPS → CPU 순서로 선택한다."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA를 사용할 수 없습니다. --device cpu 또는 mps를 지정하세요."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS를 사용할 수 없습니다. --device cpu를 지정하세요.")
    return device


def seed_everything(seed):
    """초기화·셔플 시드를 고정한다. 장치 간 완전히 같은 수치 결과를 보장하지는 않는다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_output_directory(path):
    """이전 실험을 덮어쓰지 않도록 새로운 출력 폴더만 만든다."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_checkpoint(path, payload):
    """완성된 임시 체크포인트만 교체해 저장 중 실패로 기존 가중치가 깨지지 않게 한다."""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".checkpoint-", delete=False
        ) as file:
            temporary_path = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_detector(path, device):
    """이 프로젝트가 저장한 구조·클래스 정보와 state_dict를 복원한다."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        checkpoint.get("architecture") != "TennisGridDetector"
        or checkpoint.get("format_version") != 1
    ):
        raise ValueError("이 프로젝트의 TennisGridDetector 체크포인트가 아닙니다.")
    classes = checkpoint["classes"]
    model = TennisGridDetector(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def train_epoch(model, loader, criterion, optimizer, device):
    """순전파 → 손실 → 역전파 → 가중치 갱신을 한 epoch 수행한다."""
    model.train()
    totals = {}
    count = 0
    progress = tqdm(loader, desc="Train")
    for images, targets in progress:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        raw = model(images)
        loss, values = criterion(raw, targets)
        if not torch.isfinite(loss):
            raise ValueError("손실이 유한하지 않습니다. 라벨과 학습률을 확인하세요.")
        loss.backward()
        # 비정상적으로 큰 한 배치의 기울기가 학습 전체를 망가뜨리는 것을 줄인다.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10, error_if_nonfinite=True
        )
        optimizer.step()
        count += len(images)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * len(images)
        progress.set_postfix(loss=f"{values['loss']:.4f}")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    classes,
    score_floor=0.01,
    confidence=0.25,
    nms_threshold=0.5,
):
    """검증에서는 가중치를 갱신하지 않고 손실과 클래스별 검출 성능을 기록한다."""
    model.eval()
    records = []
    loss_sum = 0.0
    count = 0
    for images, targets in tqdm(loader, desc="Evaluate"):
        raw = model(images.to(device))
        loss, _ = criterion(raw, targets)
        if not torch.isfinite(loss):
            raise ValueError("평가 손실이 유한하지 않습니다.")
        predictions = decode_predictions(raw, score_floor, nms_threshold)
        records.extend(zip(predictions, targets))
        loss_sum += float(loss) * len(images)
        count += len(images)
    metrics = detection_metrics(records, classes, confidence)
    metrics.update(
        loss=loss_sum / count,
        score_floor=score_floor,
        nms_threshold=nms_threshold,
        max_detections_per_image=100,
        candidates_per_class_before_nms=300,
    )
    return metrics
