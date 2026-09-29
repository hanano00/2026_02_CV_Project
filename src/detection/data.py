"""기존 프레임별 bbox JSON을 읽어 PyTorch Dataset으로 제공한다."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import frame_number, list_frames


def read_config(path):
    """config 내부의 데이터 경로는 config 파일 위치를 기준으로 해석한다."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise TypeError("데이터 설정은 JSON 객체여야 합니다.")
    classes = config.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(name, str) or not name.strip() for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("classes에는 중복 없는 클래스 이름 목록이 필요합니다.")
    return config, path.parent


def preprocess_image(image, image_size):
    """종횡비를 유지해 리사이즈하고 남는 공간을 채운다(letterbox).

    정수 반올림 후 실제 x/y 배율을 기록해 추론 박스를 원본 좌표로 복원한다.
    """
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, min(image_size, round(width * scale)))
    resized_height = max(1, min(image_size, round(height * scale)))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    transform = {
        "original_width": width,
        "original_height": height,
        "scale_x": resized_width / width,
        "scale_y": resized_height / height,
        "pad_left": left,
        "pad_top": top,
        "input_size": image_size,
    }
    return tensor, transform


class DetectionDataset(Dataset):
    """한 split 안의 여러 경기·랠리 폴더와 주석 파일을 하나의 데이터셋으로 묶는다."""

    def __init__(self, config_path, split, image_size=512, flip_probability=0.0):
        if image_size < 32 or image_size % 8:
            raise ValueError("image_size는 32 이상의 8의 배수여야 합니다.")
        if not 0 <= flip_probability <= 1:
            raise ValueError("좌우 반전 확률은 0~1이어야 합니다.")
        config, root = read_config(config_path)
        self.classes = config["classes"]
        self.image_size = image_size
        self.flip_probability = flip_probability
        self.samples = []
        sources = config.get(split)
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                f"{split}에는 images/annotations를 가진 목록이 필요합니다."
            )
        seen = set()
        for source in sources:
            if not isinstance(source, dict):
                raise TypeError("각 데이터 소스는 JSON 객체여야 합니다.")
            images = (root / source["images"]).resolve()
            annotation_spec = source["annotations"]
            # 통합 JSON 하나 또는 기존 클래스별 JSON을 모두 지정할 수 있다.
            if isinstance(annotation_spec, str):
                annotation_files = [(None, annotation_spec)]
            elif isinstance(annotation_spec, dict):
                if set(annotation_spec) != set(self.classes):
                    raise ValueError(
                        "분리 JSON은 설정의 모든 클래스 파일을 지정해야 합니다."
                    )
                annotation_files = list(annotation_spec.items())
            else:
                raise TypeError(
                    "annotations는 JSON 경로 또는 클래스별 경로 객체여야 합니다."
                )
            annotations = []
            for expected_class, filename in annotation_files:
                with (root / filename).open(encoding="utf-8") as file:
                    mapping = json.load(file)
                if not isinstance(mapping, dict):
                    raise TypeError(
                        "라벨 JSON은 프레임 번호를 키로 갖는 객체여야 합니다."
                    )
                annotations.append((expected_class, mapping))
            for image_path in list_frames(images):
                image_path = image_path.resolve()
                if image_path in seen:
                    raise ValueError(f"split 안에서 이미지가 중복됩니다: {image_path}")
                seen.add(image_path)
                number = str(frame_number(image_path))
                objects = []
                for expected_class, mapping in annotations:
                    entry = mapping.get(number)
                    if entry is None:
                        raise ValueError(
                            f"{image_path}: 프레임 {number}의 라벨이 없습니다. "
                            "미검수 프레임을 객체 없음으로 처리하지 않습니다."
                        )
                    if not isinstance(entry, dict) or not isinstance(
                        entry.get("objects"), list
                    ):
                        raise TypeError(
                            f"{image_path}: 프레임 {number}의 검수된 라벨이 없습니다. "
                            '객체가 없다면 명시적으로 {"objects": []}를 저장하세요.'
                        )
                    for obj in entry["objects"]:
                        if not isinstance(obj, dict):
                            raise TypeError(
                                f"객체 라벨은 JSON 객체여야 합니다: {image_path}"
                            )
                        name = obj.get("class")
                        if name not in self.classes or (
                            expected_class is not None and name != expected_class
                        ):
                            raise ValueError(
                                f"클래스가 설정과 다릅니다: {image_path}: {name}"
                            )
                        coordinates = [obj.get(key) for key in ("x1", "y1", "x2", "y2")]
                        if not all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                            for value in coordinates
                        ):
                            raise ValueError(
                                f"유한한 숫자 bbox가 필요합니다: {image_path}"
                            )
                        x1, y1, x2, y2 = coordinates
                        if not (0 <= x1 < x2 and 0 <= y1 < y2):
                            raise ValueError(
                                f"bbox 범위가 잘못되었습니다: {image_path}"
                            )
                        objects.append((self.classes.index(name), coordinates))
                self.samples.append((image_path, objects))
        if not self.samples:
            raise ValueError(f"{split}에 학습/평가할 이미지가 없습니다.")

    @property
    def object_counts(self):
        """클래스가 등록만 되고 정답은 없는 데이터셋을 학습 전에 확인한다."""
        counts = {name: 0 for name in self.classes}
        for _, objects in self.samples:
            for label, _ in objects:
                counts[self.classes[label]] += 1
        return counts

    @property
    def image_paths(self):
        """학습·검증 분할의 동일 경로 중복을 확인할 때 사용한다."""
        return {path for path, _ in self.samples}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, objects = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"이미지를 읽지 못했습니다: {path}")
        tensor, transform = preprocess_image(image, self.image_size)
        boxes, labels = [], []
        for label, (x1, y1, x2, y2) in objects:
            if x2 > image.shape[1] or y2 > image.shape[0]:
                raise ValueError(f"bbox가 이미지 경계를 벗어났습니다: {path}")
            boxes.append(
                [
                    (x1 * transform["scale_x"] + transform["pad_left"])
                    / self.image_size,
                    (y1 * transform["scale_y"] + transform["pad_top"])
                    / self.image_size,
                    (x2 * transform["scale_x"] + transform["pad_left"])
                    / self.image_size,
                    (y2 * transform["scale_y"] + transform["pad_top"])
                    / self.image_size,
                ]
            )
            labels.append(label)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if torch.rand(()).item() < self.flip_probability:
            tensor = tensor.flip(-1)
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        return tensor, {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_path": str(path),
        }


def collate_detection(batch):
    """이미지는 묶고, 이미지마다 개수가 다른 정답 박스는 리스트로 유지한다."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
