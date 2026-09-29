# PyTorch로 직접 만드는 테니스 검출 모델

## 목적과 파일 구조

사전학습 가중치 없이 CNN의 특징 추출부와 격자 출력부를 직접 정의하고 학습하는 첫 기준 모델이다.
YOLO·Ultralytics·torchvision 모델은 호출하지 않으며 가중치를 다운로드하지 않는다.
학습 실행 전에는 무작위 초기화된 모델 구조만 있고, 학습이 완료되어야 사용할 가중치가 생긴다.

| 파일 | 역할 |
|---|---|
| `model.py` | CNN 특징 추출부, 클래스별 객체 점수·bbox 출력부 |
| `data.py` | 데이터 설정과 기존 bbox JSON 읽기, letterbox, 좌우 반전, 배치 구성 |
| `loss.py` | 정답을 중심 격자에 배정하고 객체 점수·중심·크기 손실 계산 |
| `boxes.py` | 격자 출력을 박스로 변환, 클래스별 NMS, 원본 좌표 복원 |
| `metrics.py` | 클래스별 Precision/Recall/AP50 계산 |
| `runtime.py` | 장치 선택, epoch 실행, 평가, 체크포인트 저장·복원 |
| `../07_train_detector.py` | 학습 실행 |
| `../08_evaluate_detector.py` | val/test 데이터 성능 측정 |
| `../09_predict_detector.py` | 이미지 한 장 추론 |

## 모델 구조와 정답 배정

기본 입력은 `512×512` RGB 이미지이며, 픽셀값을 `[0,1]`로 정규화한다.
종횡비를 유지한 채 크기를 조정하고 남는 영역을 회색으로 채운다(letterbox).
학습 시 기본 50% 확률로 좌우 반전하며 박스 좌표도 함께 바꾼다. 검증·추론에서는 반전하지 않는다.

```text
RGB 입력
→ Conv(3→24, stride 2) + GroupNorm + SiLU
→ Conv(24→48, stride 2) + GroupNorm + SiLU
→ Conv(48→96, stride 2) + GroupNorm + SiLU
→ Conv(96→128, stride 1) + GroupNorm + SiLU
→ Conv(128→128, dilation 2) + GroupNorm + SiLU
→ 1×1 Conv 출력부
→ [배치, 클래스 수, 5, H/8, W/8]
```

각 클래스의 격자마다 객체 점수 1개와 박스 값 4개를 예측한다.

- 객체 점수: 그 클래스의 객체 중심이 해당 격자에 있는지 나타내는 logit.
- 중심 x/y: 격자 내부의 위치를 sigmoid로 `[0,1]` 범위에 표현.
- 너비/높이: 입력 전체 크기에 대한 비율의 로그값.
- 클래스는 출력 채널로 구분한다. 별도 softmax 분류 손실 대신 클래스별 객체 점수 손실을 사용한다.

정답 박스는 **중심이 속한 격자**에 배정한다. 선수와 공은 같은 격자에 있어도 별도 채널로 표현한다.
같은 클래스의 두 중심이 같은 격자에 있으면 하나를 버리지 않고 오류를 알린다.
그 경우 입력 해상도를 높이거나 여러 박스를 출력하는 구조로 확장해야 한다.

손실은 `focal 객체 점수 손실 + 5 × 중심 Smooth L1 + 크기 Smooth L1`이다.
박스 손실은 정답 객체가 있는 격자에만 적용한다. 모든 파라미터를 AdamW로 갱신하고,
epoch별 cosine 학습률 조절과 기울기 크기 제한을 사용한다.

## 데이터 설정: 기존 JSON 활용

[설정 예시](../../configs/detector.example.json)를 복사해 실제 경로로 수정한다.
상대 경로는 **설정 JSON이 있는 폴더 기준**이다. 클래스 순서는 가중치에 저장된다.

```json
{
  "classes": ["player", "ball"],
  "train": [
    {
      "images": "../datasets/detection/train/images",
      "annotations": {
        "player": "../datasets/detection/train/player.json",
        "ball": "../datasets/detection/train/ball.json"
      }
    }
  ],
  "val": [
    {
      "images": "../datasets/detection/val/images",
      "annotations": "../datasets/detection/val/combined.json"
    }
  ]
}
```

split마다 여러 경기·랠리 소스를 추가할 수 있다. `annotations`는 클래스별 JSON 경로 객체,
또는 모든 클래스가 포함된 통합 JSON 하나의 경로를 받는다.
선수·공을 따로 학습하려면 `classes`에 해당 클래스 하나만 지정하고 별도 실험 폴더를 사용한다.

주석 파일은 기존 `06_annotator.py`의 프레임 번호별 형식을 사용한다.

```json
{
  "120": {
    "objects": [
      {"class": "player", "x1": 30, "y1": 40, "x2": 80, "y2": 150}
    ]
  },
  "150": {"objects": []}
}
```

- 파일명은 `frame_000120.jpg` 형식이며 좌표는 원본 이미지 픽셀 기준이다.
- 선택한 모든 이미지에 검수된 라벨이 있어야 한다. 클래스별 파일을 사용하면 모든 클래스 파일에 해당 프레임 키가 필요하다.
- 빈 `objects`는 실제로 검수한 결과 객체가 없다는 의미다. 빠진 라벨을 자동으로 빈 라벨로 만들지 않는다.
- 학습 split에는 설정한 각 클래스의 정답 객체가 하나 이상 있어야 한다.
- 이미지가 같은 경로로 train/val에 중복되면 중단한다. 다른 경로에 복사된 이미지나 같은 랠리의 유사 프레임까지 자동 검출하지는 않으므로 경기·랠리 단위로 분리한다.
- 설정의 `test`도 동일한 소스 목록 형식이며, 최종 평가할 때만 필요하다.

## 학습·평가·추론 명령

프로젝트 루트에서 실행한다. 모든 출력 경로는 **아직 없는 새 폴더**를 지정한다.
예시는 경로를 실제 데이터에 맞게 수정한 뒤 사용한다.

```bash
uv run python src/07_train_detector.py --config configs/detector.example.json --output runs/scratch01 --epochs 50 --batch-size 4 --image-size 512 --device auto
uv run python src/08_evaluate_detector.py --weights runs/scratch01/best.pt --config configs/detector.example.json --split val --output runs/eval01
uv run python src/09_predict_detector.py --weights runs/scratch01/best.pt --image datasets/detection/val/images/frame_000120.jpg --output runs/predict01
```

장치는 `auto`에서 CUDA → MPS → CPU 순으로 선택한다. `--device cpu`, `mps`, `cuda:0`으로 지정할 수도 있다.
입력 크기는 32 이상의 8의 배수다. 메모리가 부족하면 배치 크기를 줄인다.
`--patience` 기본값 0은 조기 종료 비활성이며 양수로 지정하면 미개선 epoch 수를 기준으로 종료한다.
소프트웨어 테스트를 실행하는 명령은 포함하지 않는다. 08은 학습 모델의 성능 평가 단계다.

| 결과 | 내용 |
|---|---|
| `run_config.json` | 학습 옵션, 데이터 설정 경로, 장치, 클래스별 정답 수 |
| `history.json` | epoch별 학습 손실, 검증 손실, 클래스별 검출 지표 |
| `last.pt` | 마지막 완료 epoch의 모델·optimizer·scheduler 상태 |
| `best.pt` | 검증 AP50이 가장 좋았던 모델. AP50이 같으면 검증 손실로 비교 |
| 평가 폴더의 `metrics.json` | 선택한 val/test split의 성능과 후처리 설정 |
| 추론 폴더의 `prediction.json` / `prediction.jpg` | 원본 좌표 bbox·점수와 이미지 시각화 |

체크포인트에는 구조 이름, 클래스 순서, 입력 크기, 가중치를 함께 저장한다.
현재 CLI는 새 학습 실행과 평가·추론을 지원한다. optimizer 상태는 보관하지만 중단 학습 재개 CLI는 아직 구현하지 않았다.

## 지표와 첫 모델의 한계

- AP50은 IoU 0.5에서 점수순 예측과 정답을 1:1로 연결한 PR 곡선 면적이다. COCO mAP50-95는 아니다.
- Precision/Recall 보고 점수 임계값은 기본 0.25, AP에 포함할 예측의 최저 점수는 0.01이다.
- 같은 클래스에 NMS IoU 0.5를 적용한다. NMS 전 클래스별 300개, NMS 후 이미지 전체 100개를 상한으로 둔다.
- 평가 JSON에 이 설정을 남기므로 같은 조건의 실험끼리 비교한다. score floor 때문에 낮은 점수 예측은 AP 계산에서 제외된다.
- 정답이 없는 클래스의 AP와 Recall은 `null`이며 mAP50 평균에서 제외한다. 클래스별 정답 수와 개별 지표를 함께 본다.
- 단일 스케일 CNN과 stride 8 출력은 공처럼 작은 객체나 가림 장면에서 부족할 수 있다. 실제 성능 확인 후 입력 크기·특징 해상도·다중 스케일 구조를 검토한다.
- 박스 출력은 검출 결과다. 선수 ID 추적·코트 좌표 변환·상태전이 모델링은 다음 단계다.

## 공식 문서

- [PyTorch 신경망 모듈](https://docs.pytorch.org/docs/stable/nn.html)
- [Dataset과 DataLoader](https://docs.pytorch.org/docs/stable/data.html)
- [모델 저장과 불러오기](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
