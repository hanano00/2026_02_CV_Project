# 프로젝트 요약

## 1. 문서의 목적과 갱신 원칙

이 문서는 AI가 이해한 프로젝트 목표, 설계 방향, 현재 코드의 기능과 작업 규칙을 기록한다. 새로운 해석이나 수정안은 사용자에게 먼저 제시하고, 검수받은 뒤 반영한다.

아래 내용은 사용자의 직접 설명, 승인한 작업 순서, 코드 검토 후 반영을 요청한 수정 사항을 기준으로 정리했다. 구현된 기능과 아직 구상 단계인 내용을 구분한다.

## 2. 프로젝트 목표와 설계 방향

테니스 영상에서 선수·공을 검출하는 학습 모델을 만들고, 이를 기반으로 선수들의 랠리 패턴을 모델링하고 분석하는 프로젝트다. 객체검출은 이후 분석에 필요한 데이터를 얻는 앞단 단계다.

사용자는 사전학습 모델을 활용하는 방향이 아니라 **PyTorch로 모델 구조를 직접 정의하고 무작위 초기화부터 학습하는 방향**을 명시했다. 첫 구현은 CNN 기반 단일 스케일 격자형 검출기다.

사용자가 구상한 전체 흐름은 다음과 같다.

> 테니스 영상 → 객체검출 → 코트 구역·객체 이동·상대 관계 표현 → 상태전이 학습 → 랠리 패턴 분석

- 코트 위치를 구역화한다.
- bbox의 위치와 시간에 따른 이동을 활용한다.
- 객체 간 상대적인 위치·관계를 함께 활용한다.
- 이 정보를 바탕으로 상태전이를 학습하고 랠리 패턴을 찾는다.

코트 구역의 수와 경계, 좌표 변환, bbox 대표 위치, 관계 특징, 상태 정의, 학습 알고리즘과 평가 지표는 아직 확정됐다고 확인되지 않았다.

## 3. 현재 구현 범위와 실행 순서

현재 구현은 **영상 수집·프레임 준비·수동 라벨링과 PyTorch 검출 모델의 학습·평가·이미지 추론 코드까지**다. 데이터셋 구성이 끝났다는 전제로 코드를 작성했으며 실제 학습은 아직 실행하지 않았다. 객체 추적, 코트 구역화, 상태전이 학습, 랠리 패턴 분석은 아직 구현되지 않았다.

사용자가 확정한 데이터 준비 순서는 다음과 같다.

> 영상 다운로드 → 영상 정보 확인 → 프레임 추출 → 랠리 분할 → 프레임 선택 → bbox 라벨링

| 단계 | 파일 | 주요 기능과 출력 |
|---|---|---|
| 01 | [01_video_downloader.py](src/01_video_downloader.py) | 지정 FPS·최대 1080p 조건의 YouTube 영상 소스를 선택해 다운로드한다. FPS 변환 기능은 아니다. |
| 02 | [02_video_info.py](src/02_video_info.py) | 영상 해상도, FPS, 전체 프레임 수, 재생시간을 출력한다. |
| 03 | [03_frame_extractor.py](src/03_frame_extractor.py) | 지정 FPS로 JPG를 추출하고 원본 영상 정보를 `frames_metadata.json`에 저장한다. |
| 04 | [04_rally_splitter.py](src/04_rally_splitter.py) | 수동 지정한 시작·끝을 포함하는 입력 프레임을 랠리별 폴더에 복사하고 `rally_metadata.json`을 저장한다. |
| 05 | [05_frame_selector.py](src/05_frame_selector.py) | 개별 랠리에서 사용할 프레임을 선택해 별도 폴더에 복사하고 `selection_metadata.json`을 저장한다. |
| 06 | [06_annotator.py](src/06_annotator.py) | 선수 또는 공의 bbox를 마우스로 지정하고 프레임별 JSON으로 저장한다. |
| 07 | [07_train_detector.py](src/07_train_detector.py) | 직접 정의한 CNN을 처음부터 학습하고 best/last 가중치와 epoch별 기록을 저장한다. |
| 08 | [08_evaluate_detector.py](src/08_evaluate_detector.py) | val/test split에서 클래스별 Precision/Recall/AP50을 계산한다. |
| 09 | [09_predict_detector.py](src/09_predict_detector.py) | 이미지 한 장에서 객체를 검출하고 원본 좌표 bbox JSON과 시각화를 저장한다. |

각 파일은 독립적인 CLI다. 앞 단계의 출력 경로를 다음 단계의 입력으로 전달하며, 번호가 다음 스크립트를 자동 실행하지는 않는다. 05단계에는 랠리들을 담은 상위 폴더가 아니라 개별 랠리 폴더를 전달한다.

실행 명령과 경로 예시는 [src/README.md](src/README.md)에 정리되어 있다.

### 실행 순서에 따른 데이터셋 구조

아래는 앞으로 구성할 경로 예시이며, 데이터셋이 이미 완성되었다는 의미는 아니다.
01~06은 `src/README.md`의 실행 예시와 같은 구조다. 경기마다 `match01`,
`match02`처럼 폴더를 구분하고, 랠리별 프레임과 라벨을 유지한다.

```text
datasets/
├── videos/                                # 01 다운로드, 02 정보 확인
│   ├── match01.mp4
│   └── match02.mp4
├── match01/
│   ├── frames/                            # 03 영상에서 추출한 프레임
│   │   ├── frame_000000.jpg
│   │   ├── ...
│   │   └── frames_metadata.json
│   ├── rallies/                           # 04 구간별 프레임 복사
│   │   ├── rally_0001/
│   │   │   ├── frame_000100.jpg
│   │   │   ├── ...
│   │   │   └── rally_metadata.json
│   │   └── rally_0002/
│   │       └── ...
│   ├── selected/                          # 05 검출 학습용 프레임 선택
│   │   ├── rally_0001/
│   │   │   ├── frame_000100.jpg
│   │   │   ├── ...
│   │   │   └── selection_metadata.json
│   │   └── rally_0002/
│   │       └── ...
│   └── annotations/                       # 06 선수·공을 각각 라벨링
│       ├── rally_0001_player.json
│       ├── rally_0001_ball.json
│       ├── rally_0002_player.json
│       └── rally_0002_ball.json
└── match02/                               # 다른 경기도 같은 구조
    └── ...

configs/
└── detector.json                          # 06 이후 직접 작성: 분할별 경로 목록
runs/
├── detector01/                            # 07 학습 결과
│   ├── run_config.json
│   ├── history.json
│   ├── best.pt
│   └── last.pt
├── detector01_val/                        # 08 개선 과정의 검증 결과
│   └── metrics.json
├── detector01_test/                       # 08 최종 평가 결과
│   └── metrics.json
└── detector01_predict/                    # 09 이미지 한 장의 추론 결과
    ├── prediction.json
    └── prediction.jpg
```

03의 파일명 숫자는 원본 영상의 프레임 번호이며, 04·05에서 복사할 때도 유지한다.
06의 JSON은 이미지마다 별도 파일을 만드는 대신, 한 랠리의 프레임 번호를 키로
여러 이미지의 라벨을 저장한다. 선택한 모든 이미지에 선수·공 각각의 검수된 기록이
필요하며, 객체가 없는 이미지도 확인 후 빈 `objects` 목록을 저장한다.

**06과 07 사이에는 데이터 분할과 설정 파일 작성이 필요하다.** 현재 로더는
분할마다 여러 이미지 폴더와 라벨 파일을 받을 수 있으므로, 위 폴더를 유지한 채
`configs/detector.json`에서 경로를 연결할 수 있다. 예를 들면 다음과 같다.

```json
{
  "classes": ["player", "ball"],
  "train": [{
    "images": "../datasets/match01/selected/rally_0001",
    "annotations": {
      "player": "../datasets/match01/annotations/rally_0001_player.json",
      "ball": "../datasets/match01/annotations/rally_0001_ball.json"
    }
  }],
  "val": [{
    "images": "../datasets/match02/selected/rally_0001",
    "annotations": {
      "player": "../datasets/match02/annotations/rally_0001_player.json",
      "ball": "../datasets/match02/annotations/rally_0001_ball.json"
    }
  }],
  "test": [{
    "images": "../datasets/match03/selected/rally_0001",
    "annotations": {
      "player": "../datasets/match03/annotations/rally_0001_player.json",
      "ball": "../datasets/match03/annotations/rally_0001_ball.json"
    }
  }]
}
```

위 JSON은 경로 연결만 보여주는 최소 예시다. 실제 구성에서는 각 목록에 여러
랠리를 추가하고, 예시에 쓰인 `match03`도 동일한 준비 과정을 거친다.
상대 경로는 설정 파일이 있는 `configs/` 기준이다. 07·08 실행 시
`--config configs/detector.json`으로 지정한다.

- 경기 단위로 train/val/test를 나누고, 같은 경기의 랠리는 같은 분할에 배정한다.
  랠리 단위로만 나누는 경우에는 같은 경기 환경에서의 평가라는 한계를 기록한다.
- 여러 경기의 프레임 번호는 겹칠 수 있으므로 이미지를 한 폴더로 무작정 합치거나
  프레임 번호를 키로 쓰는 라벨 JSON을 단순 병합하지 않는다.
- 기존 `configs/detector.example.json`의 `datasets/detection/train`, `val`, `test`
  구조는 별도로 데이터를 모았을 때의 예시다. 위처럼 원래 경로를 참조하는 방식도
  지원하며, 두 구조를 모두 만들 필요는 없다. 자동 분할·복사 스크립트는 아직 없다.
- 07은 train으로 학습하고 val로 최적 가중치를 선택한다. 개선 과정의 08은
  `--split val`, 최종 확인은 `--split test`를 사용한다. 09는 07의 가중치와
  이미지 한 장을 입력받으며 08의 출력 파일을 필요로 하지 않는다.
- 원본 영상, 추출·랠리 프레임과 메타데이터는 이후 시간 순서 분석을 위해 보존한다.
  04는 03에서 추출된 프레임만 복사하므로, 낮은 FPS로 추출한 랠리 폴더에는 원본의
  모든 프레임이 들어 있지 않다. 후속 분석에 필요한 시간 해상도는 별도로 정한다.

### 직접 구현한 검출 모델

- `src/detection/model.py`의 `TennisGridDetector`는 Conv·GroupNorm·SiLU 특징 추출부와 클래스별 격자 출력부를 갖는다. stride는 8이며 모든 가중치를 새로 초기화한다.
- 각 클래스·격자에서 객체 점수, 중심 x/y, log 너비/높이를 출력한다. 같은 클래스의 두 중심이 같은 격자에 들어가면 오류로 알리며 정답을 버리지 않는다.
- 객체 점수에는 focal loss, 중심·크기에는 Smooth L1을 적용하고 AdamW로 학습한다. 추론에는 클래스별 NMS를 사용한다.
- `[player, ball]` 클래스별 기존 JSON 또는 통합 JSON을 데이터 설정으로 읽는다. 미검수 프레임을 빈 정답으로 취급하지 않는다.
- 검증 AP50과 검증 손실로 최적 체크포인트를 고른다. AP50은 IoU 0.5 지표이며 COCO mAP50-95와 구분한다.
- 세부 구조·입출력 계약·기본값·제약은 [검출 모델 문서](src/detection/README.md), 경로 설정은 [예시 JSON](configs/detector.example.json)에 정리했다.
- 이 단계의 실제 학습·성능 평가·소프트웨어 테스트는 실행하지 않았다. 성능이 입증된 모델이 아니라 첫 학습 실험을 위한 구현이다.

## 4. common.py의 역할

[common.py](src/common.py)는 CLI에서 가져다 쓰는 필수 보조 모듈이며, 별도로 실행하는 단계가 아니다.

| 함수 | 역할 |
|---|---|
| `frame_number` | `frame_<번호>.jpg`에서 원본 프레임 번호를 읽고 이름 형식을 검사한다. |
| `list_frames` | JPG를 숫자 프레임 번호순으로 정렬하고 잘못된 이름·중복 번호를 거부한다. |
| `positive_finite` | FPS 입력이 0보다 큰 유한한 값인지 검사한다. `nan`, `inf`도 거부한다. |
| `write_json_atomic` | 임시 파일에 JSON을 완성한 뒤 교체하여 쓰기 실패 시 기존 JSON을 보존한다. 파일명만 지정한 출력도 지원한다. |
| `load_frame_metadata` | 선택 → 랠리 → 추출 메타데이터 순으로 가장 최근 단계의 파일을 찾아 읽는다. 없으면 빈 정보를 반환한다. |

여러 CLI가 이 함수들을 사용하므로 `common.py`를 삭제하면 실행에 문제가 생긴다.

## 5. 반영된 저장·검증 규칙

| 대상 | 현재 동작 |
|---|---|
| 프레임 순서 | 파일명의 숫자는 원본 영상 프레임 번호다. 숫자 기준으로 정렬해 100만 이상의 번호도 시간순을 유지한다. |
| FPS·영상 읽기 | FPS와 전체 프레임 수를 검사한다. 추출 FPS는 원본 FPS를 초과할 수 없으며, 영상 자원은 예외 상황에도 반환한다. |
| 프레임 재추출 | 새 폴더 또는 빈 폴더만 허용한다. 기존 파일이 있으면 새 출력 폴더를 지정해야 한다. |
| 추출 결과 확정 | 임시 폴더에 추출하고 이미지·메타데이터 저장이 모두 성공한 뒤 최종 폴더로 옮긴다. 저장 실패나 예상 프레임 수보다 이른 읽기 종료는 오류로 처리한다. |
| 랠리 번호 | 기존 랠리 번호의 최댓값 다음 번호부터 사용한다. 이름 충돌 시 다음 번호를 찾아 기존 랠리와 병합하지 않는다. |
| 랠리 저장 | 복사와 메타데이터 준비를 마친 뒤 새 랠리 폴더로 확정한다. |
| 프레임 선택 경로 | 심볼릭 링크를 포함해 입력과 출력이 실제로 같은 폴더이면 거부한다. |
| 선택 결과 이어하기 | 기록된 출처와 이미지 내용을 검증한다. 다른 출처·다른 내용의 출력은 거부하고, 선택 해제는 관리하는 출력 사본에만 적용한다. |
| bbox 검증 | 면적이 0인 새 박스는 추가하지 않고 새 좌표를 이미지 경계 안으로 제한한다. 기존 JSON의 구조·클래스·좌표도 검사한다. |
| 클래스 분리 | 선수와 공은 별도 JSON 파일을 사용한다. 기존 객체 클래스와 `--type`이 다르면 열기를 거부한다. |

GUI 도구는 프레임이 바뀔 때만 이미지를 읽도록 캐싱한다. 키·마우스 입력, 화면 표시, 저장 처리를 함수로 나누고 기존 기능과 수정 이유를 한국어 주석·docstring으로 설명했다.

## 6. 조작법과 라벨 저장 정책

| 도구 | 키·마우스 조작 |
|---|---|
| 랠리 분할 | A/D 이동, `[`/`]` 이동 폭 조절, S 시작, E 끝, N 랠리 저장, Q 종료 |
| 프레임 선택 | A/D 이동, `[`/`]` 이동 폭 조절, S 선택·해제, W 중간 저장, Q 또는 창 닫기로 저장 후 종료 |
| bbox 라벨링 | 마우스 드래그로 박스 추가, A/D 이동, S 저장, R 현재 박스 초기화, Q 종료 |

라벨링의 저장 정책은 다음과 같다.

- 완성된 bbox 변경은 프레임 이동과 종료 시 자동 저장한다.
- S는 명시적 저장이다. 박스가 없는 프레임도 확인한 빈 라벨로 저장할 수 있다.
- 보기만 한 프레임을 자동으로 빈 라벨로 기록하지 않는다.
- R로 지운 박스도 변경 사항으로 처리하여 저장한다. 미완성 드래그는 이동·종료 시 취소한다.
- 저장 실패 시 현재 편집을 유지하고, 이동·정상 종료를 보류하여 다시 저장할 수 있게 한다.
- 화면에 미검수·저장됨·미저장 상태를 표시한다.

bbox JSON은 기존 형식을 유지한다. 프레임 번호를 키로 사용하고, 각 프레임의 `objects` 목록에 `class`, `x1`, `y1`, `x2`, `y2`를 기록한다. 좌표는 이미지 픽셀 좌표다. 화면의 `player 1`, `player 2`는 해당 프레임의 박스 순서이며 추적 ID가 아니다.

## 7. 메타데이터와 이후 모델링의 연결

| 파일 | 기록하는 정보 |
|---|---|
| `frames_metadata.json` | 원본 영상 경로, 원본 FPS, 샘플링 FPS, 원본 전체 프레임 수, 이미지 크기, 추출 프레임 수, 스키마 버전 |
| `rally_metadata.json` | 알려진 원본 정보에 입력 프레임 폴더, 랠리 시작·끝 원본 프레임 번호, 랠리 프레임 수를 추가 |
| `selection_metadata.json` | 알려진 원본·랠리 정보에 입력 폴더, 선택한 원본 프레임 번호 목록, 선택 프레임 수를 반영 |

메타데이터가 없는 과거 프레임도 사용할 수 있지만 알 수 없는 원본 정보를 추정해서 채우지는 않는다. 원본 FPS와 프레임 번호는 이후 프레임 간 시간 간격을 해석하는 기반이다.

현재 bbox 라벨에는 추적 ID, 코트 구역, 객체 간 관계가 포함되지 않는다. 메타데이터 저장은 상태전이 모델 자체를 구현한 것은 아니다.

## 8. 환경과 저장소 정리

- 누락됐던 `yt-dlp`를 `pyproject.toml` 의존성에 추가하고 `uv.lock`에 반영했다. `requirements.txt`에도 포함되어 있다.
- 영상·오디오를 합치는 다운로드에는 별도로 설치된 ffmpeg가 필요하다.
- 코드의 import와 포맷을 정리했다.
- 모델 구현에서 `ultralytics` 의존성을 제거하고 `torch`를 직접 의존성으로 등록했다. 모델 가중치와 실험 기록을 담는 `runs/`는 Git에서 제외한다.
- 사용자의 요청으로 `tests/`를 삭제하고 README의 테스트 실행 안내도 제거했다. 이 문서는 현재 코드 상태를 정리하며 실제 GUI 동작 검증 완료를 의미하지 않는다.

## 9. 사용 모듈과 공식 Documentation

### 현재 CLI에서 직접 사용하는 외부 패키지

| 패키지 / import 이름 | 프로젝트에서의 역할 | 공식 문서 |
|---|---|---|
| OpenCV / `cv2` | 영상 정보 조회, 프레임 읽기·저장, GUI 창, 키·마우스 입력, bbox 그리기 | [OpenCV 문서](https://docs.opencv.org/5.0/) |
| yt-dlp / `yt_dlp` | 다운로드 형식 선택, `YoutubeDL` 실행, 다운로드 오류 처리 | [공식 README](https://github.com/yt-dlp/yt-dlp#readme), [Python에서 사용하기](https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp) |
| tqdm / `tqdm` | 프레임 추출 진행률 표시 | [tqdm 문서](https://tqdm.github.io/) |
| PyTorch / `torch` | CNN 정의, Dataset/DataLoader, 손실·역전파·최적화, 체크포인트 | [신경망 모듈](https://docs.pytorch.org/docs/stable/nn.html), [데이터 로딩](https://docs.pytorch.org/docs/stable/data.html), [저장·복원](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) |
| NumPy / `numpy` | 이미지 전처리와 난수 시드 설정 | [NumPy 문서](https://numpy.org/doc/stable/) |

### Python 표준 라이브러리

별도 설치 없이 사용하는 모듈이다. 링크는 프로젝트의 Python 요구사항에 맞춰 3.14 문서를 기준으로 한다.

| 모듈 | 프로젝트에서의 역할 | 공식 문서 |
|---|---|---|
| `argparse` | CLI 인자, 도움말, 입력 오류 안내 | [argparse](https://docs.python.org/3.14/library/argparse.html) |
| `pathlib` | 파일·폴더 경로 표현, 실제 경로 확인, 디렉터리 생성과 파일 이동 | [pathlib](https://docs.python.org/3.14/library/pathlib.html) |
| `json` | bbox 라벨과 메타데이터 읽기·쓰기 | [json](https://docs.python.org/3.14/library/json.html) |
| `math` | FPS·좌표의 유한한 숫자 여부 검사 | [math](https://docs.python.org/3.14/library/math.html) |
| `re` | 프레임 파일명과 랠리 폴더 번호 형식 검사 | [re](https://docs.python.org/3.14/library/re.html) |
| `os` | JSON 파일 교체, 디스크 쓰기 동기화, 복사 완료 파일의 하드 링크 생성 | [os](https://docs.python.org/3.14/library/os.html) |
| `tempfile` | 완성 전 결과를 보관하는 임시 파일·폴더 생성 | [tempfile](https://docs.python.org/3.14/library/tempfile.html) |
| `shutil` | 랠리·선택 프레임 복사 | [shutil](https://docs.python.org/3.14/library/shutil.html) |
| `hashlib` | 기존 선택 이미지와 원본의 SHA-256 해시를 비교해 내용 확인 | [hashlib](https://docs.python.org/3.14/library/hashlib.html) |
| `typing` | 공통 JSON 저장 함수의 `Any` 타입 힌트 | [typing](https://docs.python.org/3.14/library/typing.html) |
| `random` | 학습 실행의 Python 난수 시드 설정 | [random](https://docs.python.org/3.14/library/random.html) |

### 의존성에 등록되어 있지만 현재 src에서 직접 import하지 않는 패키지

아래 링크는 등록된 패키지를 참고하기 위한 것이다. 패키지가 등록되어 있다는 사실만으로 모델이나 분석 방식이 확정된 것은 아니다.

| 패키지 | 제공 기능 | 공식 문서 |
|---|---|---|
| `supervision` | 검출 결과 처리와 시각화 등 컴퓨터 비전 도구 | [Supervision 문서](https://supervision.roboflow.com/latest/) |
| `pandas` | 표 형식 데이터 처리·분석 | [pandas 문서](https://pandas.pydata.org/docs/) |

### 개발·실행 도구와 프로젝트 내부 모듈

| 도구 / 모듈 | 역할 | 문서 |
|---|---|---|
| uv | Python 환경·의존성 관리, 잠금 파일 갱신, 명령 실행 | [uv 문서](https://docs.astral.sh/uv/) |
| Ruff | Python 코드 스타일 검사와 포맷 정리 | [Ruff 문서](https://docs.astral.sh/ruff/) |
| FFmpeg | yt-dlp에서 별도 영상·오디오를 병합할 때 사용하는 외부 실행 도구 | [FFmpeg 문서](https://ffmpeg.org/documentation.html) |
| `common.py` | 이 프로젝트에서 직접 작성한 공통 기능 | [소스와 함수 설명](src/common.py), 이 문서의 4절 |
