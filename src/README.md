# 테니스 데이터 준비 CLI

파일명의 번호 순서대로 실행한다. 각 스크립트 상단에는 목적과 사용 예시가,
함수에는 처리 흐름과 저장 규칙을 설명하는 한국어 주석이 있다.
실행 예시는 프로젝트 루트 기준이며, 실제 데이터 경로로 바꿔 사용한다.

## 실행 순서

| 단계 | 스크립트 | 입력 → 출력 |
|---|---|---|
| 01 | `01_video_downloader.py` | YouTube URL → 영상 |
| 02 | `02_video_info.py` | 영상 → 해상도·FPS·재생시간 출력 |
| 03 | `03_frame_extractor.py` | 영상 → 프레임 JPG + `frames_metadata.json` |
| 04 | `04_rally_splitter.py` | 추출 프레임 → 랠리별 폴더 + `rally_metadata.json` |
| 05 | `05_frame_selector.py` | 개별 랠리 폴더 → 선택 프레임 + `selection_metadata.json` |
| 06 | `06_annotator.py` | 선택 프레임 → 선수 또는 공 bbox JSON |

```bash
uv run python src/01_video_downloader.py "YOUTUBE_URL" --fps 25 --output datasets/videos/match01.mp4
uv run python src/02_video_info.py datasets/videos/match01.mp4
uv run python src/03_frame_extractor.py datasets/videos/match01.mp4 --fps 2 --output datasets/match01/frames
uv run python src/04_rally_splitter.py --frames datasets/match01/frames --output datasets/match01/rallies
uv run python src/05_frame_selector.py --frames datasets/match01/rallies/rally_0001 --output datasets/match01/selected/rally_0001
uv run python src/06_annotator.py --type player --frames datasets/match01/selected/rally_0001 --output datasets/match01/annotations/rally_0001_player.json
uv run python src/06_annotator.py --type ball --frames datasets/match01/selected/rally_0001 --output datasets/match01/annotations/rally_0001_ball.json
```

다운로더의 FPS는 제공되는 영상 소스를 고르는 조건이다. `uv run yt-dlp -F "URL"`로
지원 형식을 먼저 확인할 수 있다. 별도 영상·오디오 병합에는 ffmpeg가 필요하다.
추출기의 FPS는 이미지 샘플링 빈도다. 예시의 2 FPS는 실행 예시이며 모델링 요구에 따라 정한다.

## 마우스와 키보드

| 도구 | 조작 |
|---|---|
| 랠리 분할 | A/D 이동, `[`/`]` 이동 폭 조절, S 시작, E 끝, N 랠리 저장, Q 종료 |
| 프레임 선택 | A/D 이동, `[`/`]` 이동 폭 조절, S 선택/해제, W 저장, Q 저장 후 종료 |
| bbox 라벨링 | 마우스 드래그로 박스 추가, A/D 이동, S 저장, R 현재 박스 초기화, Q 종료 |

라벨링에서 완성된 박스 변경은 프레임 이동과 종료 시 자동 저장된다.
S는 명시적으로 저장하며, 박스가 없는 프레임도 확인한 빈 라벨로 저장할 수 있다.
아무 작업도 하지 않은 프레임은 자동으로 빈 라벨이 되지 않는다.
선수와 공은 서로 다른 JSON 파일에 저장한다. 다른 클래스의 주석 파일을 열면 오류로 알려준다.

## 저장 규칙과 파일 구조

- 프레임 파일명의 숫자는 원본 영상의 프레임 번호다. 숫자 기준으로 정렬한다.
- 추출은 새 폴더 또는 빈 폴더만 허용한다. 재실행할 때 새 출력 폴더를 지정한다.
- 새 랠리는 기존 번호의 최댓값 다음 번호를 사용하며 기존 랠리에 병합하지 않는다.
- 프레임 선택의 입력·출력 폴더는 달라야 한다. 선택 해제는 출력의 복사본에만 적용한다.
- 메타데이터는 원본 영상 경로, 원본·샘플링 FPS 등을 다음 단계에 전달한다.
  메타데이터가 없는 과거 프레임도 사용할 수 있지만, 모르는 정보를 추정해서 채우지는 않는다.
- bbox JSON은 기존처럼 프레임 번호별 `objects`에 `class`, `x1`, `y1`, `x2`, `y2`를 저장한다.
  좌표는 이미지 픽셀 좌표이며, 화면의 박스 번호는 추적 ID가 아니다.
- `common.py`는 프레임 이름 검증·숫자 정렬·안전한 JSON 저장을 공유하는 보조 모듈이다.

## 코드 스타일 확인

프로젝트 가상환경에서 코드 스타일을 확인한다.

```bash
uv run ruff check src
uv run ruff format --check src
```
