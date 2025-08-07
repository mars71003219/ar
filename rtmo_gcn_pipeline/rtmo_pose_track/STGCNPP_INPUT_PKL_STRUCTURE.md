# ST-GCN++ 입력 PKL 파일 데이터 구조 분석

이 문서는 `V_143_windows.pkl` 파일의 분석 결과를 바탕으로, 분리된 파이프라인의 2단계(추적 및 랭킹) 결과물이자 3단계(행동 인식)의 입력으로 사용되는 PKL 파일의 상세한 데이터 구조를 설명합니다.

## 1. 최상위 계층 구조

PKL 파일은 하나의 비디오 전체에 대한 정보를 담고 있는 파이썬 딕셔너리(`dict`)입니다. 이 딕셔너리는 비디오의 메타데이터와 여러 개의 "윈도우(window)" 데이터를 리스트 형태로 포함합니다.

```
/ (root: dict)
├── video_name: str
├── label_folder: str
├── label: int
├── dataset_name: str
├── total_frames: int
├── num_windows: int
├── windows: list[dict]
└── tracking_settings: dict
```

| 키 | 타입 | 설명 |
| --- | --- | --- |
| `video_name` | `str` | 원본 비디오의 이름 (예: 'V_143'). |
| `label_folder` | `str` | 비디오의 클래스 폴더 이름 (예: 'Fight'). |
| `label` | `int` | 비디오의 실제 레이블. 'Fight'는 1, 'NonFight'는 0. |
| `dataset_name` | `str` | 이 데이터가 생성된 파이프라인의 이름 (예: 'separated_pipeline'). |
| `total_frames` | `int` | 원본 비디오의 전체 프레임 수. |
| `num_windows` | `int` | 이 비디오에서 추출된 슬라이딩 윈도우의 총 개수. |
| `windows` | `list` | 각 슬라이딩 윈도우의 상세 정보를 담은 딕셔너리들의 리스트. **핵심 데이터 부분입니다.** |
| `tracking_settings`| `dict` | 이 데이터를 생성할 때 사용된 추적 관련 하이퍼파라미터 정보. |

---

## 2. `windows` 리스트 상세 구조

`windows` 키의 값은 리스트이며, 각 요소는 하나의 슬라이딩 윈도우를 나타내는 딕셔너리입니다. 첫 번째 윈도우(인덱스 0)의 구조는 다음과 같습니다.

```
/windows[0] (dict)
├── window_idx: int
├── start_frame: int
├── end_frame: int
├── num_frames: int
├── annotation: dict
├── segment_video_path: NoneType
├── persons_ranking: list
└── composite_score: float
```

| 키 | 타입 | 설명 |
| --- | --- | --- |
| `window_idx` | `int` | 윈도우의 순서 인덱스 (0부터 시작). |
| `start_frame` | `int` | 원본 비디오 기준 윈도우의 시작 프레임. |
| `end_frame` | `int` | 원본 비디오 기준 윈도우의 끝 프레임. |
| `num_frames` | `int` | 윈도우의 길이 (프레임 수). `clip_len`과 동일. |
| `annotation` | `dict` | **ST-GCN++ 모델의 입력으로 직접 사용될 핵심 포즈 및 추적 데이터.** 상세 구조는 아래 참조. |
| `segment_video_path` | `NoneType` | (사용되지 않음) 세그먼트 비디오 경로. |
| `persons_ranking` | `list` | (사용되지 않음) 인물 랭킹 정보. `annotation` 내부의 `persons`로 대체됨. |
| `composite_score` | `float` | (사용되지 않음) 윈도우의 종합 점수. |

---

## 3. `annotation` 딕셔너리 상세 구조

`annotation` 딕셔너리는 ST-GCN++ 모델이 요구하는 형식에 맞춰진 데이터와 추적/랭킹 과정에서 생성된 풍부한 메타데이터를 포함합니다.

```
/windows[0]/annotation (dict)
├── frame_ind: int
├── img_shape: tuple
├── original_shape: tuple
├── persons: dict
├── total_persons: int
└── total_frames: int
```

| 키 | 타입 | 설명 |
| --- | --- | --- |
| `frame_ind` | `int` | 윈도우의 시작 프레임 인덱스 (항상 0). |
| `img_shape` | `tuple` | 모델 입력 이미지의 해상도 `(높이, 너비)`. |
| `original_shape` | `tuple` | 원본 비디오의 해상도 `(높이, 너비)`. |
| `persons` | `dict` | **윈도우 내에서 추적된 모든 인물들의 상세 정보.** 키는 `track_id`이며, 값은 각 인물의 데이터 딕셔너리. |
| `total_persons` | `int` | 이 윈도우에서 추적된 총 인물 수. |
| `total_frames` | `int` | 이 윈도우의 총 프레임 수 (`clip_len`). |

---

## 4. `persons` 딕셔너리 상세 구조 (개별 인물 데이터)

`persons` 딕셔너리는 이 파일에서 가장 핵심적인 정보를 담고 있습니다. 각 키는 `track_id`이며, 값은 해당 인물의 모든 정보를 담은 딕셔너리입니다.

```
/windows[0]/annotation/persons/{track_id} (dict)
├── keypoint: numpy.ndarray
├── keypoint_score: numpy.ndarray
├── num_keypoints: int
├── track_id: int
├── composite_score: float
├── rank: int
├── quality_score: float
├── num_frames: int
├── score_breakdown: dict
└── region_breakdown: dict
```

| 키 | 타입 / 형태 | 설명 |
| --- | --- | --- |
| `keypoint` | `numpy.ndarray` (1, T, V, C) | **ST-GCN++ 모델의 주 입력.** `(1, 100, 17, 2)` 형태의 4D 배열. (1, 프레임 수, 관절 수, xy좌표) |
| `keypoint_score` | `numpy.ndarray` (1, T, V) | **ST-GCN++ 모델의 보조 입력.** `(1, 100, 17)` 형태의 3D 배열. 각 관절의 신뢰도 점수. |
| `num_keypoints` | `int` | 관절의 총 개수 (17). |
| `track_id` | `int` | 이 인물의 고유 추적 ID. |
| `composite_score` | `float` | 이 인물의 중요도를 나타내는 **종합 점수**. 이 점수를 기준으로 `rank`가 결정됨. |
| `rank` | `int` | 이 윈도우 내에서 이 인물의 중요도 순위 (1부터 시작). |
| `quality_score` | `float` | 포즈 품질 점수 (키포인트 신뢰도의 평균). |
| `num_frames` | `int` | 이 인물이 윈도우 내에서 실제로 나타난 프레임 수. |
| `score_breakdown` | `dict` | 종합 점수를 구성하는 5가지 세부 점수. (상세 구조 아래 참조) |
| `region_breakdown`| `dict` | 위치 점수(`position`)를 구성하는 5개 영역별 세부 점수. (상세 구조 아래 참조) |

### 4.1. `score_breakdown` 상세 구조

```
/score_breakdown (dict)
├── movement: float
├── position: float
├── interaction: float
├── temporal_consistency: float
└── persistence: float
```

- 이 5가지 점수는 `tracking_settings`에 정의된 가중치와 곱해져 `composite_score`를 계산하는 데 사용됩니다.

### 4.2. `region_breakdown` 상세 구조

```
/region_breakdown (dict)
├── top_left: float
├── top_right: float
├── bottom_left: float
├── bottom_right: float
└── center_overlap: float
```

- 화면을 5개 영역으로 나누어 각 영역에 인물이 얼마나 위치하는지를 나타내는 점수입니다. 이 점수들의 가중 합이 `score_breakdown`의 `position` 점수가 됩니다.
