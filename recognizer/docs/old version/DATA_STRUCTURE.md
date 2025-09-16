# 데이터 파이프라인 PKL 구조 문서

## 개요
RWF-2000 폭력 탐지 데이터셋 처리를 위한 3단계 파이프라인의 PKL 파일 구조를 설명합니다.

- **Stage1**: 포즈 추정 (Pose Estimation)
- **Stage2**: 추적 및 스코어링 (Tracking & Scoring)  
- **Stage3**: 최종 데이터셋 생성 (Dataset Creation)

---

## Stage1: Pose Estimation PKL 구조

### 파일 위치
```
output/RWF-2000/stage1_poses/unknown_s0.2_n0.65/
├── {video_name}_stage1_poses.pkl
```

### 데이터 구조
```python
class VisualizationData:
    video_name: str                    # 비디오 파일명 (확장자 제외)
    frame_data: List[FramePoses]       # 프레임별 포즈 데이터
    stage_info: Dict[str, Any]         # 스테이지 메타데이터
    poses_only: List[FramePoses]       # frame_data와 동일 (호환성)
    poses_with_tracking: None          # Stage2에서 사용
    tracking_info: None                # Stage2에서 사용
    poses_with_scores: None            # 향후 확장용
    scoring_info: None                 # 향후 확장용
    classification_results: None       # 향후 확장용
```

### FramePoses 구조
```python
class FramePoses:
    frame_idx: int                     # 프레임 인덱스 (0부터 시작)
    persons: List[Person]              # 프레임 내 감지된 사람들
    timestamp: float                   # 프레임 타임스탬프 (초)
    image_shape: Tuple[int, int]       # 이미지 크기 (height, width)
    metadata: Dict[str, Any]           # 추가 메타데이터
```

### Person 구조
```python
class Person:
    person_id: int                     # 사람 ID (프레임 내 고유)
    bbox: Tuple[float, float, float, float]  # 바운딩 박스 (x1, y1, x2, y2)
    keypoints: np.ndarray              # 키포인트 좌표 [17, 2] (COCO17 형식)
    score: float                       # 사람 감지 신뢰도 점수
    track_id: None                     # Stage2에서 할당됨
    timestamp: float                   # 프레임 타임스탬프
    metadata: Dict[str, Any]           # 추가 메타데이터
```

### stage_info 구조
```python
{
    'stage': 'pose_estimation',
    'total_frames': int,               # 총 프레임 수
    'config': Dict,                    # 포즈 추정 설정
    'original_path': str,              # 원본 비디오 파일 경로
    'original_label': int              # 폴더 기반 라벨 (0: NonFight, 1: Fight)
}
```

### 키포인트 형식
- **포맷**: COCO17 (17개 키포인트)
- **좌표**: (x, y) 픽셀 좌표
- **키포인트 순서**:
  ```
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
  ```

---

## Stage2: Tracking & Scoring PKL 구조

### 파일 위치
```
output/RWF-2000/stage2_tracking/bytetrack_h0.3_l0.1_t0.2/
├── {video_name}_stage2_tracking.pkl
```

### 데이터 구조
```python
class VisualizationData:
    video_name: str                    # 비디오 파일명
    frame_data: List[FramePoses]       # 원본 프레임 데이터 (참조용)
    stage_info: Dict[str, Any]         # 스테이지 메타데이터
    poses_only: None                   # Stage1에서 사용됨
    poses_with_tracking: List[FramePoses]  # 추적이 적용된 프레임 데이터
    tracking_info: Dict[str, Any]      # 추적 메타데이터
    poses_with_scores: None            # 향후 확장용
    scoring_info: None                 # 향후 확장용
    classification_results: None       # 향후 확장용
```

### 추적이 적용된 Person 구조
```python
class Person:
    person_id: int                     # 사람 ID (프레임 내)
    bbox: Tuple[float, float, float, float]  # 바운딩 박스
    keypoints: np.ndarray              # 키포인트 좌표 [17, 2]
    score: float                       # 사람 감지 신뢰도
    track_id: int                      # 추적 ID (비디오 전체에서 고유)
    timestamp: float                   # 타임스탬프
    metadata: Dict[str, Any]           # 스코어링 정보 포함
```

### stage_info 구조
```python
{
    'stage': 'tracking_scoring',
    'total_frames': int,
    'tracking_config': Dict,           # ByteTracker 설정
    'scoring_config': Dict,            # 스코어링 설정
    'original_path': str,              # 원본 비디오 파일 경로 (보존)
    'original_label': int              # 원본 라벨 (보존)
}
```

### tracking_info 구조
```python
{
    'total_tracks': int,               # 생성된 총 추적 수
    'config': Dict                     # 추적 설정 정보
}
```

---

## Stage3: Dataset Creation PKL 구조

### 파일 위치
```
output/RWF-2000/stage3_dataset/unknown_s0.2_n0.65_bytetrack_h0.3_l0.1_t0.2_split0.7-0.2-0.1/
├── train.pkl                        # 훈련 데이터셋
├── val.pkl                          # 검증 데이터셋
├── test.pkl                         # 테스트 데이터셋
├── metadata.json                    # 데이터셋 메타데이터
└── stage3_path_info.json           # 경로 정보
```

### 데이터 구조
각 PKL 파일은 샘플 딕셔너리의 리스트입니다:

```python
List[Dict[str, Any]]
```

### 샘플 구조
```python
{
    'frame_dir': str,                  # 프레임 식별자 (예: "V_960")
    'total_frames': int,               # 총 프레임 수
    'img_shape': Tuple[int, int],      # 이미지 크기 (height, width)
    'original_shape': Tuple[int, int], # 원본 이미지 크기
    'label': int,                      # 클래스 라벨 (0: NonFight, 1: Fight)
    'keypoint': np.ndarray,            # 키포인트 데이터 [M, T, V, C]
    'keypoint_score': np.ndarray       # 키포인트 신뢰도 점수 [M, T, V]
}
```

### 키포인트 텐서 차원
- **keypoint**: `[M, T, V, C]`
  - `M`: 최대 사람 수 (max_persons=4)
  - `T`: 시간 프레임 수 (total_frames)
  - `V`: 키포인트 수 (17개, COCO17)
  - `C`: 좌표 차원 (2차원: x, y)

- **keypoint_score**: `[M, T, V]`
  - `M`: 최대 사람 수 (4)
  - `T`: 시간 프레임 수
  - `V`: 키포인트 수 (17개)

### 데이터 범위
- **keypoint**: 픽셀 좌표 (예: -18.58 ~ 222.77)
- **keypoint_score**: 신뢰도 점수 (0.0 ~ 1.0)
- **label**: 0 (NonFight) 또는 1 (Fight)

---

## 데이터셋 통계

### 최종 데이터셋 구성
- **총 샘플**: 5,694개
- **훈련 세트**: 3,985개 (70%)
- **검증 세트**: 1,138개 (20%)
- **테스트 세트**: 571개 (10%)

### 클래스 분포
- **NonFight(0)**: 2,583개 (45.4%)
- **Fight(1)**: 3,111개 (54.6%)

### 실패한 파일
- **실패 파일 수**: 253개
- **실패 원인**: 키포인트 데이터 부족, 추적 실패 등

---

## 사용 예시

### Stage1 데이터 로드
```python
import pickle

with open('output/RWF-2000/stage1_poses/unknown_s0.2_n0.65/{video_name}_stage1_poses.pkl', 'rb') as f:
    stage1_data = pickle.load(f)

# 첫 번째 프레임의 첫 번째 사람 키포인트 접근
keypoints = stage1_data.frame_data[0].persons[0].keypoints  # [17, 2]
original_label = stage1_data.stage_info['original_label']   # 0 or 1
```

### Stage2 데이터 로드
```python
with open('output/RWF-2000/stage2_tracking/bytetrack_h0.3_l0.1_t0.2/{video_name}_stage2_tracking.pkl', 'rb') as f:
    stage2_data = pickle.load(f)

# 추적된 첫 번째 프레임의 첫 번째 사람 정보
person = stage2_data.poses_with_tracking[0].persons[0]
track_id = person.track_id  # 추적 ID
```

### Stage3 데이터 로드
```python
with open('output/RWF-2000/stage3_dataset/unknown_s0.2_n0.65_bytetrack_h0.3_l0.1_t0.2_split0.7-0.2-0.1/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

# 첫 번째 샘플 정보
sample = train_data[0]
keypoints = sample['keypoint']      # [4, T, 17, 2]
scores = sample['keypoint_score']   # [4, T, 17]
label = sample['label']             # 0 or 1
```

---

## 주요 특징

1. **라벨 보존**: 원본 RWF-2000의 폴더 구조 기반 라벨이 모든 스테이지에서 보존됩니다.

2. **추적 일관성**: Stage2에서 ByteTracker를 통해 비디오 전체에서 일관된 person tracking이 제공됩니다.

3. **MMAction2 호환성**: Stage3 출력 형식은 MMAction2 STGCN++ 모델과 직접 호환됩니다.

4. **메타데이터 풍부**: 각 스테이지에서 처리 설정과 통계 정보를 포함합니다.

5. **에러 추적**: 실패한 파일들이 metadata에 기록되어 디버깅이 용이합니다.