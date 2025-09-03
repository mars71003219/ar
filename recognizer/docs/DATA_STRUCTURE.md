# Recognizer 데이터 구조 문서

**완전 모듈화된 비디오 분석 시스템의 데이터 구조 상세 설명**

## 개요

Recognizer 시스템에서 사용되는 모든 데이터 구조와 클래스에 대한 상세한 설명을 제공합니다. 이 문서는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.

---

## 목차

1. [핵심 데이터 구조](#핵심-데이터-구조)
2. [포즈 관련 구조](#포즈-관련-구조)
3. [분류 관련 구조](#분류-관련-구조)
4. [설정 관련 구조](#설정-관련-구조)
5. [이벤트 관련 구조](#이벤트-관련-구조)
6. [유틸리티 구조](#유틸리티-구조)
7. [데이터 흐름](#데이터-흐름)
8. [변환 예시](#변환-예시)

---

## 핵심 데이터 구조

### PersonPose

개별 인물의 포즈 데이터를 저장하는 기본 구조입니다.

```python
@dataclass
class PersonPose:
    """개별 person 포즈 데이터"""
    person_id: int                    # 인물 식별자
    bbox: List[float]                 # [x1, y1, x2, y2] 바운딩 박스
    keypoints: np.ndarray            # [17, 3] (x, y, confidence) 키포인트
    score: float                     # 전체 탐지 신뢰도
    track_id: Optional[int] = None   # 추적 ID (ByteTracker)
    timestamp: Optional[float] = None # 타임스탬프

    def to_dict(self) -> Dict[str, Any]
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonPose'
```

**필드 상세:**
- `person_id`: 프레임 내 고유 식별자 (0부터 시작)
- `bbox`: 픽셀 좌표의 바운딩 박스 [x1, y1, x2, y2]
- `keypoints`: COCO 17개 키포인트 (x, y, confidence)
- `score`: 포즈 추정 전체 신뢰도 (0.0-1.0)
- `track_id`: 프레임 간 추적 ID (ByteTracker에서 할당)
- `timestamp`: 포즈 탐지 시점 (초 단위)

**COCO 키포인트 순서:**
```python
KEYPOINT_ORDER = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]
```

### FramePoses

단일 프레임의 모든 인물 포즈를 포함하는 구조입니다.

```python
@dataclass
class FramePoses:
    """단일 프레임의 모든 포즈"""
    frame_idx: int                          # 프레임 번호
    timestamp: float                        # 타임스탬프 (초)
    persons: List[PersonPose]               # 인물 포즈 리스트
    frame_shape: Tuple[int, int, int]       # (H, W, C) 프레임 크기
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FramePoses'
    
    def get_person_count(self) -> int
    def get_max_confidence(self) -> float
```

**사용 예시:**
```python
frame_poses = FramePoses(
    frame_idx=100,
    timestamp=3.33,
    persons=[person1, person2],
    frame_shape=(720, 1280, 3)
)
```

### WindowAnnotation

분류를 위한 윈도우 데이터 구조입니다. MMAction2 표준을 따릅니다.

```python
@dataclass 
class WindowAnnotation:
    """윈도우 어노테이션 데이터 (MMAction2 표준)"""
    window_idx: int                           # 윈도우 번호
    window_id: str                           # 고유 식별자
    keypoint: np.ndarray                     # (M, T, V, C) 키포인트 데이터
    keypoint_score: np.ndarray               # (M, T, V) 키포인트 신뢰도
    total_frames: int                        # 총 프레임 수
    label: Optional[int] = None              # ground truth 라벨 (0: NonFight, 1: Fight)
    frame_data: Optional[List[FramePoses]] = None  # 원본 프레임 데이터
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowAnnotation'
```

**형태 설명:**
- `M`: 최대 인물 수 (max_persons, 기본값: 4)
- `T`: 시간축 길이 (window_size, 기본값: 100)
- `V`: 키포인트 수 (17개, COCO 표준)
- `C`: 좌표 차원 (2D: x,y | 3D: x,y,z)

**데이터 채우기 규칙:**
```python
# 인물이 부족한 경우: 0으로 패딩
# 프레임이 부족한 경우: 마지막 프레임 복제 또는 0 패딩
# 키포인트 누락: 해당 위치에 0 값
```

---

## 포즈 관련 구조

### PoseEstimationConfig

포즈 추정기 설정 구조입니다.

```python
@dataclass
class PoseEstimationConfig:
    """포즈 추정 설정"""
    model_name: str                    # 'rtmo_s', 'rtmo_m', 'rtmo_l'
    inference_mode: str                # 'pth', 'onnx', 'tensorrt'
    device: str = 'cuda:0'            # 디바이스
    score_threshold: float = 0.3       # 탐지 임계값
    input_size: Optional[Tuple[int, int]] = None  # 입력 크기
    
    # 모드별 경로 설정
    config_file: Optional[str] = None   # PyTorch config 파일
    checkpoint_path: Optional[str] = None  # PyTorch 체크포인트
    onnx_path: Optional[str] = None    # ONNX 모델 경로
    tensorrt_path: Optional[str] = None  # TensorRT 엔진 경로
```

**지원 모델:**
- `rtmo_s`: 작고 빠른 모델 (실시간 처리용)
- `rtmo_m`: 중간 크기 모델 (균형)
- `rtmo_l`: 큰 모델 (고정확도)

**추론 모드 특징:**
- `pth`: PyTorch 네이티브 (개발용)
- `onnx`: 크로스 플랫폼 (배포용, 2-3배 빠름)
- `tensorrt`: GPU 최적화 (최고 성능)

---

## 분류 관련 구조

### ClassificationResult

행동 분류 결과 구조입니다.

```python
@dataclass
class ClassificationResult:
    """분류 결과"""
    prediction: int                    # 예측 클래스 (0: NonFight, 1: Fight)  
    confidence: float                  # 신뢰도 (0.0-1.0)
    probabilities: List[float]         # 클래스별 확률 [nonfight_prob, fight_prob]
    model_name: str                    # 사용 모델 ('stgcn', 'stgcn_onnx')
    window_id: Optional[str] = None    # 대상 윈도우 ID
    timestamp: Optional[float] = None  # 분류 시점
    metadata: Optional[Dict[str, Any]] = None  # 추가 메타데이터

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult'
    
    def get_predicted_class_name(self) -> str
    def is_fight(self) -> bool
    def is_confident(self, threshold: float = 0.5) -> bool
```

**사용 예시:**
```python
result = ClassificationResult(
    prediction=1,
    confidence=0.87,
    probabilities=[0.13, 0.87],  # [NonFight, Fight]
    model_name='stgcn_onnx'
)

print(result.get_predicted_class_name())  # 'Fight'
print(result.is_fight())                  # True
print(result.is_confident(0.8))           # True
```

### ActionClassificationConfig

행동 분류기 설정 구조입니다.

```python
@dataclass
class ActionClassificationConfig:
    """행동 분류 설정"""
    model_name: str                    # 'stgcn', 'stgcn_onnx'
    checkpoint_path: str               # 모델 파일 경로
    config_file: Optional[str] = None  # PyTorch 설정 파일
    device: str = 'cuda:0'            # 디바이스
    window_size: int = 100            # 윈도우 크기 (프레임 수)
    confidence_threshold: float = 0.5  # 분류 임계값
    class_names: Optional[List[str]] = None  # 클래스명 ['NonFight', 'Fight']
    max_persons: int = 4              # 최대 인물 수
    input_format: str = 'stgcn'      # 입력 형태 ('stgcn', 'stgcn_onnx')
    coordinate_dimensions: int = 2     # 좌표 차원 (2D 또는 3D)
    expected_keypoint_count: int = 17  # 키포인트 수

    def validate(self) -> bool
```

---

## 설정 관련 구조

### TrackingConfig

객체 추적 설정 구조입니다.

```python
@dataclass
class TrackingConfig:
    """객체 추적 설정"""
    model_name: str = 'bytetrack'     # 추적 알고리즘
    track_thresh: float = 0.5         # 추적 임계값
    track_buffer: int = 30            # 추적 버퍼 크기
    match_thresh: float = 0.8         # 매칭 임계값
    min_box_area: float = 200.0       # 최소 박스 크기
    mot20: bool = False               # MOT20 모드
```

### ScoringConfig

점수 계산 설정 구조입니다.

```python
@dataclass 
class ScoringConfig:
    """점수 계산 설정"""
    model_name: str = 'region_based'   # 'region_based', 'movement_based'
    distance_threshold: float = 100.0  # 상호작용 거리 임계값
    movement_threshold: float = 50.0   # 움직임 임계값
    temporal_window: int = 10          # 시간 윈도우
    weight_distance: float = 0.5       # 거리 가중치
    weight_movement: float = 0.3       # 움직임 가중치
    weight_overlap: float = 0.2        # 겹침 가중치
```

---

## 이벤트 관련 구조

### EventData

이벤트 데이터 구조입니다.

```python
@dataclass
class EventData:
    """이벤트 데이터"""
    event_type: str                   # 'violence_start', 'violence_end', 'violence_ongoing'
    timestamp: float                  # 이벤트 발생 시점
    confidence: float                 # 신뢰도
    duration: Optional[float] = None  # 지속 시간 (초)
    window_id: Optional[str] = None   # 관련 윈도우 ID
    frame_number: Optional[int] = None  # 프레임 번호
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventData'
```

### EventConfig

이벤트 관리 설정 구조입니다.

```python
@dataclass
class EventConfig:
    """이벤트 관리 설정"""
    alert_threshold: float = 0.8         # 알림 임계값
    min_consecutive_detections: int = 3  # 최소 연속 탐지 수
    normal_threshold: float = 0.5        # 정상 임계값  
    min_consecutive_normal: int = 2      # 최소 연속 정상 수
    min_event_duration: float = 2.0      # 최소 이벤트 지속 시간
    max_event_duration: float = 10.0     # 최대 이벤트 지속 시간
    cooldown_duration: float = 5.0       # 쿨다운 시간
    enable_ongoing_alerts: bool = True   # 진행중 알림 활성화
    ongoing_alert_interval: float = 30.0  # 진행중 알림 간격
    save_event_log: bool = True          # 이벤트 로그 저장
    event_log_format: str = 'json'       # 로그 형식 ('json', 'csv')
    event_log_path: str = 'output/event_logs'  # 로그 저장 경로
```

---

## 유틸리티 구조

### ProcessingStats

처리 통계 구조입니다.

```python
@dataclass
class ProcessingStats:
    """처리 통계"""
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    avg_fps: float = 0.0
    pose_estimation_fps: float = 0.0
    tracking_fps: float = 0.0
    classification_fps: float = 0.0
    total_classifications: int = 0
    successful_classifications: int = 0
    
    processing_times: Dict[str, List[float]] = field(default_factory=dict)
    
    def update_fps(self, stage: str, fps: float)
    def add_processing_time(self, stage: str, time_ms: float)
    def get_average_time(self, stage: str) -> float
    def to_dict(self) -> Dict[str, Any]
```

### MultiProcessConfig

멀티프로세스 설정 구조입니다.

```python
@dataclass
class MultiProcessConfig:
    """멀티프로세스 설정"""
    enabled: bool = False
    num_processes: int = 4
    gpus: List[int] = field(default_factory=lambda: [0, 1])
    gpu_assignments: Optional[List[int]] = None  # 호환성 유지
    chunk_size: Optional[int] = None
    max_queue_size: int = 100
    timeout: float = 300.0  # 5분
```

---

## 데이터 흐름

### 1. Inference 모드 데이터 흐름

```
입력 영상
    ↓
[포즈 추정] → PersonPose[]
    ↓  
[추적] → PersonPose[] (track_id 추가)
    ↓
[점수 계산] → FramePoses (scores 추가)
    ↓
[윈도우 처리] → WindowAnnotation[]
    ↓
[행동 분류] → ClassificationResult[]
    ↓
[이벤트 관리] → EventData[]
    ↓
[시각화/저장] → 출력 파일
```

### 2. Annotation 모드 데이터 흐름

```
입력 비디오 폴더
    ↓
[Stage 1] → PKL 파일 (FramePoses[])
    ↓
[Stage 2] → PKL 파일 (WindowAnnotation[])
    ↓
[Stage 3] → PKL 파일 (ClassificationResult[])
    ↓
[시각화] → 어노테이션된 비디오
```

### 3. 멀티프로세스 데이터 흐름

```
입력 데이터 분할
    ↓
[프로세스 1] → GPU 0 → 결과 1
[프로세스 2] → GPU 1 → 결과 2  
[프로세스 N] → GPU N → 결과 N
    ↓
결과 병합 → 최종 출력
```

---

## 변환 예시

### PersonPose → WindowAnnotation 변환

```python
def create_window_annotation(frame_poses_list: List[FramePoses], 
                           window_size: int = 100, 
                           max_persons: int = 4) -> WindowAnnotation:
    """FramePoses 리스트를 WindowAnnotation으로 변환"""
    
    # 데이터 초기화
    M, T, V, C = max_persons, window_size, 17, 2
    keypoint = np.zeros((M, T, V, C), dtype=np.float32)
    keypoint_score = np.ones((M, T, V), dtype=np.float32)
    
    # 프레임별 데이터 채우기
    for t, frame_poses in enumerate(frame_poses_list[:T]):
        for m, person in enumerate(frame_poses.persons[:M]):
            if person.keypoints is not None:
                keypoint[m, t] = person.keypoints[:, :2]  # x, y만
                keypoint_score[m, t] = person.keypoints[:, 2]  # confidence
    
    return WindowAnnotation(
        window_idx=0,
        window_id=f"window_{timestamp}",
        keypoint=keypoint,
        keypoint_score=keypoint_score, 
        total_frames=T,
        frame_data=frame_poses_list
    )
```

### ONNX 모델 입력 형태 변환

```python
def prepare_onnx_input(window_data: WindowAnnotation) -> np.ndarray:
    """WindowAnnotation을 ONNX 입력 형태로 변환"""
    
    # MMAction2 형태: (M, T, V, C) → ONNX 형태: (1, M, T, V, C)
    keypoint_data = window_data.keypoint
    
    if keypoint_data.shape[-1] == 2:
        # 2D → 3D 변환 (z=0 추가)
        M, T, V, C = keypoint_data.shape
        z_coords = np.zeros((M, T, V, 1), dtype=np.float32)
        keypoint_data = np.concatenate([keypoint_data, z_coords], axis=-1)
    
    # 배치 차원 추가
    batch_input = np.expand_dims(keypoint_data, axis=0)
    
    return batch_input.astype(np.float32)
```

### Temperature Scaling 적용

```python
def apply_temperature_scaling(raw_logits: np.ndarray, 
                            temperature: float = 0.005) -> np.ndarray:
    """ONNX raw logits에 temperature scaling 적용"""
    
    # Temperature scaling
    scaled_logits = raw_logits * temperature
    
    # Softmax 적용
    exp_scores = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities
```

---

## 성능 고려사항

### 메모리 최적화

```python
# 큰 윈도우 처리 시 메모리 최적화
@dataclass
class WindowAnnotation:
    # 메모리 사용량 감소를 위한 dtype 최적화
    keypoint: np.ndarray          # float32 사용 (float64 대신)
    keypoint_score: np.ndarray    # float32 사용
    
    def compress(self):
        """메모리 압축"""
        if self.keypoint.dtype != np.float32:
            self.keypoint = self.keypoint.astype(np.float32)
        if self.keypoint_score.dtype != np.float32:
            self.keypoint_score = self.keypoint_score.astype(np.float32)
```

### 직렬화 최적화

```python
# PKL 파일 크기 최적화
import pickle

def save_compressed(data: Any, filepath: str):
    """압축된 형태로 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_compressed(filepath: str) -> Any:
    """압축된 형태에서 로드"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
```

---

*이 문서는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.*