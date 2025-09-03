# Recognizer API 레퍼런스

**완전 모듈화된 비디오 분석 시스템 API 문서**

## 개요

Recognizer 시스템의 모든 클래스, 메서드, 함수에 대한 상세한 API 레퍼런스를 제공합니다. 이 문서는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.

---

## 목차

1. [Core API](#core-api)
2. [모드 관리 API](#모드-관리-api)  
3. [포즈 추정 API](#포즈-추정-api)
4. [행동 분류 API](#행동-분류-api)
5. [추적 API](#추적-api)
6. [점수 계산 API](#점수-계산-api)
7. [시각화 API](#시각화-api)
8. [유틸리티 API](#유틸리티-api)
9. [데이터 구조](#데이터-구조)
10. [설정 구조](#설정-구조)

---

## Core API

### ModeManager

메인 모드 관리자 클래스로, 모든 실행 모드를 등록하고 실행합니다.

#### 클래스 정의
```python
class ModeManager:
    """통합 모드 관리자"""
```

#### 생성자
```python
def __init__(self, config: Dict[str, Any])
```

**매개변수:**
- `config` (Dict[str, Any]): 전체 시스템 설정

**사용 예시:**
```python
from core import ModeManager

config = load_config('config.yaml')
manager = ModeManager(config)
```

#### 주요 메서드

##### execute()
```python
def execute(self, mode_name: str) -> bool
```

지정된 모드를 실행합니다.

**매개변수:**
- `mode_name` (str): 실행할 모드명 (예: 'inference.analysis')

**반환값:** 
- `bool`: 실행 성공 여부

**사용 예시:**
```python
success = manager.execute('inference.realtime')
```

##### list_modes()
```python
def list_modes() -> Dict[str, str]
```

사용 가능한 모든 모드를 반환합니다.

**반환값:**
- `Dict[str, str]`: 모드명과 설명의 딕셔너리

**사용 예시:**
```python
modes = manager.list_modes()
for mode, description in modes.items():
    print(f"{mode}: {description}")
```

### BaseMode

모든 실행 모드의 추상 기본 클래스입니다.

#### 클래스 정의
```python
class BaseMode(ABC):
    """모든 모드의 기본 클래스"""
```

#### 추상 메서드

##### execute()
```python
@abstractmethod
def execute(self) -> bool
```

모드 실행 로직을 구현해야 합니다.

**반환값:**
- `bool`: 실행 성공 여부

---

## 모드 관리 API

### Inference 모드

#### AnalysisMode
```python
class AnalysisMode(BaseMode):
    """분석 모드 - 완전한 비디오 분석 및 결과 저장"""
```

**기능:**
- 전체 비디오 처리
- JSON/PKL 파일 생성
- 평가 모드 지원 (차트, 혼동행렬, 보고서)

#### RealtimeMode
```python
class RealtimeMode(BaseMode):
    """실시간 모드 - 라이브 비디오 스트림 처리"""
```

**기능:**
- 실시간 디스플레이
- 이벤트 감지 및 알림
- 성능 모니터링

#### VisualizeMode
```python
class VisualizeMode(BaseMode):
    """시각화 모드 - PKL 데이터 기반 오버레이 생성"""
```

**기능:**
- PKL 파일 기반 시각화
- 고품질 오버레이 생성

### Annotation 모드

#### AnnotationPipelineMode
```python
class AnnotationPipelineMode(BaseMode):
    """통합 어노테이션 파이프라인"""
```

**기능:**
- Stage 1-3 자동 연결
- 최적화된 메모리 사용

#### Stage1Mode
```python
class Stage1Mode(BaseMode):
    """Stage 1: 포즈 추정 결과 생성"""
```

#### Stage2Mode  
```python
class Stage2Mode(BaseMode):
    """Stage 2: 윈도우 생성"""
```

#### Stage3Mode
```python
class Stage3Mode(BaseMode):
    """Stage 3: 행동 분류"""
```

#### AnnotationVisualizeMode
```python
class AnnotationVisualizeMode(BaseMode):
    """어노테이션 결과 시각화"""
```

---

## 포즈 추정 API

### BasePoseEstimator

모든 포즈 추정기의 추상 기본 클래스입니다.

#### 클래스 정의
```python
class BasePoseEstimator(ABC):
    """포즈 추정기 기본 클래스"""
```

#### 추상 메서드

##### estimate_pose()
```python
@abstractmethod
def estimate_pose(self, frame: np.ndarray) -> List[PersonPose]
```

프레임에서 포즈를 추정합니다.

**매개변수:**
- `frame` (np.ndarray): 입력 이미지 (H, W, 3)

**반환값:**
- `List[PersonPose]`: 탐지된 인물들의 포즈 리스트

### RTMO 추정기들

#### RTMOPoseEstimator
```python
class RTMOPoseEstimator(BasePoseEstimator):
    """RTMO PyTorch 포즈 추정기"""
```

#### RTMOONNXEstimator
```python
class RTMOONNXEstimator(BasePoseEstimator):
    """RTMO ONNX 포즈 추정기"""
```

**특징:**
- 빠른 추론 속도
- 메모리 효율성

#### RTMOTensorRTEstimator
```python
class RTMOTensorRTEstimator(BasePoseEstimator):
    """RTMO TensorRT 포즈 추정기"""
```

**특징:**
- 최고 성능
- GPU 전용

---

## 행동 분류 API

### BaseActionClassifier

행동 분류기의 기본 클래스입니다.

#### 클래스 정의
```python
class BaseActionClassifier(ABC):
    """행동 분류기 기본 클래스"""
```

#### 생성자
```python
def __init__(self, config: ActionClassificationConfig)
```

#### 주요 메서드

##### initialize_model()
```python
@abstractmethod
def initialize_model(self) -> bool
```

모델을 초기화합니다.

##### classify_single_window()
```python
@abstractmethod
def classify_single_window(self, window_data: WindowAnnotation) -> ClassificationResult
```

단일 윈도우를 분류합니다.

**매개변수:**
- `window_data` (WindowAnnotation): 윈도우 포즈 데이터

**반환값:**
- `ClassificationResult`: 분류 결과

##### classify_multiple_windows()
```python
def classify_multiple_windows(self, windows: List[WindowAnnotation]) -> List[ClassificationResult]
```

다중 윈도우를 배치로 분류합니다.

### ST-GCN++ 분류기들

#### STGCNActionClassifier
```python
class STGCNActionClassifier(BaseActionClassifier):
    """ST-GCN++ PyTorch 분류기"""
```

#### STGCNONNXClassifier  
```python
class STGCNONNXClassifier(BaseActionClassifier):
    """ST-GCN++ ONNX 분류기"""
```

**특징:**
- Temperature scaling 자동 적용
- Raw logits → 확률값 변환
- PyTorch와 동등한 성능

**Temperature Scaling:**
```python
temperature = 0.005  # 자동 조정
scaled_scores = pred_scores * temperature
probabilities = softmax(scaled_scores)
```

---

## 추적 API

### ByteTrackerWrapper

ByteTracker 기반 객체 추적기입니다.

#### 클래스 정의
```python
class ByteTrackerWrapper:
    """ByteTracker 래퍼 클래스"""
```

#### 주요 메서드

##### update()
```python
def update(self, detections: List[Detection]) -> List[Track]
```

추적을 업데이트합니다.

**매개변수:**
- `detections` (List[Detection]): 현재 프레임의 탐지 결과

**반환값:**
- `List[Track]`: 업데이트된 추적 결과

---

## 점수 계산 API

### RegionBasedScorer

영역 기반 상호작용 점수 계산기입니다.

#### 클래스 정의
```python
class RegionBasedScorer:
    """영역 기반 점수 계산기"""
```

#### 주요 메서드

##### calculate_scores()
```python
def calculate_scores(self, poses: List[PersonPose]) -> Dict[str, float]
```

포즈 기반으로 상호작용 점수를 계산합니다.

---

## 시각화 API

### InferenceVisualizer

추론 결과 시각화 클래스입니다.

#### 클래스 정의
```python
class InferenceVisualizer:
    """추론 결과 시각화기"""
```

#### 주요 메서드

##### draw_poses()
```python
def draw_poses(self, frame: np.ndarray, poses: List[PersonPose]) -> np.ndarray
```

프레임에 포즈를 그립니다.

##### draw_classification_results()  
```python
def draw_classification_results(self, frame: np.ndarray, 
                               classification_results: List[Dict[str, Any]]) -> np.ndarray
```

분류 결과를 시각화합니다.

### PoseVisualizer

포즈 전용 시각화 클래스입니다.

#### 클래스 정의
```python
class PoseVisualizer:
    """포즈 시각화기"""
```

---

## 유틸리티 API

### ModuleFactory

모든 모듈의 팩토리 클래스입니다.

#### 클래스 정의
```python
class ModuleFactory:
    """모듈 팩토리"""
```

#### 주요 메서드

##### register_pose_estimator()
```python
@classmethod
def register_pose_estimator(cls, name: str, estimator_class: Type, 
                           default_config: Dict[str, Any])
```

포즈 추정기를 등록합니다.

##### register_classifier()
```python
@classmethod  
def register_classifier(cls, name: str, classifier_class: Type,
                       default_config: Dict[str, Any])
```

분류기를 등록합니다.

##### create_pose_estimator()
```python
@classmethod
def create_pose_estimator(cls, name: str, config: Dict[str, Any]) -> BasePoseEstimator
```

포즈 추정기를 생성합니다.

### WindowProcessor

윈도우 처리 유틸리티입니다.

#### SlidingWindowProcessor
```python
class SlidingWindowProcessor:
    """슬라이딩 윈도우 처리기"""
```

#### 주요 메서드

##### add_frame()
```python
def add_frame(self, frame_poses: FramePoses) -> List[WindowAnnotation]
```

프레임을 추가하고 완성된 윈도우를 반환합니다.

---

## 데이터 구조

### PersonPose
```python
@dataclass
class PersonPose:
    """개별 인물의 포즈 데이터"""
    track_id: int
    keypoints: np.ndarray      # shape: (17, 3) - [x, y, confidence]
    bbox: Optional[np.ndarray]  # [x1, y1, x2, y2]
    confidence: float
```

### FramePoses
```python
@dataclass
class FramePoses:
    """단일 프레임의 모든 포즈"""
    frame_idx: int
    timestamp: float
    persons: List[PersonPose]
    frame_shape: Tuple[int, int, int]  # (H, W, C)
```

### WindowAnnotation
```python
@dataclass
class WindowAnnotation:
    """윈도우 어노테이션 데이터"""
    window_idx: int
    window_id: str
    keypoint: np.ndarray        # (M, T, V, C) - MMAction2 표준
    keypoint_score: np.ndarray  # (M, T, V)
    total_frames: int
    label: Optional[int] = None
    frame_data: Optional[List[FramePoses]] = None
```

### ClassificationResult
```python
@dataclass
class ClassificationResult:
    """분류 결과"""
    prediction: int             # 예측 클래스 (0: NonFight, 1: Fight)
    confidence: float           # 신뢰도 (0.0-1.0)
    probabilities: List[float]  # 클래스별 확률 [nonfight_prob, fight_prob]
    model_name: str            # 사용된 모델명
    metadata: Optional[Dict[str, Any]] = None
```

### ActionClassificationConfig
```python
@dataclass
class ActionClassificationConfig:
    """행동 분류 설정"""
    model_name: str
    checkpoint_path: str
    config_file: Optional[str] = None
    device: str = 'cuda:0'
    window_size: int = 100
    confidence_threshold: float = 0.5
    class_names: Optional[List[str]] = None
    max_persons: int = 4
    input_format: str = 'stgcn'
    coordinate_dimensions: int = 3
    expected_keypoint_count: int = 17
```

---

## 설정 구조

### 전체 설정 구조
```yaml
models:
  pose_estimation:
    inference_mode: str        # 'pth', 'onnx', 'tensorrt'
    model_name: str           # 'rtmo_s', 'rtmo_m', 'rtmo_l'  
    device: str               # 'cuda:0', 'cpu'
    score_threshold: float    # 0.0-1.0
    
  action_classification:
    model_name: str           # 'stgcn', 'stgcn_onnx'
    checkpoint_path: str      # 모델 파일 경로
    window_size: int          # 윈도우 크기 (기본: 100)
    confidence_threshold: float
    max_persons: int
    
  tracking:
    model_name: str           # 'bytetrack'
    track_thresh: float
    min_box_area: float
    
  scoring:
    model_name: str           # 'region_based', 'movement_based'
    distance_threshold: float

inference:
  analysis:
    input: str               # 파일 또는 폴더 경로
    output_dir: str
    evaluation:
      enabled: bool
      ground_truth_dir: str
      
  realtime:
    input: Union[int, str]   # 웹캠 번호 또는 파일 경로
    display_window_size: List[int]
    fps_limit: int
    
annotation:
  input: str
  output_dir: str
  multi_process:
    enabled: bool
    num_processes: int
    gpus: List[int]
```

---

## 예외 처리

### 커스텀 예외 클래스

#### ConfigurationError
```python
class ConfigurationError(Exception):
    """설정 오류 예외"""
```

#### ModuleInitializationError
```python
class ModuleInitializationError(Exception):
    """모듈 초기화 오류 예외"""
```

#### InferenceError
```python
class InferenceError(Exception):
    """추론 오류 예외"""
```

---

## 사용 예시

### 기본 사용법
```python
from core import ModeManager
from utils.config_loader import load_config

# 설정 로드
config = load_config('config.yaml')

# 모드 관리자 생성
manager = ModeManager(config)

# 실시간 모드 실행
success = manager.execute('inference.realtime')
```

### 커스텀 모델 등록
```python
from utils.factory import ModuleFactory

# 커스텀 분류기 등록
ModuleFactory.register_classifier(
    name='custom_classifier',
    classifier_class=MyCustomClassifier,
    default_config={'param1': 'value1'}
)
```

### 직접 API 사용
```python
from action_classification.stgcn.stgcn_onnx_classifier import STGCNONNXClassifier
from utils.data_structure import ActionClassificationConfig

# 설정 생성
config = ActionClassificationConfig(
    model_name='stgcn_onnx',
    checkpoint_path='/path/to/model.onnx',
    window_size=100,
    confidence_threshold=0.4
)

# 분류기 생성 및 초기화
classifier = STGCNONNXClassifier(config)
classifier.initialize_model()

# 윈도우 분류
result = classifier.classify_single_window(window_data)
print(f"Classification: {result.prediction}, Confidence: {result.confidence}")
```

---

## 성능 최적화

### ONNX 모델 사용
- **메모리 사용량**: PyTorch 대비 30-50% 감소
- **추론 속도**: 2-3배 향상
- **Temperature Scaling**: 자동 적용으로 정확한 확률값

### 멀티프로세스 처리
- **프로세스 수**: CPU 코어 수와 동일 권장
- **GPU 분산**: 라운드 로빈 자동 할당
- **메모리 효율성**: 프로세스별 독립 메모리

### 실시간 처리 최적화
- **FPS 제한**: 시스템 성능에 맞게 조정
- **프레임 스킵**: 성능 향상을 위한 프레임 건너뛰기
- **배치 처리**: 여러 윈도우 동시 처리

---

*이 API 레퍼런스는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.*