# 실시간 추론 시스템 아키텍처

## 개요

Violence Detection 시스템의 `inference.realtime` 모드의 전체 아키텍처와 로직 흐름을 상세히 설명한다. 비디오/RTSP 입력부터 이벤트 관리 및 최종 결과 출력까지의 완전한 처리 과정을 다룬다.

## 목차

1. [시스템 개요](#시스템-개요)
2. [전체 아키텍처](#전체-아키텍처)
3. [핵심 컴포넌트](#핵심-컴포넌트)
4. [데이터 플로우](#데이터-플로우)
5. [주요 클래스 상세](#주요-클래스-상세)
6. [API 참조](#api-참조)
7. [성능 최적화](#성능-최적화)
8. [이벤트 관리](#이벤트-관리)

---

## 시스템 개요

### 목적

실시간 비디오 스트림에서 폭력 행동을 탐지하고, 관련 이벤트를 관리하여 즉각적인 알림과 로깅을 제공하는 시스템이다.

### 🔧 기술 스택

- **컴퓨터 비전**: OpenCV, RTMO (포즈 추정)
- **딥러닝**: PyTorch, ONNX, TensorRT
- **추론 엔진**: ST-GCN++ (행동 분류)
- **추적**: ByteTracker
- **이벤트 관리**: 커스텀 EventManager
- **시각화**: OpenCV GUI

### 성능 지표

- **실시간 처리**: 30-75 FPS (GPU 환경)
- **지연 시간**: 100-200ms (분류 포함)
- **정확도**: 90%+ (RWF-2000+ 데이터셋 기준)

---

## 전체 아키텍처

```mermaid
graph TD
    A[입력 소스] --> B[RealtimeInputManager]
    B --> C[InferencePipeline]
    C --> D[RTMO 포즈 추정]
    D --> E[ByteTracker 추적]
    E --> F[RegionBasedScorer 점수화]
    F --> G[SlidingWindowProcessor 윈도우 처리]
    G --> H[STGCN 행동 분류]
    H --> I[EventManager 이벤트 관리]
    I --> J[RealtimeVisualizer 시각화]
    I --> K[EventLogger 로깅]
  
    subgraph 입력
        A1[비디오 파일]
        A2[RTSP 스트림]
        A3[웹캠]
    end
  
    subgraph 출력
        J1[실시간 화면 표시]
        K1[JSON/CSV 로그]
        K2[이벤트 알림]
    end
  
    A1 --> A
    A2 --> A
    A3 --> A
  
    J --> J1
    K --> K1
    K --> K2
```

---

## 핵심 컴포넌트

### 1. 입력 처리 계층

#### RealtimeInputManager

- **역할**: 다양한 입력 소스 통합 관리
- **지원 형식**: 비디오 파일, RTSP/RTMP 스트림, 웹캠
- **특징**: 비동기 프레임 캡처, 버퍼링, FPS 제어

### 2. 추론 처리 계층

#### InferencePipeline

- **역할**: 전체 추론 파이프라인 오케스트레이션
- **특징**: 모듈식 구조, 성능 추적, 비동기 처리

#### RTMO (Real-Time Multi-Person Pose Estimation)

- **역할**: 실시간 다중 인체 포즈 추정
- **지원 모드**: PyTorch, ONNX, TensorRT
- **출력**: 17개 키포인트 좌표 + 신뢰도

#### ByteTracker

- **역할**: 다중 객체 추적
- **특징**: ID 일관성 유지, 누락/재등장 처리

#### ST-GCN++ (Spatial-Temporal Graph Convolutional Networks)

- **역할**: 골격 기반 행동 분류
- **입력**: 시계열 포즈 시퀀스 (100 프레임)
- **출력**: Fight/NonFight 확률

### 3. 이벤트 관리 계층

#### EventManager

- **역할**: 폭력 이벤트 생명주기 관리
- **기능**: 발생/지속/해제 로직, 임계값 기반 판단

#### EventLogger

- **역할**: 이벤트 로깅 및 저장
- **지원 형식**: JSON, CSV

### 4. 시각화 계층

#### RealtimeVisualizer

- **역할**: 실시간 결과 시각화
- **기능**: 포즈 표시, 분류 결과, 이벤트 상태, 성능 지표

---

## 데이터 플로우

### 메인 처리 루프

```python
while video_playing:
    # 1. 프레임 캡처
    frame = input_manager.get_frame()
  
    # 2. 포즈 추정
    poses = pose_estimator.estimate(frame)
  
    # 3. 다중 객체 추적
    tracked_poses = tracker.track(poses)
  
    # 4. 점수화 및 필터링
    scored_poses = scorer.score(tracked_poses)
  
    # 5. 윈도우 버퍼 업데이트
    window_processor.add_frame(scored_poses)
  
    # 6. 윈도우 준비 시 분류 수행 (비동기)
    if window_processor.is_ready():
        window_data = window_processor.get_window()
        classification_queue.put(window_data)
  
    # 7. 분류 결과 처리
    if classification_result_available:
        result = get_classification_result()
        event_data = event_manager.process_result(result)
      
        # 8. 시각화 업데이트
        visualizer.update(frame, poses, result, event_data)
  
    # 9. 화면 표시
    visualizer.show()
```

### 데이터 구조 변환

```
원본 프레임 (H×W×3)
    ↓
포즈 데이터 (N×17×3) [N명, 17키포인트, (x,y,conf)]
    ↓  
추적 데이터 (N×17×3 + track_id)
    ↓
점수화 데이터 (M×17×3 + score) [M≤N, 필터링됨]
    ↓
윈도우 데이터 (100×M×17×2) [100프레임, M명, 17키포인트, (x,y)]
    ↓
분류 결과 (Fight확률, NonFight확률)
    ↓
이벤트 데이터 (event_type, confidence, timestamp)
```

---

## 주요 클래스 상세

### InferencePipeline

```python
class InferencePipeline(BasePipeline):
    """실시간 추론 파이프라인 메인 클래스"""
  
    def __init__(self, config: Dict[str, Any]):
        """
        파이프라인 초기화
      
        Args:
            config: 통합 설정 딕셔너리
        """
  
    def initialize_pipeline(self) -> bool:
        """모든 모듈 초기화 및 설정"""
  
    def run_realtime_mode(self, input_source: str) -> bool:
        """실시간 모드 실행"""
  
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[FramePoses, Dict]:
        """단일 프레임 처리"""
  
    def _classification_worker(self):
        """비동기 분류 처리 워커"""
  
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
```

#### 주요 메서드 시그니처

##### initialize_pipeline()

```python
def initialize_pipeline(self) -> bool:
    """
    파이프라인 모듈 초기화
  
    Returns:
        bool: 초기화 성공 여부
  
    초기화 순서:
    1. 포즈 추정기 (RTMO)
    2. 추적기 (ByteTracker) 
    3. 점수화기 (RegionBasedScorer)
    4. 분류기 (STGCN)
    5. 윈도우 프로세서
    6. 이벤트 관리자
    """
```

##### process_frame()

```python
def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[FramePoses, Dict]:
    """
    단일 프레임 처리
  
    Args:
        frame: 입력 프레임 (H×W×3)
        frame_idx: 프레임 인덱스
  
    Returns:
        Tuple[FramePoses, Dict]: (처리된 포즈 데이터, 오버레이 정보)
  
    처리 단계:
    1. 포즈 추정: frame → poses
    2. 추적: poses → tracked_poses  
    3. 점수화: tracked_poses → scored_poses
    4. 윈도우 업데이트: scored_poses → window_buffer
    5. 오버레이 정보 생성
    """
```

### EventManager

```python
class EventManager:
    """이벤트 관리 시스템"""
  
    def __init__(self, config: EventConfig):
        """이벤트 관리자 초기화"""
  
    def process_classification_result(self, result: Dict[str, Any]) -> Optional[EventData]:
        """분류 결과 처리 및 이벤트 생성"""
  
    def add_event_callback(self, event_type: EventType, callback: Callable):
        """이벤트 콜백 등록"""
  
    def get_current_status(self) -> Dict[str, Any]:
        """현재 이벤트 상태 반환"""
```

#### 이벤트 처리 로직

##### process_classification_result()

```python
def process_classification_result(self, result: Dict[str, Any]) -> Optional[EventData]:
    """
    분류 결과를 바탕으로 이벤트 처리
  
    Args:
        result: {
            'window_id': int,
            'prediction': str,  # 'violence' or 'normal'
            'confidence': float,
            'timestamp': float,
            'probabilities': List[float]
        }
  
    Returns:
        Optional[EventData]: 생성된 이벤트 (없으면 None)
  
    처리 로직:
    1. 임계값 검증 (alert_threshold: 0.7)
    2. 연속 탐지 카운터 업데이트
    3. 이벤트 상태 전환 판단
    4. 이벤트 생성 및 로깅
    5. 콜백 함수 호출
    """
```

### RealtimeInputManager

```python
class RealtimeInputManager:
    """실시간 입력 관리자"""
  
    def __init__(self, input_source: Union[str, int], buffer_size: int = 10, 
                 target_fps: Optional[int] = None, frame_skip: int = 0):
        """입력 관리자 초기화"""
  
    def start(self) -> bool:
        """입력 스트림 시작"""
  
    def get_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """다음 프레임 가져오기"""
  
    def stop(self):
        """입력 스트림 중지"""
  
    def get_video_info(self) -> Dict[str, Any]:
        """비디오 정보 반환"""
```

##### 지원하는 입력 형식

```python
# 비디오 파일
input_source = "/path/to/video.mp4"

# RTSP 스트림  
input_source = "rtsp://camera_ip:554/stream"

# 웹캠
input_source = 0  # 또는 "/dev/video0"

# RTMP 스트림
input_source = "rtmp://server/live/stream"
```

### RealtimeVisualizer

```python
class RealtimeVisualizer:
    """실시간 시각화 클래스"""
  
    def __init__(self, window_name: str = "Violence Detection",
                 display_width: int = 1280, display_height: int = 720,
                 fps_limit: int = 30, confidence_threshold: float = 0.4):
        """시각화 초기화"""
  
    def show_frame(self, frame: np.ndarray, poses: Optional[FramePoses] = None,
                   classification: Optional[Dict[str, Any]] = None,
                   additional_info: Optional[Dict[str, Any]] = None,
                   overlay_data: Optional[Dict[str, Any]] = None) -> bool:
        """프레임 표시"""
  
    def update_event_history(self, event_data: Dict[str, Any]):
        """이벤트 히스토리 업데이트"""
  
    def update_classification_history(self, classification: Dict[str, Any]):
        """분류 결과 히스토리 업데이트"""
```

---

## API 참조

### 설정 구조

#### 메인 설정 (config.yaml)

```yaml
mode: inference.realtime

models:
  pose_estimation:
    inference_mode: onnx  # pth, onnx, tensorrt
    onnx:
      model_path: /path/to/rtmo.onnx
      device: cuda:0
      score_threshold: 0.3
      input_size: [640, 640]
  
  action_classification:
    model_name: stgcn
    checkpoint_path: /path/to/stgcn.pth
    confidence_threshold: 0.4  # 시각화 임계값
    window_size: 100
    device: cuda:0

  tracking:
    tracker_name: bytetrack
    track_thresh: 0.4
    track_buffer: 50
    match_thresh: 0.8

events:
  alert_threshold: 0.7        # 이벤트 발생 임계값
  normal_threshold: 0.5       # 이벤트 해제 임계값
  min_consecutive_detections: 3
  min_consecutive_normal: 5
  min_event_duration: 2.0
  max_event_duration: 300.0
  cooldown_duration: 10.0
  save_event_log: true
  event_log_format: json
  event_log_path: output/event_logs

files:
  video:
    input_source: /path/to/video.mp4  # 또는 RTSP URL
    output_path: output/result_video.mp4
    save_output: false

performance:
  target_fps: 30
  max_queue_size: 200
  processing_mode: realtime
```

#### EventConfig 클래스

```python
@dataclass
class EventConfig:
    """이벤트 관리 설정"""
    alert_threshold: float = 0.7          # 폭력 탐지 신뢰도 임계값
    min_consecutive_detections: int = 3   # 연속 탐지 최소 횟수
    normal_threshold: float = 0.5         # 정상 상태 신뢰도 임계값
    min_consecutive_normal: int = 5       # 연속 정상 최소 횟수
    min_event_duration: float = 2.0       # 최소 이벤트 지속 시간 (초)
    max_event_duration: float = 300.0     # 최대 이벤트 지속 시간 (초)
    cooldown_duration: float = 10.0       # 이벤트 쿨다운 시간 (초)
    enable_ongoing_alerts: bool = True    # 진행 중 알림 활성화
    ongoing_alert_interval: float = 30.0  # 진행 중 알림 간격 (초)
    save_event_log: bool = True          # 이벤트 로그 저장 여부
    event_log_format: str = "json"       # 로그 형식 (json/csv)
    event_log_path: str = "output/event_logs"  # 로그 저장 경로
```

### 데이터 구조

#### FramePoses

```python
@dataclass
class FramePoses:
    """프레임 내 모든 포즈 데이터"""
    persons: List[PersonPose]
    frame_idx: int
    timestamp: float
    video_info: Dict[str, Any]
```

#### PersonPose

```python
@dataclass
class PersonPose:
    """개별 인체 포즈 데이터"""
    keypoints: np.ndarray    # (17, 3) [x, y, confidence]
    bbox: Optional[np.ndarray]  # [x1, y1, x2, y2]
    track_id: Optional[int]
    score: float
    detection_confidence: float
```

#### EventData

```python
@dataclass  
class EventData:
    """이벤트 데이터"""
    event_type: EventType
    timestamp: float
    window_id: int
    confidence: float
    duration: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None
  
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
```

#### EventType

```python
class EventType(Enum):
    """이벤트 타입"""
    VIOLENCE_START = "violence_start"    # 폭력 시작
    VIOLENCE_END = "violence_end"        # 폭력 종료  
    VIOLENCE_ONGOING = "violence_ongoing" # 폭력 진행 중
    NORMAL = "normal"                    # 정상 상태
```

### 성능 지표

#### PerformanceTracker

```python
class PerformanceTracker:
    """성능 추적 클래스"""
  
    def get_stage_fps(self) -> Dict[str, float]:
        """단계별 FPS 반환"""
        return {
            'pose_estimation': float,
            'tracking': float, 
            'scoring': float,
            'classification': float,
            'overall': float
        }
  
    def get_processing_time_stats(self) -> Dict[str, float]:
        """처리 시간 통계"""
        return {
            'avg_processing_time': float,
            'max_processing_time': float,
            'min_processing_time': float
        }
```

---

## 성능 최적화

### 최적화 전략

#### 1. GPU 메모리 최적화

```python
# ONNX/TensorRT 사용으로 메모리 효율성 증대
pose_estimation:
  inference_mode: onnx  # 또는 tensorrt
  
# 배치 처리 최적화
batch_size: 1  # 실시간 처리용
```

#### 2. 비동기 처리

```python
# 분류 작업을 별도 스레드에서 비동기 처리
classification_queue = Queue(maxsize=10)
classification_thread = threading.Thread(target=self._classification_worker)
```

#### 3. 프레임 스킵핑

```python
# 고FPS 입력에서 선택적 프레임 처리
frame_skip: 2  # 2프레임마다 1프레임 처리
```

#### 4. 윈도우 슬라이딩 최적화

```python
# 효율적인 윈도우 관리
window_size: 100
window_stride: 50  # 50% 겹침으로 부드러운 분류
```

### 성능 모니터링

#### 실시간 FPS 추적

- **포즈 추정 FPS**: RTMO 추론 속도
- **추적 FPS**: ByteTracker 처리 속도
- **분류 FPS**: ST-GCN++ 추론 속도
- **전체 FPS**: 엔드투엔드 처리 속도

#### 메모리 사용량 모니터링

- **GPU 메모리**: 모델 로딩 + 추론 버퍼
- **CPU 메모리**: 프레임 버퍼 + 윈도우 데이터
- **큐 사용률**: 분류 큐 점유율

---

## 이벤트 관리

### 이벤트 생명주기

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> Violence_Detected : confidence ≥ 0.7 (3회 연속)
    Violence_Detected --> Violence_Ongoing : 지속 중
    Violence_Ongoing --> Violence_Ongoing : 지속 중 (30초마다 알림)
    Violence_Ongoing --> Normal : Normal confidence ≥ 0.5 (5회 연속)
    Violence_Detected --> Normal : Normal confidence ≥ 0.5 (5회 연속)
    Normal --> [*]
```

### ️ 임계값 설정

#### confidence_threshold (0.4)

- **용도**: 시각화 색상 결정
- **로직**: Fight/NonFight 확률이 0.4 이상이고 상대방보다 높으면 해당 색상 표시
- **영향**: 화면 표시 색상만 결정

#### alert_threshold (0.7)

- **용도**: 이벤트 발생 판단
- **로직**: Violence 예측 확률이 0.7 이상일 때 이벤트 시작
- **영향**: 실제 알림 및 로깅

#### normal_threshold (0.5)

- **용도**: 이벤트 해제 판단
- **로직**: Normal 예측 확률이 0.5 이상일 때 이벤트 종료
- **영향**: 이벤트 종료 및 복구

### 이벤트 로깅

#### JSON 형식

```json
{
  "event_id": "evt_20241220_143022_001",
  "event_type": "violence_start",
  "timestamp": 1703058622.123,
  "window_id": 15,
  "confidence": 0.847,
  "duration": null,
  "session_id": "session_20241220_143000",
  "additional_info": {
    "probabilities": [0.153, 0.847],
    "frame_number": 1500
  }
}
```

#### CSV 형식

```csv
timestamp,event_type,window_id,confidence,duration,session_id
1703058622.123,violence_start,15,0.847,,session_20241220_143000
1703058634.456,violence_end,23,0.523,12.333,session_20241220_143000
```

### 이벤트 콜백

```python
# 이벤트 콜백 등록 예시
def on_violence_start(event_data: EventData):
    """폭력 시작 시 호출"""
    print(f"[ALERT] Violence detected! Confidence: {event_data.confidence:.3f}")
    # 알림 전송, 녹화 시작 등

def on_violence_end(event_data: EventData):
    """폭력 종료 시 호출"""
    print(f"[INFO] Violence ended. Duration: {event_data.duration:.1f}s")
    # 녹화 중지, 보고서 생성 등

# 콜백 등록
event_manager.add_event_callback(EventType.VIOLENCE_START, on_violence_start)
event_manager.add_event_callback(EventType.VIOLENCE_END, on_violence_end)
```

---

## 실행 가이드

### 기본 실행

```bash
# 기본 설정으로 실행
python main.py --mode inference.realtime

# 특정 설정 파일 사용
python main.py --config custom_config.yaml --mode inference.realtime

# 디버그 모드
python main.py --mode inference.realtime --log-level DEBUG
```

### 고급 설정

#### RTSP 스트림 처리

```yaml
files:
  video:
    input_source: "rtsp://admin:password@192.168.1.100:554/stream1"
    buffer_size: 10
    target_fps: 25
```

#### TensorRT 가속

```yaml
models:
  pose_estimation:
    inference_mode: tensorrt
    tensorrt:
      model_path: /path/to/rtmo.engine
      device: cuda:0
```

#### 이벤트 알림 강화

```yaml
events:
  alert_threshold: 0.6        # 더 민감한 탐지
  min_consecutive_detections: 2  # 더 빠른 반응
  enable_ongoing_alerts: true
  ongoing_alert_interval: 15.0   # 15초마다 알림
```

---

## 문제 해결

### 일반적인 문제

#### 1. GPU 메모리 부족

```python
# 해결책: 배치 크기 줄이기, 모델 경량화
models:
  pose_estimation:
    inference_mode: onnx  # PyTorch → ONNX로 변경
```

#### 2. 실시간 처리 지연

```python
# 해결책: 프레임 스킵, 해상도 조정
performance:
  frame_skip: 1
  input_resolution: [640, 640]  # 더 작은 해상도
```

#### 3. 이벤트 과민 반응

```python
# 해결책: 임계값 조정
events:
  alert_threshold: 0.8           # 더 높은 임계값
  min_consecutive_detections: 5  # 더 많은 연속 탐지 필요
```

### 성능 튜닝

#### CPU vs GPU 밸런싱

- **포즈 추정**: GPU 중심 (ONNX/TensorRT)
- **추적**: CPU 효율적
- **분류**: GPU 가속 필요
- **시각화**: CPU 처리

#### 메모리 최적화

- **윈도우 버퍼**: 최소 크기 유지
- **프레임 큐**: 적절한 버퍼 사이즈
- **결과 캐시**: 주기적 정리

---

## 확장성

### 모듈 확장

#### 새로운 포즈 추정기 추가

```python
from pose_estimation.base import BasePoseEstimator

class CustomPoseEstimator(BasePoseEstimator):
    def estimate(self, frame: np.ndarray) -> FramePoses:
        # 커스텀 구현
        pass

# 팩토리에 등록
ModuleFactory.register_pose_estimator(
    name='custom_pose',
    estimator_class=CustomPoseEstimator,
    default_config={'param': 'value'}
)
```

#### 새로운 이벤트 타입 추가

```python
class EventType(Enum):
    VIOLENCE_START = "violence_start"
    VIOLENCE_END = "violence_end"
    VIOLENCE_ONGOING = "violence_ongoing"
    NORMAL = "normal"
    CUSTOM_EVENT = "custom_event"  # 새로운 이벤트 타입
```

### 분산 처리

#### 멀티 카메라 지원

```python
# 여러 입력 소스 동시 처리
cameras = [
    "rtsp://camera1/stream",
    "rtsp://camera2/stream", 
    "rtsp://camera3/stream"
]

for camera_id, source in enumerate(cameras):
    pipeline = InferencePipeline(config)
    pipeline.run_realtime_mode(source)
```

---

## 결론

본 실시간 추론 시스템은 모듈식 아키텍처와 이벤트 중심 설계를 통해 확장 가능하고 유지보수가 용이한 폭력 탐지 솔루션을 제공한다. 각 컴포넌트는 독립적으로 최적화될 수 있으며, 다양한 입력 소스와 배포 환경을 지원한다.

### 주요 장점

- **실시간 성능**: 30-75 FPS 처리 능력
- **모듈식 구조**: 각 컴포넌트 독립적 교체 가능
- **이벤트 기반**: 실시간 알림 및 로깅
- **다중 백엔드**: PyTorch, ONNX, TensorRT 지원
- **확장성**: 새로운 모델 및 기능 쉽게 추가

### 향후 개선 방향

- **엣지 디바이스**: Jetson, 모바일 디바이스 지원
- **클라우드 연동**: AWS, Azure 클라우드 서비스 통합
- **고급 분석**: 행동 패턴 분석, 예측 모델링
- **다중 모달**: 오디오, 텍스트 등 추가 모달리티 지원

---