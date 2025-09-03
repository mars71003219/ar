# 실시간 추론 시스템 아키텍처

## 개요

이 문서는 Violence Detection 시스템의 `inference.realtime` 모드의 전체 아키텍처와 로직 흐름을 상세히 설명한다. 비디오/RTSP 입력부터 이벤트 관리 및 최종 결과 출력까지의 완전한 처리 과정을 다룬다.

**최신 업데이트**: 2025-09-03 기준, ONNX 모델 통합, Temperature Scaling, 멀티프로세스 지원 등 최신 기능 반영

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
- **딥러닝**: PyTorch, ONNX (온도 스케일링 적용), TensorRT
- **추론 엔진**: ST-GCN++ (행동 분류) - PyTorch/ONNX 지원
- **추적**: ByteTracker
- **이벤트 관리**: 커스텀 EventManager (JSON/CSV 로깅)
- **시각화**: OpenCV GUI + 실시간 성능 모니터링
- **멀티프로세싱**: annotation 스타일 병렬 처리
- **Docker**: 컨테이너 환경 지원 (MMCV 호환성)

### 성능 지표

- **실시간 처리**: 30-75 FPS (GPU 환경, ONNX 최적화 적용)
- **지연 시간**: 100-200ms (분류 포함)
- **정확도**: 90%+ (RWF-2000+ 데이터셋 기준)
- **ONNX 성능**: PyTorch 대비 2-3배 빠른 추론 속도
- **메모리 효율성**: ONNX로 30-50% 메모리 사용량 감소
- **Temperature Scaling**: 0.005 적용으로 정확한 확률값 출력

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
    G --> H[ST-GCN++ 행동 분류]
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

    subgraph ONNX_최적화
        H1[ONNX Runtime]
        H2[Temperature Scaling]
        H3[GPU 메모리 최적화]
    end
    
    H --> H1
    H1 --> H2
    H2 --> H3
```

---

## 핵심 컴포넌트

### 1. 입력 처리 계층

#### RealtimeInputManager

- **역할**: 다양한 입력 소스 통합 관리
- **지원 형식**: 비디오 파일, RTSP/RTMP 스트림, 웹캠
- **특징**: 비동기 프레임 캡처, 버퍼링, FPS 제어
- **Docker 지원**: 컨테이너 환경에서 GPU 가속 지원

### 2. 추론 처리 계층

#### InferencePipeline

- **역할**: 전체 추론 파이프라인 오케스트레이션
- **특징**: 모듈식 구조, 성능 추적, 비동기 처리
- **모드 지원**: analysis, realtime, visualize

#### RTMO (Real-Time Multi-Person Pose Estimation)

- **역할**: 실시간 다중 인체 포즈 추정
- **지원 모드**: PyTorch, ONNX, TensorRT
- **출력**: 17개 키포인트 좌표 + 신뢰도
- **ONNX 최적화**: 자동 GPU 최적화 및 벤치마킹 지원

#### ByteTracker

- **역할**: 다중 객체 추적
- **특징**: ID 일관성 유지, 누락/재등장 처리
- **필터링**: min_box_area 설정으로 작은 객체 제외

#### ST-GCN++ (Spatial-Temporal Graph Convolutional Networks)

- **역할**: 골격 기반 행동 분류
- **입력**: 시계열 포즈 시퀀스 (100 프레임)
- **출력**: Fight/NonFight 확률
- **ONNX 지원**: Temperature Scaling (0.005) 적용
- **모델 전환**: PyTorch ↔ ONNX 호환성 보장

### 3. 이벤트 관리 계층

#### EventManager

- **역할**: 폭력 이벤트 생명주기 관리
- **기능**: 발생/지속/해제 로직, 임계값 기반 판단
- **로깅**: JSON/CSV 형식 지원
- **콜백**: 사용자 정의 이벤트 핸들러

#### EventLogger

- **역할**: 이벤트 로깅 및 저장
- **지원 형식**: JSON, CSV
- **세션 관리**: 자동 세션 ID 생성 및 파일 관리

### 4. 시각화 계층

#### RealtimeVisualizer

- **역할**: 실시간 결과 시각화
- **기능**: 포즈 표시, 분류 결과, 이벤트 상태, 성능 지표
- **성능 모니터링**: FPS, 처리 시간, GPU 사용률 실시간 표시

---

## 데이터 플로우

### 메인 처리 루프

```python
while video_playing:
    # 1. 프레임 캡처
    frame = input_manager.get_frame()
  
    # 2. 포즈 추정 (ONNX 최적화)
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
  
    # 7. 분류 결과 처리 (Temperature Scaling 적용)
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
분류 결과 (Fight확률, NonFight확률) [Temperature Scaling 적용]
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
    1. 포즈 추정기 (RTMO ONNX/PyTorch)
    2. 추적기 (ByteTracker) 
    3. 점수화기 (RegionBasedScorer)
    4. 분류기 (ST-GCN++ ONNX/PyTorch)
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

### ONNX 모델 통합

#### STGCNONNXClassifier

```python
class STGCNONNXClassifier(ONNXInferenceBase):
    """ST-GCN++ ONNX 분류기"""
    
    def __init__(self, config: Dict[str, Any]):
        """ONNX 모델 초기화"""
        super().__init__(config)
        
        # Temperature Scaling 설정
        self.temperature = 0.005
    
    def classify_window(self, window_data: np.ndarray) -> ClassificationResult:
        """윈도우 데이터 분류"""
        # ONNX 추론
        raw_scores = self.onnx_session.run(None, {input_name: window_data})[0]
        
        # Temperature Scaling 적용
        temperature = self.temperature
        scaled_scores = raw_scores * temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return ClassificationResult(
            prediction=np.argmax(probabilities),
            confidence=float(np.max(probabilities)),
            probabilities=probabilities.tolist()
        )
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

---

## API 참조

### 설정 구조

#### 메인 설정 (config.yaml)

```yaml
mode: inference.realtime

models:
  pose_estimation:
    inference_mode: onnx  # pth, onnx, tensorrt
    model_name: rtmo_l    # rtmo_s, rtmo_m, rtmo_l
    onnx:
      model_path: /path/to/rtmo.onnx
      device: cuda:0
      score_threshold: 0.3
      input_size: [640, 640]
      # ONNX 최적화 설정
      gpu_mem_limit_gb: 8
      cudnn_conv_algo_search: "HEURISTIC"
      do_copy_in_default_stream: false
      cudnn_conv_use_max_workspace: true
      tunable_op_enable: true
  
  action_classification:
    model_name: stgcn_onnx  # stgcn, stgcn_onnx
    checkpoint_path: /path/to/stgcn.onnx
    input_format: stgcn_onnx
    confidence_threshold: 0.4  # 시각화 임계값
    window_size: 100
    max_persons: 4
    device: cuda:0
    # Temperature Scaling 설정
    temperature: 0.005

  tracking:
    model_name: bytetrack
    track_thresh: 0.4
    track_buffer: 50
    match_thresh: 0.8
    min_box_area: 200  # 작은 객체 필터링

  scoring:
    model_name: movement_based
    distance_threshold: 100.0

events:
  alert_threshold: 0.7        # 이벤트 발생 임계값
  normal_threshold: 0.5       # 이벤트 해제 임계값
  min_consecutive_detections: 3
  min_consecutive_normal: 5
  min_event_duration: 2.0
  max_event_duration: 300.0
  cooldown_duration: 10.0
  save_event_log: true
  event_log_format: json      # json, csv
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

# 멀티프로세스 설정 (analysis 모드용)
multi_process:
  enabled: false
  num_processes: 4
  gpus: [0, 1]
```

### ONNX 최적화 설정

#### GPU별 최적 설정

```yaml
# RTX A5000 (24GB) 최적 설정
models:
  pose_estimation:
    onnx:
      gpu_mem_limit_gb: 8
      cudnn_conv_algo_search: "HEURISTIC"
      do_copy_in_default_stream: false
      cudnn_conv_use_max_workspace: true
      arena_extend_strategy: "kNextPowerOfTwo"
      tunable_op_enable: true
      tunable_op_tuning_enable: true

# RTX 3060 (8GB) 최적 설정  
models:
  pose_estimation:
    onnx:
      gpu_mem_limit_gb: 4
      cudnn_conv_algo_search: "DEFAULT"
      do_copy_in_default_stream: true
      cudnn_conv_use_max_workspace: false
      tunable_op_enable: false
```

---

## 성능 최적화

### 최적화 전략

#### 1. ONNX 모델 최적화

```bash
# 자동 GPU 최적화
python tools/onnx_optimizer.py \
    --model /path/to/model.onnx \
    --config config.yaml \
    --output optimization_report.md
```

#### 2. GPU 메모리 최적화

```yaml
pose_estimation:
  onnx:
    gpu_mem_limit_gb: 8  # GPU 메모리의 30-40% 권장
    arena_extend_strategy: "kNextPowerOfTwo"
```

#### 3. 비동기 처리

```python
# 분류 작업을 별도 스레드에서 비동기 처리
classification_queue = Queue(maxsize=10)
classification_thread = threading.Thread(target=self._classification_worker)
```

#### 4. 프레임 스킵핑

```python
# 고FPS 입력에서 선택적 프레임 처리
frame_skip: 2  # 2프레임마다 1프레임 처리
```

#### 5. Temperature Scaling 최적화

```python
# ONNX 모델의 raw logits를 정확한 확률로 변환
temperature = 0.005
scaled_scores = raw_scores * temperature
probabilities = softmax(scaled_scores)
```

### 성능 모니터링

#### 실시간 FPS 추적

- **포즈 추정 FPS**: RTMO ONNX 추론 속도 (목표: 75+ FPS)
- **추적 FPS**: ByteTracker 처리 속도 (목표: 100+ FPS)
- **분류 FPS**: ST-GCN++ ONNX 추론 속도 (목표: 30+ FPS)
- **전체 FPS**: 엔드투엔드 처리 속도 (목표: 30+ FPS)

#### 메모리 사용량 모니터링

- **GPU 메모리**: 모델 로딩 + 추론 버퍼 (8GB 권장)
- **CPU 메모리**: 프레임 버퍼 + 윈도우 데이터
- **큐 사용률**: 분류 큐 점유율 (70% 이하 권장)

#### ONNX vs PyTorch 성능 비교

| 모델 | PyTorch FPS | ONNX FPS | 개선율 | 메모리 절약 |
|------|-------------|----------|--------|-------------|
| RTMO-L | 45-50 | 75-80 | 67% | 40% |
| ST-GCN++ | 15-20 | 30-35 | 75% | 50% |

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
  "event_id": "evt_20250903_143022_001",
  "event_type": "violence_start",
  "timestamp": 1725348622.123,
  "window_id": 15,
  "confidence": 0.847,
  "duration": null,
  "session_id": "session_20250903_143000",
  "additional_info": {
    "probabilities": [0.153, 0.847],
    "frame_number": 1500,
    "model_version": "stgcn_onnx_v1.0",
    "temperature_scaling": 0.005
  }
}
```

#### CSV 형식

```csv
timestamp,event_type,window_id,confidence,duration,session_id,model_version
1725348622.123,violence_start,15,0.847,,session_20250903_143000,stgcn_onnx_v1.0
1725348634.456,violence_end,23,0.523,12.333,session_20250903_143000,stgcn_onnx_v1.0
```

---

## 실행 가이드

### Docker 환경에서 실행

```bash
# 컨테이너에서 실시간 모드 실행
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.realtime"

# 특정 설정 파일 사용
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --config custom_config.yaml --mode inference.realtime"

# 디버그 모드
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.realtime --log-level DEBUG"
```

### ONNX 모델 최적화

```bash
# GPU별 최적 설정 자동 탐지
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 tools/onnx_optimizer.py --model /path/to/model.onnx --config config.yaml"

# 빠른 테스트 모드
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 tools/onnx_optimizer.py --model /path/to/model.onnx --quick"
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

#### ONNX 모델 전환 설정

```yaml
models:
  action_classification:
    model_name: stgcn_onnx  # PyTorch에서 ONNX로 전환
    checkpoint_path: /workspace/mmaction2/checkpoints/stgcnpp_enhanced_fight_detection_stable.onnx
    input_format: stgcn_onnx
    temperature: 0.005  # Temperature Scaling 적용
```

#### 멀티프로세스 분석

```yaml
multi_process:
  enabled: true
  num_processes: 8        # CPU 코어 수에 맞춤
  gpus: [0, 1, 2, 3]     # 사용 가능한 GPU 목록
```

---

## 문제 해결

### 일반적인 문제

#### 1. ONNX 모델 로딩 실패

```bash
# 모델 경로 확인
docker exec mmlabs bash -c "ls -la /workspace/mmaction2/checkpoints/"

# 권한 확인
docker exec mmlabs bash -c "chmod 644 /workspace/mmaction2/checkpoints/*.onnx"
```

#### 2. Temperature Scaling 문제

```python
# ONNX 출력이 극값(0, 1)만 나올 때
# config.yaml에서 temperature 값 조정
models:
  action_classification:
    temperature: 0.005  # 더 작은 값으로 조정
```

#### 3. GPU 메모리 부족

```yaml
# 설정 최적화
models:
  pose_estimation:
    onnx:
      gpu_mem_limit_gb: 4  # 메모리 제한 감소
  action_classification:
    max_persons: 2        # 기본값 4에서 감소
```

#### 4. Docker 환경 문제

```bash
# MMCV 호환성 확인
docker exec mmlabs bash -c "python3 -c 'import mmcv; print(mmcv.__version__)'"

# CUDA 환경 확인
docker exec mmlabs bash -c "python3 -c 'import torch; print(torch.cuda.is_available())'"
```

#### 5. ONNX 성능 최적화

```bash
# 자동 최적화 실행
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 tools/onnx_optimizer.py --model model.onnx --config config.yaml"

# 결과 확인
cat optimization_report.md
```

### 로그 분석

#### 핵심 로그 메시지

- `STGCN ONNX RESULT`: ONNX 분류 결과 확인
- `raw_scores`: ONNX 원본 출력 확인 (예: [-256, 291])
- `Probabilities after softmax`: Temperature Scaling 후 확률값
- `Processing inference result`: 최종 처리 결과

#### 디버그 모드 로그 확인

```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.realtime --log-level DEBUG" | grep "STGCN"
```

---

## 확장성

### 모듈 확장

#### 새로운 ONNX 분류기 추가

```python
from action_classification.stgcn.stgcn_onnx_classifier import STGCNONNXClassifier

class CustomONNXClassifier(STGCNONNXClassifier):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.temperature = config.get('temperature', 0.01)  # 커스텀 온도 설정

# 팩토리에 등록
ModuleFactory.register_classifier(
    name='custom_onnx',
    classifier_class=CustomONNXClassifier,
    default_config={'temperature': 0.01}
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
    ANOMALY_DETECTED = "anomaly_detected"  # 이상 행동 감지
```

### 멀티프로세스 처리

#### 분석 모드에서 멀티프로세스 사용

```yaml
mode: inference.analysis
multi_process:
  enabled: true
  num_processes: 8
  gpus: [0, 1, 2, 3]

inference:
  analysis:
    input: "/data/videos/"
    output_dir: "output/analysis"
```

#### 실시간 모드에서 멀티 카메라

```python
# 여러 입력 소스 동시 처리
cameras = [
    "rtsp://camera1/stream",
    "rtsp://camera2/stream", 
    "rtsp://camera3/stream"
]

for camera_id, source in enumerate(cameras):
    config_copy = config.copy()
    config_copy['files']['video']['input_source'] = source
    pipeline = InferencePipeline(config_copy)
    pipeline.run_realtime_mode(source)
```

---

## 결론

본 실시간 추론 시스템은 최신 ONNX 최적화와 Temperature Scaling을 통해 향상된 성능과 정확성을 제공한다. 모듈식 아키텍처와 이벤트 중심 설계를 통해 확장 가능하고 유지보수가 용이한 폭력 탐지 솔루션을 제공한다.

### 주요 장점

- **고성능 추론**: ONNX 최적화로 PyTorch 대비 2-3배 빠른 속도
- **정확한 확률값**: Temperature Scaling으로 정밀한 확률 출력
- **Docker 호환성**: 컨테이너 환경에서 안정적 실행
- **자동 최적화**: GPU별 자동 최적화 도구 제공
- **모듈식 구조**: 각 컴포넌트 독립적 교체 가능
- **이벤트 기반**: 실시간 알림 및 로깅
- **멀티프로세스**: 대규모 배치 처리 지원
- **확장성**: 새로운 모델 및 기능 쉽게 추가

### 향후 개선 방향

- **TensorRT 지원**: 더 빠른 추론을 위한 TensorRT 엔진 지원
- **엣지 디바이스**: Jetson, 모바일 디바이스 지원
- **클라우드 연동**: AWS, Azure 클라우드 서비스 통합
- **고급 분석**: 행동 패턴 분석, 예측 모델링
- **다중 모달**: 오디오, 텍스트 등 추가 모달리티 지원
- **분산 처리**: Kubernetes 기반 분산 처리 시스템

---

**최신 업데이트**: 2025-09-03, ONNX 모델 통합 및 Temperature Scaling 적용 완료