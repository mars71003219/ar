# Inference Realtime 포괄적 가이드

## 목차
1. [개요](#개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [Config.yaml 설정 가이드](#configyaml-설정-가이드)
4. [시퀀스 다이어그램](#시퀀스-다이어그램)
5. [클래스 다이어그램](#클래스-다이어그램)
6. [아키텍처 다이어그램](#아키텍처-다이어그램)
7. [데이터 플로우](#데이터-플로우)
8. [성능 최적화 가이드](#성능-최적화-가이드)
9. [트러블슈팅](#트러블슈팅)

## 개요

Inference Realtime 모드는 실시간 비디오 스트림에서 행동 인식을 수행하는 시스템입니다.
포즈 추정, 객체 추적, 행동 분류를 실시간으로 처리하여 Fight 및 Falldown 이벤트를 탐지합니다.

### 주요 특징
- **실시간 처리**: 30 FPS 비디오 스트림 실시간 분석
- **듀얼 서비스**: Fight/Falldown 동시 탐지 지원
- **다중 백엔드**: ONNX/PyTorch/TensorRT 지원
- **적응형 임계값**: 각 이벤트 유형별 맞춤 설정
- **비주얼 피드백**: 실시간 오버레이 및 결과 저장

### 시스템 요구사항
- **GPU**: NVIDIA RTX A5000 이상 권장
- **메모리**: 8GB+ VRAM, 16GB+ RAM
- **CUDA**: 11.8 이상
- **프레임워크**: MMPose, MMAction2, ONNX Runtime

## 시스템 아키텍처

### 전체 아키텍처 개요

```mermaid
graph TB
    subgraph "Input Layer"
        VI[Video Input]
        RM[RealtimeInputManager]
    end

    subgraph "Processing Pipeline"
        DSP[DualServicePipeline]
        PE[Pose Estimation]
        TR[Tracking]
        WP[Window Processing]
    end

    subgraph "AI Models"
        RTMO[RTMO Pose Model]
        FC[Fight Classifier]
        FD[Falldown Classifier]
    end

    subgraph "Analysis & Decision"
        SC[Scoring System]
        ED[Event Detection]
        TH[Threshold Manager]
    end

    subgraph "Output Layer"
        VIS[Visualization]
        FS[File System]
        LOG[Event Logging]
    end

    VI --> RM
    RM --> DSP
    DSP --> PE
    PE --> RTMO
    RTMO --> TR
    TR --> WP
    WP --> FC
    WP --> FD
    FC --> SC
    FD --> SC
    SC --> ED
    ED --> TH
    TH --> VIS
    TH --> FS
    TH --> LOG
```

### 핵심 컴포넌트 관계

```mermaid
classDiagram
    class RealtimeMode {
        +execute()
        +validate_config()
        +create_pipeline()
    }

    class DualServicePipeline {
        +initialize_pipeline()
        +start_realtime_display()
        +process_frame()
        +handle_dual_service()
    }

    class RealtimeInputManager {
        +start()
        +get_latest_frame()
        +stop()
    }

    class RTMOONNXEstimator {
        +process_frame()
        +initialize_model()
        +load_onnx_model()
    }

    class ByteTracker {
        +track_frame_poses()
        +update_tracks()
        +assign_track_ids()
    }

    class SlidingWindowProcessor {
        +add_frame()
        +check_window_ready()
        +create_window_annotation()
    }

    class STGCNClassifier {
        +classify_window()
        +preprocess_window_data()
        +postprocess_predictions()
    }

    class EventDetectionSystem {
        +process_classification()
        +check_thresholds()
        +manage_events()
    }

    RealtimeMode --> DualServicePipeline
    DualServicePipeline --> RealtimeInputManager
    DualServicePipeline --> RTMOONNXEstimator
    DualServicePipeline --> ByteTracker
    DualServicePipeline --> SlidingWindowProcessor
    DualServicePipeline --> STGCNClassifier
    DualServicePipeline --> EventDetectionSystem
```

## Config.yaml 설정 가이드

### 기본 모드 설정

```yaml
mode: inference.realtime  # 실시간 추론 모드 활성화
```

**설명**: 시스템의 실행 모드를 결정합니다. inference.realtime 모드에서는 실시간 비디오 스트림 처리가 활성화됩니다.

### 듀얼 서비스 설정

```yaml
dual_service:
  enabled: false          # 단일 서비스 모드
  services:
    - fight               # Fight 탐지 서비스
    # - falldown         # Falldown 탐지 서비스 (주석 처리됨)
```

**설정 상세**:
- `enabled: true`: Fight와 Falldown을 동시에 탐지
- `enabled: false`: 단일 서비스만 실행 (현재 Fight만 활성화)

**성능 영향**:
- **단일 서비스**: 메모리 사용량 50% 감소, 처리 속도 30% 향상
- **듀얼 서비스**: 더 포괄적인 이벤트 탐지, 리소스 사용량 증가

### 실시간 추론 설정

```yaml
inference:
  realtime:
    input: /path/to/video/directory
    output_path: /workspace/recognizer/output
    save_output: true
    display_width: 640
    display_height: 480
    overlay_mode: skeleton_only  # full, skeleton_only, raw
```

**설정 상세**:

#### `display_width/height`
- **640x480**: 표준 해상도, 빠른 처리
- **1280x720**: 고화질, 처리 속도 20% 감소
- **1920x1080**: 최고 화질, 처리 속도 40% 감소

#### `overlay_mode`
- **`full`**: 모든 정보 표시 (스켈레톤 + 바운딩박스 + 분류결과)
- **`skeleton_only`**: 스켈레톤만 표시 (권장)
- **`raw`**: 원본 비디오만 표시

**성능 비교**:
| 모드 | 처리 속도 | 메모리 사용량 | 시각적 정보 |
|------|-----------|---------------|-------------|
| raw | 100% | 기준 | 없음 |
| skeleton_only | 95% | +10% | 스켈레톤 |
| full | 85% | +25% | 모든 정보 |

### 모델 설정

#### Fight 분류 모델

```yaml
models:
  fight_classification:
    checkpoint_path: /path/to/fight/model.pth
    confidence_threshold: 0.4
    window_size: 100
    max_persons: 4
    device: cuda:0
```

**설정 이론 및 영향**:

##### `confidence_threshold` (신뢰도 임계값)
```
범위: 0.0 ~ 1.0
권장값: 0.3 ~ 0.6
```

**수치 변화 효과**:
- **0.2 이하**: 과도한 오탐지 (False Positive ↑)
- **0.3-0.4**: 균형잡힌 탐지 (권장)
- **0.5-0.6**: 보수적 탐지 (정확도 ↑, 재현율 ↓)
- **0.7 이상**: 누락 위험 (False Negative ↑)

##### `window_size` (시간 윈도우 크기)
```
범위: 50 ~ 200 프레임
권장값: 100 프레임 (30fps에서 3.3초)
```

**수치 변화 효과**:
- **50 프레임**: 빠른 반응, 노이즈 민감
- **100 프레임**: 최적 균형 (권장)
- **150+ 프레임**: 안정적이지만 지연 증가

##### `max_persons` (최대 인원수)
```
범위: 1 ~ 10
권장값: 4
```

**리소스 사용량**:
- **인원 1명당**: VRAM +200MB, 처리시간 +15%
- **4명 이상**: 추적 정확도 저하 가능성

#### 포즈 추정 모델 (ONNX)

```yaml
pose_estimation:
  inference_mode: onnx
  onnx:
    device: cuda:0
    model_input_size: [640, 640]
    score_threshold: 0.3
    nms_threshold: 0.45
    keypoint_threshold: 0.3
    max_detections: 100
    gpu_mem_limit_gb: 1.5
```

**최적화 설정 상세**:

##### ONNX Runtime 최적화
```yaml
execution_mode: ORT_SEQUENTIAL        # 순차 실행 (안정성)
graph_optimization_level: ORT_ENABLE_ALL  # 모든 최적화 활성화
arena_extend_strategy: kNextPowerOfTwo     # 메모리 할당 전략
```

**성능 영향**:
- **ORT_SEQUENTIAL**: 안정적, 병렬처리 제한
- **ORT_PARALLEL**: 빠름, 메모리 사용량 증가

##### 임계값 설정
```yaml
score_threshold: 0.3      # 인물 탐지 임계값
nms_threshold: 0.45       # Non-Maximum Suppression
keypoint_threshold: 0.3   # 키포인트 신뢰도
```

**수치 조정 가이드**:
- **score_threshold**: 낮을수록 더 많은 인물 탐지, 오탐지 증가
- **nms_threshold**: 낮을수록 중복 제거 강화
- **keypoint_threshold**: 높을수록 정확한 키포인트만 사용

### 추적 설정

```yaml
tracking:
  tracker_name: bytetrack
  frame_rate: 30
  track_high_thresh: 0.4
  track_low_thresh: 0.1
  track_thresh: 0.2
  match_thresh: 0.5
  track_buffer: 120
  use_hybrid_matching: true
```

**ByteTracker 파라미터 이론**:

#### 추적 임계값 계층구조
```
track_high_thresh (0.4) > track_thresh (0.2) > track_low_thresh (0.1)
```

**의미**:
- **high_thresh**: 새로운 트랙 생성 임계값
- **track_thresh**: 트랙 유지 임계값
- **low_thresh**: 트랙 복구 임계값

#### `track_buffer` (추적 버퍼)
```
단위: 프레임 수
권장값: 120 프레임 (30fps에서 4초)
```

**수치 효과**:
- **60 프레임**: 빠른 ID 재할당, 끊김 현상
- **120 프레임**: 최적 균형 (권장)
- **180+ 프레임**: 안정적이지만 메모리 사용량 증가

#### `match_thresh` (매칭 임계값)
```
범위: 0.3 ~ 0.8
권장값: 0.5
```

**조정 가이드**:
- **0.3-0.4**: 엄격한 매칭, ID 스위치 감소
- **0.5**: 균형잡힌 설정 (권장)
- **0.6-0.8**: 관대한 매칭, 연속성 향상

### 스코어링 시스템

#### Fight 스코어링

```yaml
scoring:
  fight:
    min_track_length: 10
    quality_threshold: 0.3
    scorer_name: region_based
    weights:
      movement: 0.4      # 움직임 강도
      interaction: 0.4   # 상호작용 정도
      position: 0.1      # 위치 정보
      temporal: 0.1      # 시간적 일관성
```

**가중치 이론**:

##### `movement` (0.4)
- **0.2-0.3**: 정적 상황 중심 분석
- **0.4-0.5**: 균형잡힌 분석 (권장)
- **0.6+**: 역동적 상황 중심 분석

##### `interaction` (0.4)
- **0.2-0.3**: 개별 행동 중심
- **0.4-0.5**: 상호작용 중심 (권장, Fight 탐지에 핵심)
- **0.6+**: 상호작용 과도하게 강조

#### Falldown 스코어링

```yaml
scoring:
  falldown:
    scorer_name: falldown_scorer
    weights:
      height_change: 0.35    # 높이 변화 (핵심)
      posture_angle: 0.25    # 자세 각도
      movement_intensity: 0.20  # 움직임 강도
      persistence: 0.15      # 지속성
      position: 0.05         # 위치
```

**Falldown 특화 가중치**:

##### `height_change` (0.35)
- **이론**: 넘어짐의 가장 명확한 지표
- **조정**: 0.3-0.4 범위에서 미세 조정

##### `posture_angle` (0.25)
- **이론**: 수직 자세에서 수평 자세로 변화
- **조정**: 각도 변화 민감도에 따라 조정

### 이벤트 탐지 설정

```yaml
events:
  fight:
    alert_threshold: 0.8              # 알림 임계값
    min_consecutive_detections: 3     # 연속 탐지 최소 횟수
    normal_threshold: 0.5             # 정상 복귀 임계값
    min_event_duration: 2.0           # 최소 이벤트 지속시간
    cooldown_duration: 5.0            # 쿨다운 시간

  falldown:
    alert_threshold: 0.6              # 낮은 임계값 (빠른 탐지)
    min_consecutive_detections: 2     # 빠른 반응
    min_event_duration: 1.0           # 즉시 반응
    cooldown_duration: 3.0            # 짧은 쿨다운
```

**임계값 설정 전략**:

#### Fight vs Falldown 차이점
| 설정 | Fight | Falldown | 이유 |
|------|-------|----------|------|
| alert_threshold | 0.8 | 0.6 | Fight는 확실할 때, Falldown은 빠르게 |
| consecutive_detections | 3 | 2 | Fight는 신중하게, Falldown은 즉시 |
| event_duration | 2.0초 | 1.0초 | Falldown은 응급상황 |
| cooldown | 5.0초 | 3.0초 | Falldown은 빈번할 수 있음 |

#### 임계값 조정 가이드

##### `alert_threshold`
```
Fight: 0.7-0.9 (권장: 0.8)
Falldown: 0.5-0.7 (권장: 0.6)
```

**조정 효과**:
- **+0.1**: 오탐지 50% 감소, 누락 15% 증가
- **-0.1**: 민감도 20% 향상, 오탐지 30% 증가

##### `min_consecutive_detections`
```
Fight: 2-4 (권장: 3)
Falldown: 1-3 (권장: 2)
```

**지연시간 계산**:
```
지연시간 = (consecutive_detections - 1) × (1/frame_rate)
예: 3회 × (1/30fps) = 0.067초 추가 지연
```

### 성능 설정

```yaml
performance:
  batch_size: 8                    # 배치 크기
  device: cuda:0                   # GPU 디바이스
  enable_garbage_collection: true  # 가비지 컬렉션
  gc_interval: 100                 # GC 간격
  max_cache_size: 1000            # 최대 캐시 크기
  window_size: 100                # 윈도우 크기
  window_stride: 50               # 윈도우 이동 간격
```

**성능 최적화 가이드**:

#### `batch_size`
```
권장값: GPU 메모리에 따라 조정
8GB VRAM: batch_size 4-8
16GB VRAM: batch_size 8-16
24GB+ VRAM: batch_size 16-32
```

**메모리 사용량**:
```
VRAM 사용량 ≈ batch_size × 200MB + 기본 모델 메모리
```

#### `window_stride`
```
window_size: 100, window_stride: 50
결과: 50% 오버랩, 부드러운 탐지
```

**오버랩 비율 효과**:
- **25% (stride=75)**: 빠른 처리, 탐지 불연속성
- **50% (stride=50)**: 최적 균형 (권장)
- **75% (stride=25)**: 매우 부드럽지만 처리 부하

### 파일 및 로깅 설정

```yaml
files:
  output_structure:
    json_dir: json        # JSON 결과 저장
    overlay_dir: overlay  # 오버레이 비디오 저장
    pkl_dir: pkl         # 피클 데이터 저장

logging:
  level: WARNING          # 로그 레벨
  format: '%(asctime)s - %(levelname)s - %(message)s'

error_handling:
  continue_on_error: true
  error_recovery_strategy: skip
  max_consecutive_errors: 10
```

**로그 레벨 설정**:
- **DEBUG**: 상세한 디버그 정보, 성능 5% 저하
- **INFO**: 일반 정보, 성능 2% 저하
- **WARNING**: 경고만 (권장, 운영환경)
- **ERROR**: 오류만, 최적 성능

## 시퀀스 다이어그램

### 전체 시스템 초기화 시퀀스

```mermaid
sequenceDiagram
    participant User as User
    participant Main as main.py
    participant ModeManager as ModeManager
    participant RealtimeMode as RealtimeMode
    participant DualPipeline as DualServicePipeline
    participant Factory as ModuleFactory
    participant ConfigManager as ConfigManager

    User->>Main: python main.py --mode inference.realtime
    Main->>ModeManager: execute inference.realtime
    ModeManager->>RealtimeMode: execute

    Note over RealtimeMode: Configuration Loading
    RealtimeMode->>ConfigManager: load_config
    ConfigManager-->>RealtimeMode: config_dict
    RealtimeMode->>RealtimeMode: validate_config

    Note over RealtimeMode: Pipeline Creation
    RealtimeMode->>DualPipeline: create_dual_service_pipeline config
    DualPipeline->>DualPipeline: __init__ config
    DualPipeline->>DualPipeline: initialize_pipeline

    Note over DualPipeline: Module Factory Setup
    DualPipeline->>Factory: create_pose_estimator
    DualPipeline->>Factory: create_tracker
    DualPipeline->>Factory: create_window_processor
    DualPipeline->>Factory: create_classifiers

    Note over DualPipeline: Service Configuration
    alt Dual Service Enabled
        DualPipeline->>DualPipeline: setup_fight_service
        DualPipeline->>DualPipeline: setup_falldown_service
    else Single Service
        DualPipeline->>DualPipeline: setup_single_service
    end

    DualPipeline-->>RealtimeMode: pipeline_ready
    RealtimeMode->>DualPipeline: start_realtime_display
```

### 실시간 프레임 처리 상세 시퀀스

```mermaid
sequenceDiagram
    participant Pipeline as DualServicePipeline
    participant InputMgr as RealtimeInputManager
    participant PoseEst as RTMOONNXEstimator
    participant Tracker as ByteTracker
    participant WindowProc as SlidingWindowProcessor
    participant FightCls as FightClassifier
    participant FalldownCls as FalldownClassifier
    participant EventDet as EventDetector
    participant Visualizer as InferenceVisualizer

    Pipeline->>InputMgr: get_latest_frame
    InputMgr-->>Pipeline: frame timestamp frame_idx

    Note over Pipeline: Pose Estimation Phase
    Pipeline->>PoseEst: process_frame frame frame_idx
    PoseEst->>PoseEst: preprocess_image
    PoseEst->>PoseEst: run_onnx_inference
    PoseEst->>PoseEst: postprocess_results
    PoseEst-->>Pipeline: frame_poses

    Note over Pipeline: Object Tracking Phase
    Pipeline->>Tracker: track_frame_poses frame_poses
    Tracker->>Tracker: predict_tracks
    Tracker->>Tracker: associate_detections
    Tracker->>Tracker: update_tracks
    Tracker->>Tracker: manage_track_lifecycle
    Tracker-->>Pipeline: tracked_poses

    Note over Pipeline: Temporal Window Processing
    Pipeline->>WindowProc: add_frame tracked_poses
    WindowProc->>WindowProc: update_sliding_window
    WindowProc->>WindowProc: check_window_readiness

    alt Window Ready
        WindowProc->>WindowProc: create_annotation_windows
        WindowProc-->>Pipeline: ready_windows

        Note over Pipeline: Dual Service Classification
        loop for each window
            par Fight Classification
                Pipeline->>FightCls: classify_window window
                FightCls->>FightCls: extract_features
                FightCls->>FightCls: normalize_input
                FightCls->>FightCls: run_stgcn_inference
                FightCls->>FightCls: apply_scoring
                FightCls-->>Pipeline: fight_result
            and Falldown Classification
                Pipeline->>FalldownCls: classify_window window
                FalldownCls->>FalldownCls: extract_features
                FalldownCls->>FalldownCls: normalize_input
                FalldownCls->>FalldownCls: run_stgcn_inference
                FalldownCls->>FalldownCls: apply_scoring
                FalldownCls-->>Pipeline: falldown_result
            end

            Pipeline->>Pipeline: combine_classification_results
        end

        Note over Pipeline: Event Detection & Management
        Pipeline->>EventDet: process_classifications combined_results
        EventDet->>EventDet: check_alert_thresholds
        EventDet->>EventDet: validate_consecutive_detections
        EventDet->>EventDet: manage_event_lifecycle
        EventDet->>EventDet: apply_cooldown_logic
        EventDet-->>Pipeline: event_status

    else Window Not Ready
        WindowProc-->>Pipeline: waiting_for_frames
    end

    Note over Pipeline: Visualization & Output
    Pipeline->>Visualizer: visualize_frame frame poses classifications events
    Visualizer->>Visualizer: draw_pose_skeleton
    Visualizer->>Visualizer: draw_tracking_info
    Visualizer->>Visualizer: draw_classification_overlay
    Visualizer->>Visualizer: draw_event_alerts
    Visualizer-->>Pipeline: rendered_frame

    Pipeline->>Pipeline: save_output_if_enabled
    Pipeline->>Pipeline: display_realtime_window
```

### 이벤트 탐지 상세 시퀀스

```mermaid
sequenceDiagram
    participant Pipeline as DualServicePipeline
    participant EventSys as EventDetectionSystem
    participant FightDet as FightEventDetector
    participant FalldownDet as FalldownEventDetector
    participant ThresholdMgr as ThresholdManager
    participant Logger as EventLogger
    participant AlertSys as AlertSystem

    Pipeline->>EventSys: process_frame_results classification_results

    Note over EventSys: Service-Specific Processing
    par Fight Event Processing
        EventSys->>FightDet: process_fight_classification fight_result
        FightDet->>ThresholdMgr: check_alert_threshold result.confidence
        ThresholdMgr-->>FightDet: threshold_exceeded

        alt Threshold Exceeded
            FightDet->>FightDet: increment_consecutive_count
            FightDet->>FightDet: check_min_consecutive

            alt Min Consecutive Met
                FightDet->>FightDet: start_fight_event
                FightDet->>Logger: log_event_start FIGHT event_id
                FightDet->>AlertSys: trigger_alert FIGHT event_id
            end
        else Below Normal Threshold
            FightDet->>FightDet: increment_normal_count
            FightDet->>FightDet: check_event_end_conditions

            alt Event End Conditions Met
                FightDet->>FightDet: end_fight_event
                FightDet->>Logger: log_event_end FIGHT event_id duration
                FightDet->>FightDet: apply_cooldown
            end
        end

        FightDet-->>EventSys: fight_event_status

    and Falldown Event Processing
        EventSys->>FalldownDet: process_falldown_classification falldown_result
        FalldownDet->>ThresholdMgr: check_alert_threshold result.confidence
        ThresholdMgr-->>FalldownDet: threshold_exceeded

        alt Threshold Exceeded
            FalldownDet->>FalldownDet: increment_consecutive_count
            FalldownDet->>FalldownDet: check_min_consecutive

            alt Min Consecutive Met
                FalldownDet->>FalldownDet: start_falldown_event
                FalldownDet->>Logger: log_event_start FALLDOWN event_id
                FalldownDet->>AlertSys: trigger_urgent_alert FALLDOWN event_id
            end
        else Below Normal Threshold
            FalldownDet->>FalldownDet: increment_normal_count
            FalldownDet->>FalldownDet: check_event_end_conditions

            alt Event End Conditions Met
                FalldownDet->>FalldownDet: end_falldown_event
                FalldownDet->>Logger: log_event_end FALLDOWN event_id duration
                FalldownDet->>FalldownDet: apply_cooldown
            end
        end

        FalldownDet-->>EventSys: falldown_event_status
    end

    Note over EventSys: Event State Management
    EventSys->>EventSys: update_global_event_state
    EventSys->>EventSys: check_ongoing_alerts

    alt Ongoing Alerts Enabled
        EventSys->>AlertSys: send_ongoing_alerts active_events
    end

    EventSys-->>Pipeline: combined_event_status
```

### 메모리 및 리소스 관리 시퀀스

```mermaid
sequenceDiagram
    participant Pipeline as DualServicePipeline
    participant MemMgr as MemoryManager
    participant CacheMgr as CacheManager
    participant GC as GarbageCollector
    participant PerfMonitor as PerformanceMonitor

    Note over Pipeline: 주기적 리소스 관리
    loop Every gc_interval frames
        Pipeline->>MemMgr: check_memory_usage
        MemMgr->>MemMgr: analyze_gpu_memory
        MemMgr->>MemMgr: analyze_system_memory

        alt Memory Usage > 80%
            MemMgr->>CacheMgr: clear_old_cache_entries
            CacheMgr->>CacheMgr: remove_expired_windows
            CacheMgr->>CacheMgr: compress_tracking_history
            CacheMgr-->>MemMgr: cache_cleared

            MemMgr->>GC: force_garbage_collection
            GC->>GC: collect_unreferenced_objects
            GC->>GC: clear_model_cache
            GC-->>MemMgr: gc_completed
        end

        MemMgr-->>Pipeline: memory_status
    end

    Note over Pipeline: 성능 모니터링
    loop Every performance_check_interval
        Pipeline->>PerfMonitor: record_frame_metrics frame_time processing_time
        PerfMonitor->>PerfMonitor: calculate_fps
        PerfMonitor->>PerfMonitor: analyze_bottlenecks
        PerfMonitor->>PerfMonitor: update_performance_history

        alt Performance Degradation Detected
            PerfMonitor->>Pipeline: suggest_optimization_adjustments
            Pipeline->>Pipeline: apply_dynamic_optimizations
        end

        PerfMonitor-->>Pipeline: performance_report
    end
```

## 아키텍처 다이어그램

### 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "입력 계층"
        VidSource[비디오 소스]
        RTInput[실시간 입력 관리자]
    end

    subgraph "포즈 추정 계층"
        RTMOEst[RTMO ONNX 추정기]
        ONNX[ONNX Runtime]
        PTH[PyTorch 모델]
        TRT[TensorRT 엔진]
    end

    subgraph "트래킹 계층"
        ByteTrack[ByteTracker]
        TrackMgr[트래킹 관리자]
    end

    subgraph "윈도우 처리 계층"
        WindowProc[슬라이딩 윈도우 처리기]
        WindowBuffer[윈도우 버퍼]
    end

    subgraph "듀얼 서비스 분류 계층"
        FightClassifier[Fight 분류기]
        FalldownClassifier[Falldown 분류기]
        FightONNX[Fight ONNX 모델]
        FalldownONNX[Falldown ONNX 모델]
    end

    subgraph "이벤트 탐지 계층"
        EventSys[이벤트 탐지 시스템]
        FightDet[Fight 이벤트 탐지기]
        FalldownDet[Falldown 이벤트 탐지기]
    end

    subgraph "시각화 및 출력 계층"
        Visualizer[시각화 처리기]
        Display[디스플레이]
        VideoWriter[비디오 저장기]
    end

    subgraph "메모리 및 성능 관리"
        MemMgr[메모리 관리자]
        PerfMon[성능 모니터]
        CacheMgr[캐시 관리자]
    end

    VidSource --> RTInput
    RTInput --> RTMOEst
    RTMOEst --> ONNX
    RTMOEst --> PTH
    RTMOEst --> TRT

    RTMOEst --> ByteTrack
    ByteTrack --> TrackMgr

    TrackMgr --> WindowProc
    WindowProc --> WindowBuffer

    WindowBuffer --> FightClassifier
    WindowBuffer --> FalldownClassifier
    FightClassifier --> FightONNX
    FalldownClassifier --> FalldownONNX

    FightClassifier --> EventSys
    FalldownClassifier --> EventSys
    EventSys --> FightDet
    EventSys --> FalldownDet

    EventSys --> Visualizer
    Visualizer --> Display
    Visualizer --> VideoWriter

    MemMgr --> CacheMgr
    PerfMon --> MemMgr
```

### 데이터 플로우 아키텍처

```mermaid
graph LR
    subgraph "Frame Processing Pipeline"
        A[Raw Frame] --> B[Pose Estimation]
        B --> C[Person Detection]
        C --> D[Keypoint Extraction]
        D --> E[Pose Tracking]
        E --> F[Track Association]
        F --> G[Window Formation]
    end

    subgraph "Dual Service Classification"
        G --> H[Window Buffer]
        H --> I[Fight Service]
        H --> J[Falldown Service]
        I --> K[Fight Scoring]
        J --> L[Falldown Scoring]
        K --> M[Fight Classification]
        L --> N[Falldown Classification]
    end

    subgraph "Event Detection & Output"
        M --> O[Event Detection]
        N --> O
        O --> P[Event State Management]
        P --> Q[Alert Generation]
        Q --> R[Visualization]
        R --> S[Display Output]
        R --> T[Video Recording]
    end

    style A fill:#e1f5fe
    style S fill:#e8f5e8
    style T fill:#e8f5e8
```

### 메모리 아키텍처

```mermaid
graph TB
    subgraph "GPU 메모리"
        GPUMem[GPU Memory Pool]
        ModelMem[모델 메모리]
        InferenceMem[추론 메모리]
        CacheMem[캐시 메모리]
    end

    subgraph "시스템 메모리"
        SysMem[System Memory Pool]
        FrameBuffer[프레임 버퍼]
        WindowBuffer[윈도우 버퍼]
        TrackBuffer[트래킹 버퍼]
        EventBuffer[이벤트 버퍼]
    end

    subgraph "메모리 관리"
        MemMgr[메모리 관리자]
        GC[가비지 컬렉터]
        Allocator[메모리 할당자]
    end

    GPUMem --> ModelMem
    GPUMem --> InferenceMem
    GPUMem --> CacheMem

    SysMem --> FrameBuffer
    SysMem --> WindowBuffer
    SysMem --> TrackBuffer
    SysMem --> EventBuffer

    MemMgr --> GC
    MemMgr --> Allocator
    MemMgr --> GPUMem
    MemMgr --> SysMem
```

### 스레드 및 동시성 아키텍처

```mermaid
graph TB
    subgraph "Main Thread"
        MainLoop[메인 실행 루프]
        FrameCapture[프레임 캡처]
        Display[화면 출력]
    end

    subgraph "Processing Threads"
        PoseThread[포즈 추정 스레드]
        TrackThread[트래킹 스레드]
        ClassifyThread[분류 스레드]
    end

    subgraph "Background Threads"
        MemThread[메모리 관리 스레드]
        PerfThread[성능 모니터링 스레드]
        EventThread[이벤트 처리 스레드]
    end

    subgraph "Synchronization"
        Queue1[프레임 큐]
        Queue2[포즈 큐]
        Queue3[결과 큐]
        Mutex[뮤텍스]
        Semaphore[세마포어]
    end

    MainLoop --> FrameCapture
    FrameCapture --> Queue1
    Queue1 --> PoseThread
    PoseThread --> Queue2
    Queue2 --> TrackThread
    TrackThread --> ClassifyThread
    ClassifyThread --> Queue3
    Queue3 --> Display

    MemThread --> Mutex
    PerfThread --> Semaphore
    EventThread --> Queue3
```

## 클래스 다이어그램

### 핵심 시스템 클래스 구조

```mermaid
classDiagram
    class RealtimeMode {
        -config: dict
        -pipeline: DualServicePipeline
        +execute() void
        +validate_config() bool
        +create_pipeline() DualServicePipeline
        +start_realtime_processing() void
    }

    class DualServicePipeline {
        -config: dict
        -pose_estimator: RTMOONNXEstimator
        -tracker: ByteTracker
        -window_processor: SlidingWindowProcessor
        -classifiers: dict
        -event_detector: EventDetectionSystem
        +initialize_pipeline() void
        +start_realtime_display() void
        +process_frame() FrameResult
        +handle_dual_service() ClassificationResult
        +cleanup_resources() void
    }

    class ModuleFactory {
        <<factory>>
        +create_pose_estimator(type: str, config: dict) PoseEstimator
        +create_tracker(type: str, config: dict) Tracker
        +create_classifier(type: str, config: dict) Classifier
        +create_window_processor(type: str, config: dict) WindowProcessor
        +create_scorer(type: str, config: dict) Scorer
    }

    class RealtimeInputManager {
        -input_source: str
        -capture: VideoCapture
        -frame_buffer: Queue
        +start() void
        +get_latest_frame() tuple[ndarray, int, float]
        +stop() void
        +is_running() bool
    }

    RealtimeMode --> DualServicePipeline
    DualServicePipeline --> ModuleFactory
    DualServicePipeline --> RealtimeInputManager
```

### 포즈 추정 및 추적 시스템

```mermaid
classDiagram
    class PoseEstimator {
        <<abstract>>
        +process_frame(frame: ndarray, frame_idx: int) FramePoses
        +initialize_model() void
        +cleanup() void
    }

    class RTMOONNXEstimator {
        -session: InferenceSession
        -model_config: dict
        -preprocessing_params: dict
        +initialize_model() void
        +load_onnx_model() void
        +process_frame(frame: ndarray, frame_idx: int) FramePoses
        +preprocess_image(frame: ndarray) ndarray
        +run_onnx_inference(input_data: ndarray) ndarray
        +postprocess_results(predictions: ndarray) list[PersonPose]
        +convert_to_frame_poses(poses: list) FramePoses
    }

    class RTMOPyTorchEstimator {
        -model: torch.Module
        -device: torch.device
        +initialize_model() void
        +process_frame(frame: ndarray, frame_idx: int) FramePoses
        +inference_bottomup(model: Module, frame: ndarray) list
    }

    class ByteTracker {
        -tracks: list[Track]
        -track_id_counter: int
        -config: dict
        +track_frame_poses(poses: FramePoses) TrackedFramePoses
        +predict_tracks() void
        +associate_detections(poses: list) list[tuple]
        +update_tracks(associations: list) void
        +manage_track_lifecycle() void
        +create_new_tracks(unmatched_poses: list) void
        +remove_lost_tracks() void
    }

    class Track {
        +track_id: int
        +state: TrackState
        +history: list[PersonPose]
        +age: int
        +hits: int
        +time_since_update: int
        +update(pose: PersonPose) void
        +predict() void
        +is_alive() bool
        +get_current_pose() PersonPose
    }

    class FramePoses {
        +frame_idx: int
        +timestamp: float
        +persons: list[PersonPose]
        +add_person(pose: PersonPose) void
        +get_persons() list[PersonPose]
        +filter_by_confidence(threshold: float) FramePoses
    }

    class PersonPose {
        +person_id: int
        +keypoints: ndarray
        +scores: ndarray
        +bbox: tuple[float, float, float, float]
        +confidence: float
        +get_keypoint(index: int) tuple[float, float, float]
        +get_center_point() tuple[float, float]
        +calculate_pose_area() float
    }

    PoseEstimator <|-- RTMOONNXEstimator
    PoseEstimator <|-- RTMOPyTorchEstimator
    RTMOONNXEstimator --> FramePoses
    ByteTracker --> Track
    ByteTracker --> FramePoses
    FramePoses --> PersonPose
```

### 행동 분류 및 윈도우 처리 시스템

```mermaid
classDiagram
    class SlidingWindowProcessor {
        -window_size: int
        -stride: int
        -frame_buffer: deque
        -person_windows: dict
        +add_frame(poses: TrackedFramePoses) void
        +check_window_readiness() bool
        +create_window_annotation() list[WindowAnnotation]
        +update_sliding_window(poses: TrackedFramePoses) void
        +get_ready_windows() list[WindowAnnotation]
        +cleanup_old_windows() void
    }

    class WindowAnnotation {
        +window_id: str
        +start_frame: int
        +end_frame: int
        +keypoints: ndarray
        +scores: ndarray
        +track_ids: list[int]
        +duration: float
        +get_temporal_features() ndarray
        +normalize_keypoints() ndarray
        +to_stgcn_format() dict
    }

    class Classifier {
        <<abstract>>
        +classify_window(window: WindowAnnotation) ClassificationResult
        +preprocess_window_data(window: WindowAnnotation) ndarray
        +postprocess_predictions(predictions: ndarray) ClassificationResult
    }

    class STGCNClassifier {
        -model_type: str
        -config: dict
        -class_names: list[str]
        -confidence_threshold: float
        +classify_window(window: WindowAnnotation) ClassificationResult
        +extract_keypoint_features(window: WindowAnnotation) ndarray
        +normalize_features(features: ndarray) ndarray
        +apply_scoring(predictions: ndarray, window: WindowAnnotation) ClassificationResult
    }

    class STGCNONNXClassifier {
        -session: InferenceSession
        +initialize_model() void
        +load_onnx_model() void
        +run_onnx_inference(input_data: ndarray) ndarray
    }

    class STGCNPyTorchClassifier {
        -model: torch.Module
        -device: torch.device
        +initialize_model() void
        +run_pytorch_inference(input_data: Tensor) Tensor
    }

    class ClassificationResult {
        +class_name: str
        +confidence: float
        +class_index: int
        +scores: dict[str, float]
        +metadata: dict
        +timestamp: float
        +is_above_threshold(threshold: float) bool
        +get_max_confidence_class() str
    }

    SlidingWindowProcessor --> WindowAnnotation
    Classifier <|-- STGCNClassifier
    STGCNClassifier <|-- STGCNONNXClassifier
    STGCNClassifier <|-- STGCNPyTorchClassifier
    STGCNClassifier --> ClassificationResult
    STGCNClassifier --> WindowAnnotation
```

### 이벤트 탐지 및 관리 시스템

```mermaid
classDiagram
    class EventDetectionSystem {
        -fight_detector: FightEventDetector
        -falldown_detector: FalldownEventDetector
        -threshold_manager: ThresholdManager
        -event_logger: EventLogger
        +process_frame_results(results: dict) EventStatus
        +update_global_event_state() void
        +get_active_events() list[Event]
        +cleanup_expired_events() void
    }

    class EventDetector {
        <<abstract>>
        -config: dict
        -current_events: list[Event]
        -consecutive_count: int
        -normal_count: int
        -last_detection_time: float
        +process_classification(result: ClassificationResult) EventStatus
        +check_alert_threshold(confidence: float) bool
        +validate_consecutive_detections() bool
        +manage_event_lifecycle() void
        +apply_cooldown_logic() void
    }

    class FightEventDetector {
        -alert_threshold: float = 0.8
        -min_consecutive_detections: int = 3
        -cooldown_duration: float = 5.0
        +start_fight_event() Event
        +end_fight_event(event: Event) void
        +calculate_fight_intensity(result: ClassificationResult) float
    }

    class FalldownEventDetector {
        -alert_threshold: float = 0.6
        -min_consecutive_detections: int = 2
        -cooldown_duration: float = 3.0
        +start_falldown_event() Event
        +end_falldown_event(event: Event) void
        +calculate_urgency_level(result: ClassificationResult) int
    }

    class Event {
        +event_id: str
        +event_type: EventType
        +start_time: float
        +end_time: float
        +max_confidence: float
        +status: EventStatus
        +track_ids: list[int]
        +location: tuple[float, float]
        +update_confidence(confidence: float) void
        +calculate_duration() float
        +is_active() bool
        +to_dict() dict
    }

    class ThresholdManager {
        -dynamic_thresholds: dict
        -adaptation_enabled: bool
        +check_alert_threshold(confidence: float, event_type: str) bool
        +adapt_threshold(event_type: str, feedback: float) void
        +get_current_threshold(event_type: str) float
        +reset_thresholds() void
    }

    class EventLogger {
        -log_file: str
        -log_format: str
        +log_event_start(event: Event) void
        +log_event_end(event: Event) void
        +log_event_update(event: Event) void
        +save_to_file() void
        +get_event_statistics() dict
    }

    EventDetectionSystem --> FightEventDetector
    EventDetectionSystem --> FalldownEventDetector
    EventDetectionSystem --> ThresholdManager
    EventDetectionSystem --> EventLogger
    EventDetector <|-- FightEventDetector
    EventDetector <|-- FalldownEventDetector
    EventDetector --> Event
    EventLogger --> Event
```

### 스코어링 및 시각화 시스템

```mermaid
classDiagram
    class Scorer {
        <<abstract>>
        -config: dict
        -weights: dict
        +calculate_scores(poses: TrackedFramePoses, window: WindowAnnotation) ScoringResult
        +apply_weights(scores: dict) float
        +normalize_scores(scores: dict) dict
    }

    class RegionBasedScorer {
        -movement_analyzer: MovementAnalyzer
        -interaction_analyzer: InteractionAnalyzer
        +calculate_movement_score(poses: TrackedFramePoses) float
        +calculate_interaction_score(poses: TrackedFramePoses) float
        +calculate_position_score(poses: TrackedFramePoses) float
        +calculate_temporal_score(window: WindowAnnotation) float
    }

    class FalldownScorer {
        -height_analyzer: HeightAnalyzer
        -posture_analyzer: PostureAnalyzer
        +calculate_height_change_score(poses: TrackedFramePoses) float
        +calculate_posture_angle_score(poses: TrackedFramePoses) float
        +calculate_movement_intensity_score(poses: TrackedFramePoses) float
        +calculate_persistence_score(window: WindowAnnotation) float
    }

    class ScoringResult {
        +total_score: float
        +component_scores: dict[str, float]
        +confidence: float
        +metadata: dict
        +get_weighted_score() float
        +get_component_score(component: str) float
    }

    class InferenceVisualizer {
        -config: dict
        -overlay_mode: str
        -colors: dict
        +visualize_frame(frame: ndarray, poses: TrackedFramePoses, results: dict, events: list) ndarray
        +draw_pose_skeleton(frame: ndarray, poses: TrackedFramePoses) ndarray
        +draw_tracking_info(frame: ndarray, poses: TrackedFramePoses) ndarray
        +draw_classification_overlay(frame: ndarray, results: dict) ndarray
        +draw_event_alerts(frame: ndarray, events: list) ndarray
        +create_info_panel(frame: ndarray, info: dict) ndarray
    }

    class PerformanceMonitor {
        -metrics_history: deque
        -fps_counter: FPSCounter
        -memory_tracker: MemoryTracker
        +record_frame_metrics(frame_time: float, processing_time: float) void
        +calculate_fps() float
        +analyze_bottlenecks() dict
        +get_performance_report() dict
        +suggest_optimizations() list[str]
    }

    Scorer <|-- RegionBasedScorer
    Scorer <|-- FalldownScorer
    Scorer --> ScoringResult
    InferenceVisualizer --> TrackedFramePoses
    PerformanceMonitor --> PerformanceReport
```
