# 실시간 추론 시스템 UML 다이어그램

## 개요

본 문서는 Violence Detection 실시간 추론 시스템의 UML 다이어그램을 제공한다. 클래스 다이어그램, 컴포넌트 다이어그램, 패키지 다이어그램을 통해 시스템의 구조적 관계를 시각화한다.
---

## 목차

- [실시간 추론 시스템 UML 다이어그램](#실시간-추론-시스템-uml-다이어그램)
  - [개요](#개요)
  - [목차](#목차)
  - [전체 시스템 클래스 다이어그램](#전체-시스템-클래스-다이어그램)
  - [파이프라인 클래스 다이어그램](#파이프라인-클래스-다이어그램)
  - [이벤트 관리 클래스 다이어그램](#이벤트-관리-클래스-다이어그램)
  - [데이터 구조 클래스 다이어그램](#데이터-구조-클래스-다이어그램)
  - [컴포넌트 다이어그램](#컴포넌트-다이어그램)
  - [패키지 다이어그램](#패키지-다이어그램)
  - [상속 계층 다이어그램](#상속-계층-다이어그램)
  - [인터페이스 다이어그램](#인터페이스-다이어그램)
  - [메서드 호출 관계 다이어그램](#메서드-호출-관계-다이어그램)
  - [데이터 플로우 클래스 다이어그램](#데이터-플로우-클래스-다이어그램)
  - [UML 다이어그램 요약](#uml-다이어그램-요약)
    - [주요 설계 패턴](#주요-설계-패턴)
    - [핵심 아키텍처 특징](#핵심-아키텍처-특징)
    - [확장 포인트](#확장-포인트)

---

## 전체 시스템 클래스 다이어그램

```mermaid
classDiagram
    class InferencePipeline {
        -BasePoseEstimator pose_estimator
        -BaseTracker tracker
        -BaseScorer scorer
        -BaseActionClassifier classifier
        -SlidingWindowProcessor window_processor
        -EventManager event_manager
        -List frame_buffer
        -Queue classification_queue
        -Thread classification_thread
        +__init__(config)
        +initialize_pipeline() bool
        +run_realtime_mode(input_source) bool
        +process_frame(frame, frame_idx) Tuple
        +get_performance_stats() Dict
        +_classification_worker() void
        +_start_classification_thread() void
        +_stop_classification_thread() void
    }

    class RealtimeInputManager {
        -Union input_source
        -int buffer_size
        -Optional target_fps
        -int frame_skip
        -VideoCapture cap
        -Queue frame_queue
        -Thread capture_thread
        -bool is_running
        +__init__(input_source, buffer_size, target_fps, frame_skip)
        +start() bool
        +stop() void
        +get_frame() Optional
        +get_video_info() Dict
        -_analyze_source_type(source) str
        -_capture_loop() void
    }

    class EventManager {
        -EventConfig config
        -bool current_event_active
        -int consecutive_violence
        -int consecutive_normal
        -List event_history
        -Dict event_callbacks
        -EventLogger logger
        +__init__(config)
        +process_classification_result(result) Optional
        +add_event_callback(event_type, callback) void
        +get_current_status() Dict
        +get_event_history(limit) List
        -_handle_violence_detection(result) Optional
        -_handle_normal_detection(result) Optional
        -_create_event_data(event_type, result) EventData
    }

    class RealtimeVisualizer {
        -str window_name
        -int display_width
        -int display_height
        -int fps_limit
        -float confidence_threshold
        -PoseVisualizer pose_visualizer
        -List classification_history
        -List event_history
        -bool is_running
        +__init__(window_name, display_width, display_height, fps_limit, confidence_threshold)
        +start_display() void
        +stop_display() void
        +show_frame(frame, poses, classification, additional_info, overlay_data) bool
        +update_event_history(event_data) void
        +update_classification_history(classification) void
        -_draw_event_history(frame) ndarray
        -draw_classification_results(frame) ndarray
        -add_overlay_info(frame, additional_info) ndarray
    }

    class BasePoseEstimator {
        <<abstract>>
        +estimate_poses(frame) FramePoses
        +set_score_threshold(threshold) void
        +get_model_info() Dict
        +warmup() void
    }

    class BaseTracker {
        <<abstract>>
        +track(poses) FramePoses
        +reset() void
        +get_active_tracks() List
        +update_tracks(detections) void
    }

    class BaseScorer {
        <<abstract>>
        +score_poses(poses) FramePoses
        +set_quality_threshold(threshold) void
        +get_scoring_criteria() Dict
        +filter_poses(poses) FramePoses
    }

    class BaseActionClassifier {
        <<abstract>>
        +classify_window(window_data) ClassificationResult
        +set_confidence_threshold(threshold) void
        +get_class_names() List
        +warmup() void
    }

    class SlidingWindowProcessor {
        -int window_size
        -int stride
        -List frames_buffer
        -int current_window_id
        +__init__(window_size, stride)
        +add_frame_data(frame_poses) void
        +is_ready() bool
        +get_window_data() ndarray
        +reset() void
        +get_buffer_size() int
    }

    class PoseVisualizer {
        -List keypoint_connections
        -Dict color_palette
        +draw_poses(frame, poses) ndarray
        +draw_keypoints(frame, keypoints) ndarray
        +draw_skeleton(frame, keypoints) ndarray
        +set_colors(colors) void
    }

    InferencePipeline --> RealtimeInputManager
    InferencePipeline --> EventManager
    InferencePipeline --> RealtimeVisualizer
    InferencePipeline --> BasePoseEstimator
    InferencePipeline --> BaseTracker
    InferencePipeline --> BaseScorer
    InferencePipeline --> BaseActionClassifier
    InferencePipeline --> SlidingWindowProcessor
    RealtimeVisualizer --> PoseVisualizer
```

---

## 파이프라인 클래스 다이어그램

```mermaid
classDiagram
    class BasePipeline {
        <<abstract>>
        #Dict config
        #bool _initialized
        +__init__(config)
        +initialize_pipeline()* bool
        +run()* bool
        +cleanup() void
    }

    class InferencePipeline {
        -str mode
        -BasePoseEstimator pose_estimator
        -BaseTracker tracker
        -BaseScorer scorer
        -BaseActionClassifier classifier
        -SlidingWindowProcessor window_processor
        -EventManager event_manager
        -PerformanceTracker performance_tracker
        -Dict stage_timings
        -List classification_results
        -Queue classification_queue
        -Thread classification_thread
        -bool classification_running
        +initialize_pipeline() bool
        +run_realtime_mode(input_source) bool
        +run_analysis_mode(input_source) bool
        +process_frame(frame, frame_idx) Tuple
        +get_performance_stats() Dict
        +get_stage_fps() Dict
        -_initialize_event_manager(config) void
        -_classification_worker() void
        -_start_classification_thread() void
        -_stop_classification_thread() void
        -_add_stage_timing(stage_name, timing) void
    }

    class ModuleFactory {
        <<singleton>>
        -Dict _pose_estimators
        -Dict _trackers
        -Dict _scorers
        -Dict _classifiers
        -Dict _window_processors
        +register_pose_estimator(name, estimator_class, default_config) void
        +register_tracker(name, tracker_class, default_config) void
        +register_scorer(name, scorer_class, default_config) void
        +register_classifier(name, classifier_class, default_config) void
        +register_window_processor(name, processor_class, default_config) void
        +create_pose_estimator(name, config) BasePoseEstimator
        +create_tracker(name, config) BaseTracker
        +create_scorer(name, config) BaseScorer
        +create_classifier(name, config) BaseActionClassifier
        +create_window_processor(name, config) SlidingWindowProcessor
        +list_registered_modules() Dict
    }

    class PerformanceTracker {
        -List frame_times
        -Dict stage_timings
        -float start_time
        -int frame_count
        +__init__()
        +start_timing(stage_name) void
        +end_timing(stage_name) void
        +add_frame_time(frame_time) void
        +get_overall_fps() float
        +get_stage_fps() Dict
        +get_avg_processing_time() float
        +reset() void
    }

    InferencePipeline --|> BasePipeline
    InferencePipeline --> ModuleFactory
    InferencePipeline --> PerformanceTracker
```

---

## 이벤트 관리 클래스 다이어그램

```mermaid
classDiagram
    class EventType {
        <<enumeration>>
        VIOLENCE_START
        VIOLENCE_END
        VIOLENCE_ONGOING
        NORMAL
    }

    class EventData {
        +EventType event_type
        +float timestamp
        +int window_id
        +float confidence
        +Optional duration
        +Optional additional_info
        +__init__(event_type, timestamp, window_id, confidence, duration, additional_info)
        +to_dict() Dict
        +to_json() str
        +__str__() str
    }

    class EventConfig {
        +float alert_threshold
        +int min_consecutive_detections
        +float normal_threshold
        +int min_consecutive_normal
        +float min_event_duration
        +float max_event_duration
        +float cooldown_duration
        +bool enable_ongoing_alerts
        +float ongoing_alert_interval
        +bool save_event_log
        +str event_log_format
        +str event_log_path
        +__init__(...)
        +validate() bool
        +to_dict() Dict
    }

    class EventManager {
        -EventConfig config
        -bool current_event_active
        -Optional current_event_start_time
        -Optional current_event_start_window
        -Optional last_event_end_time
        -Optional last_ongoing_alert_time
        -int consecutive_violence
        -int consecutive_normal
        -List event_history
        -Dict event_callbacks
        -Optional logger
        +__init__(config)
        +process_classification_result(result) Optional
        +add_event_callback(event_type, callback) void
        +remove_event_callback(event_type, callback) bool
        +get_current_status() Dict
        +get_event_history(limit) List
        +reset() void
        -_handle_violence_detection(result) Optional
        -_handle_normal_detection(result) Optional
        -_create_event_data(event_type, result) EventData
        -_trigger_callbacks(event_data) void
        -_should_generate_ongoing_alert() bool
        -_is_in_cooldown() bool
    }

    class EventLogger {
        -str log_path
        -str log_format
        -bool enable_logging
        -Optional current_session_id
        -Optional log_file_path
        +__init__(log_path, log_format, enable_logging)
        +log_event(event_data) bool
        +set_session(session_id) str
        +close_session() void
        +get_log_file_path() Optional
        -_create_log_file() str
        -_write_json_event(event_data) bool
        -_write_csv_event(event_data) bool
        -_ensure_log_directory() void
    }

    EventManager --> EventConfig
    EventManager --> EventData
    EventManager --> EventLogger
    EventManager --> EventType
    EventData --> EventType
    EventLogger --> EventData
```

---

## 데이터 구조 클래스 다이어그램

```mermaid
classDiagram
    class PersonPose {
        +ndarray keypoints
        +Optional bbox
        +Optional track_id
        +float score
        +float detection_confidence
        +__init__(keypoints, bbox, track_id, score, detection_confidence)
        +get_keypoint(index) Tuple
        +get_bbox_center() Tuple
        +is_valid() bool
        +to_dict() Dict
        +from_dict(data) PersonPose
        +__str__() str
    }

    class FramePoses {
        +List persons
        +int frame_idx
        +float timestamp
        +Dict video_info
        +__init__(persons, frame_idx, timestamp, video_info)
        +get_valid_persons() List
        +get_person_by_track_id(track_id) Optional
        +add_person(person) void
        +remove_person(track_id) bool
        +to_dict() Dict
        +from_dict(data) FramePoses
        +__len__() int
        +__iter__() Iterator
    }

    class ClassificationResult {
        +int prediction
        +float confidence
        +List probabilities
        +float processing_time
        +int window_id
        +float timestamp
        +__init__(prediction, confidence, probabilities, processing_time, window_id, timestamp)
        +get_predicted_class_name(class_names) str
        +get_max_probability() float
        +to_dict() Dict
        +from_dict(data) ClassificationResult
        +__str__() str
    }

    class WindowAnnotation {
        +int window_id
        +int start_frame
        +int end_frame
        +List poses_sequence
        +Optional label
        +Optional confidence
        +Dict metadata
        +__init__(window_id, start_frame, end_frame, poses_sequence, label, confidence, metadata)
        +get_window_length() int
        +get_frame_at_index(index) Optional
        +add_frame(frame_poses) void
        +to_numpy() ndarray
        +to_dict() Dict
        +from_dict(data) WindowAnnotation
    }

    FramePoses --> PersonPose
    WindowAnnotation --> FramePoses
```

---

## 컴포넌트 다이어그램

```mermaid
graph TB
    subgraph InputLayer ["Input Layer"]
        A[RealtimeInputManager]
        A1[VideoCapture]
        A2[RTSPClient]
        A3[WebcamCapture]
    end

    subgraph ProcessingLayer ["Processing Layer"]
        B[InferencePipeline]
        B1[RTMO_PoseEstimator]
        B2[ByteTracker]
        B3[RegionBasedScorer]
        B4[SlidingWindowProcessor]
        B5[STGCN_Classifier]
    end

    subgraph EventLayer ["Event Management Layer"]
        C[EventManager]
        C1[EventLogger]
        C2[EventConfig]
    end

    subgraph VisualizationLayer ["Visualization Layer"]
        D[RealtimeVisualizer]
        D1[PoseVisualizer]
        D2[OpenCV_GUI]
    end

    subgraph UtilityLayer ["Utility Layer"]
        E[ModuleFactory]
        E1[PerformanceTracker]
        E2[ConfigLoader]
    end

    subgraph DataLayer ["Data Layer"]
        F[FramePoses]
        F1[PersonPose]
        F2[ClassificationResult]
        F3[EventData]
    end

    A -->|Frame_Stream| B
    B -->|Classification_Results| C
    B -->|Pose_Data_Results| D
    C -->|Event_Data| D
    C -->|Event_Logging| C1
  
    E -->|Module_Creation| B
    E1 -->|Performance_Monitoring| B
    E2 -->|Configuration| B

    B -->|Data_Processing| F
    C -->|Event_Creation| F3
    D -->|Data_Visualization| F

    A1 --> A
    A2 --> A
    A3 --> A

    B1 --> B
    B2 --> B
    B3 --> B
    B4 --> B
    B5 --> B

    C2 --> C
    D1 --> D
    D2 --> D

    F1 --> F
    F2 --> F
```

---

## 패키지 다이어그램

```mermaid
graph TB
    subgraph recognizer_package ["recognizer"]
        subgraph pipelines_package ["pipelines"]
            subgraph inference_package ["inference"]
                P1[pipeline_py]
            end
            subgraph base_package ["base"]
                P2[base_pipeline_py]
            end
        end

        subgraph pose_estimation_package ["pose_estimation"]
            subgraph rtmo_package ["rtmo"]
                PE1[rtmo_estimator_py]
                PE2[rtmo_onnx_estimator_py]
                PE3[rtmo_tensorrt_estimator_py]
            end
            PE4[base_py]
        end

        subgraph tracking_package ["tracking"]
            subgraph bytetrack_package ["bytetrack"]
                T1[byte_tracker_py]
            end
            T2[base_py]
        end

        subgraph action_classification_package ["action_classification"]
            subgraph stgcn_package ["stgcn"]
                AC1[stgcn_classifier_py]
            end
            AC2[base_py]
        end

        subgraph events_package ["events"]
            E1[event_manager_py]
            E2[event_types_py]
            E3[event_logger_py]
        end

        subgraph visualization_package ["visualization"]
            V1[realtime_visualizer_py]
            V2[pose_visualizer_py]
        end

        subgraph utils_package ["utils"]
            U1[realtime_input_py]
            U2[window_processor_py]
            U3[factory_py]
            U4[config_loader_py]
            U5[data_structure_py]
        end
    end

    P1 --> PE1
    P1 --> T1
    P1 --> AC1
    P1 --> E1
    P1 --> V1
    P1 --> U1
    P1 --> U2
    P1 --> U3

    PE1 --> PE4
    PE2 --> PE4
    PE3 --> PE4

    T1 --> T2
    AC1 --> AC2

    E1 --> E2
    E1 --> E3

    V1 --> V2
    V1 --> U5

    U1 --> U5
    U2 --> U5
    U3 --> U5
```

---

## 상속 계층 다이어그램

```mermaid
classDiagram
    class BasePipeline {
        <<abstract>>
        +initialize_pipeline() bool
        +run() bool
        +cleanup() void
    }

    class BasePoseEstimator {
        <<abstract>>
        +estimate_poses(frame) FramePoses
        +set_score_threshold(threshold) void
        +get_model_info() Dict
    }

    class BaseTracker {
        <<abstract>>
        +track(poses) FramePoses
        +reset() void
        +get_active_tracks() List
    }

    class BaseScorer {
        <<abstract>>
        +score_poses(poses) FramePoses
        +set_quality_threshold(threshold) void
        +get_scoring_criteria() Dict
    }

    class BaseActionClassifier {
        <<abstract>>
        +classify_window(window_data) ClassificationResult
        +set_confidence_threshold(threshold) void
        +get_class_names() List
        +warmup() void
    }

    class InferencePipeline {
        +run_realtime_mode(input_source) bool
        +run_analysis_mode(input_source) bool
    }

    class RTMOPoseEstimator {
        +estimate_poses(frame) FramePoses
        -_preprocess(frame) ndarray
        -_postprocess(output) FramePoses
    }

    class RTMOONNXEstimator {
        +estimate_poses(frame) FramePoses
        -_load_onnx_model() void
        -_onnx_inference(input_data) ndarray
    }

    class RTMOTensorRTEstimator {
        +estimate_poses(frame) FramePoses
        -_load_tensorrt_engine() void
        -_tensorrt_inference(input_data) ndarray
    }

    class ByteTrackerWrapper {
        +track(poses) FramePoses
        -_update_tracker(detections) List
        -_convert_to_poses(tracks) FramePoses
    }

    class RegionBasedScorer {
        +score_poses(poses) FramePoses
        -_calculate_region_score(pose) float
        -_filter_by_quality(poses) List
    }

    class STGCNActionClassifier {
        +classify_window(window_data) ClassificationResult
        -_preprocess_window(window_data) ndarray
        -_pytorch_inference(input_data) ndarray
        -_postprocess_output(output) ClassificationResult
    }

    BasePipeline <|-- InferencePipeline
    BasePoseEstimator <|-- RTMOPoseEstimator
    BasePoseEstimator <|-- RTMOONNXEstimator
    BasePoseEstimator <|-- RTMOTensorRTEstimator
    BaseTracker <|-- ByteTrackerWrapper
    BaseScorer <|-- RegionBasedScorer
    BaseActionClassifier <|-- STGCNActionClassifier
```

---

## 인터페이스 다이어그램

```mermaid
classDiagram
    class IPoseEstimator {
        <<interface>>
        +estimate_poses(frame) FramePoses
        +set_score_threshold(threshold) void
        +get_model_info() Dict
    }

    class ITracker {
        <<interface>>
        +track(poses) FramePoses
        +reset() void
        +get_active_tracks() List
    }

    class IActionClassifier {
        <<interface>>
        +classify_window(window_data) ClassificationResult
        +set_confidence_threshold(threshold) void
        +get_class_names() List
    }

    class IEventManager {
        <<interface>>
        +process_classification_result(result) Optional
        +add_event_callback(event_type, callback) void
        +get_current_status() Dict
    }

    class IVisualizer {
        <<interface>>
        +show_frame(frame, poses, classification, additional_info) bool
        +start_display() void
        +stop_display() void
    }

    class ILogger {
        <<interface>>
        +log_event(event_data) bool
        +set_session(session_id) str
        +close_session() void
    }

    IPoseEstimator <|.. RTMOONNXEstimator
    ITracker <|.. ByteTrackerWrapper
    IActionClassifier <|.. STGCNActionClassifier
    IEventManager <|.. EventManager
    IVisualizer <|.. RealtimeVisualizer
    ILogger <|.. EventLogger
```

---

## 메서드 호출 관계 다이어그램

```mermaid
graph LR
    A[InferencePipeline_run_realtime_mode] --> B[RealtimeInputManager_get_frame]
    A --> C[InferencePipeline_process_frame]
  
    C --> D[RTMOONNXEstimator_estimate_poses]
    C --> E[ByteTrackerWrapper_track]
    C --> F[RegionBasedScorer_score_poses]
    C --> G[SlidingWindowProcessor_add_frame_data]
  
    G --> H[SlidingWindowProcessor_is_ready]
    H --> I[SlidingWindowProcessor_get_window_data]
    I --> J[ClassificationQueue_put]
  
    K[classification_worker] --> L[ClassificationQueue_get]
    L --> M[STGCNActionClassifier_classify_window]
    M --> N[EventManager_process_classification_result]
    N --> O[EventLogger_log_event]
    N --> P[EventManager_trigger_callbacks]
  
    C --> Q[RealtimeVisualizer_show_frame]
    Q --> R[RealtimeVisualizer_draw_event_history]
    Q --> S[RealtimeVisualizer_draw_classification_results]
  
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style M fill:#fff3e0
    style N fill:#e8f5e8
    style Q fill:#fce4ec
```

---

## 데이터 플로우 클래스 다이어그램

```mermaid
classDiagram
    class DataFrame {
        +ndarray frame
        +int frame_idx
        +float timestamp
        +Dict metadata
    }

    class PoseData {
        +List persons
        +DataFrame frame_info
        +float processing_time
    }

    class TrackingData {
        +List tracked_persons
        +List active_tracks
        +List lost_tracks
        +List new_tracks
    }

    class ScoredData {
        +List scored_persons
        +Dict quality_metrics
        +int filtered_count
    }

    class WindowData {
        +int window_id
        +ndarray sequence
        +int start_frame
        +int end_frame
        +int persons_count
    }

    class ResultData {
        +ClassificationResult classification
        +Optional event_data
        +Dict performance_stats
    }

    DataFrame --> PoseData
    PoseData --> TrackingData
    TrackingData --> ScoredData
    ScoredData --> WindowData
    WindowData --> ResultData
```

---

## UML 다이어그램 요약

### 주요 설계 패턴

1. **Strategy Pattern**: 각 모듈(포즈 추정, 추적, 분류)은 교체 가능한 전략으로 구현
2. **Factory Pattern**: ModuleFactory를 통한 모듈 생성 및 관리
3. **Observer Pattern**: EventManager의 콜백 시스템
4. **Template Method**: BasePipeline의 추상 메서드 구조
5. **Singleton Pattern**: ModuleFactory의 전역 인스턴스 관리

### 핵심 아키텍처 특징

1. **계층화된 구조**: 입력, 처리, 이벤트 관리, 시각화 계층 분리
2. **플러그인 아키텍처**: 각 모듈은 독립적으로 교체 가능
3. **비동기 처리**: 분류 작업의 별도 스레드 처리
4. **이벤트 기반**: 결과 처리를 위한 이벤트 시스템
5. **데이터 중심**: 명확한 데이터 구조와 변환 흐름

### 확장 포인트

1. **새로운 포즈 추정기**: BasePoseEstimator 상속
2. **새로운 분류기**: BaseActionClassifier 상속
3. **새로운 이벤트 타입**: EventType 열거형 확장
4. **새로운 시각화**: IVisualizer 인터페이스 구현
5. **새로운 로거**: ILogger 인터페이스 구현


