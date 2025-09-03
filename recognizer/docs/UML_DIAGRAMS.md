# 실시간 추론 시스템 UML 다이어그램

## 개요

본 문서는 Violence Detection 실시간 추론 시스템의 UML 다이어그램을 제공한다. 클래스 다이어그램, 컴포넌트 다이어그램, 패키지 다이어그램을 통해 시스템의 구조적 관계를 시각화한다.

**최신 업데이트**: 2025-09-03 기준, ONNX 모델 통합, Temperature Scaling, 멀티프로세스 지원 반영

---

## 목차

- [실시간 추론 시스템 UML 다이어그램](#실시간-추론-시스템-uml-다이어그램)
  - [개요](#개요)
  - [목차](#목차)
  - [전체 시스템 클래스 다이어그램](#전체-시스템-클래스-다이어그램)
  - [파이프라인 클래스 다이어그램](#파이프라인-클래스-다이어그램)
  - [ONNX 모델 통합 클래스 다이어그램](#onnx-모델-통합-클래스-다이어그램)
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
        -str mode
        +__init__(config)
        +initialize_pipeline() bool
        +run_realtime_mode(input_source) bool
        +run_analysis_mode(input_source) bool
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
        +run_visualize_mode(pkl_path, video_path) bool
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

    class BatchAnalysisProcessor {
        -Dict config
        -InferencePipeline pipeline
        -bool save_results
        +__init__(config)
        +process_file(input_path, output_dir) Dict
        +process_folder(input_dir, output_dir) Dict
        -_save_results(results, output_path) bool
        -_generate_csv_report(results, output_path) bool
    }

    InferencePipeline --|> BasePipeline
    InferencePipeline --> ModuleFactory
    InferencePipeline --> PerformanceTracker
    BatchAnalysisProcessor --> InferencePipeline
```

---

## ONNX 모델 통합 클래스 다이어그램

```mermaid
classDiagram
    class ONNXInferenceBase {
        <<abstract>>
        #str model_path
        #str device
        #dict onnx_providers
        #object onnx_session
        +__init__(config)
        +_load_onnx_model() void
        +_get_providers(device) List
        +_create_session(model_path, providers) object
        +warmup() void*
    }

    class RTMOONNXEstimator {
        -tuple input_size
        -float score_threshold
        -str input_name
        -str output_name
        -dict onnx_config
        +__init__(config)
        +estimate_poses(frame) FramePoses
        +warmup() void
        -_preprocess(frame) ndarray
        -_postprocess(outputs) List
        -_onnx_inference(input_data) List
        -_apply_nms(detections) List
    }

    class STGCNONNXClassifier {
        -int window_size
        -int max_persons
        -float temperature
        -str input_format
        -List class_names
        +__init__(config)
        +classify_window(window_data) ClassificationResult
        +warmup() void
        -_preprocess_window(window_data) ndarray
        -_apply_temperature_scaling(raw_scores) ndarray
        -_postprocess_result(probabilities, window_id) ClassificationResult
    }

    class STGCNActionClassifier {
        -str model_path
        -object model
        -str device
        -int window_size
        -List class_names
        +__init__(config)
        +classify_window(window_data) ClassificationResult
        +warmup() void
        -_load_pytorch_model() void
        -_preprocess_window(window_data) Tensor
        -_pytorch_inference(input_data) Tensor
        -_postprocess_output(output) ClassificationResult
    }

    class TemperatureScaling {
        <<utility>>
        +apply_scaling(raw_scores, temperature) ndarray
        +find_optimal_temperature(logits, labels) float
        +validate_probabilities(probabilities) bool
        +calibrate_model(model, validation_data) float
    }

    ONNXInferenceBase <|-- RTMOONNXEstimator
    ONNXInferenceBase <|-- STGCNONNXClassifier
    BaseActionClassifier <|-- STGCNONNXClassifier
    BaseActionClassifier <|-- STGCNActionClassifier
    STGCNONNXClassifier --> TemperatureScaling
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
        +to_csv_row() str
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
        +from_dict(data) EventConfig
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
        -object csv_writer
        +__init__(log_path, log_format, enable_logging)
        +log_event(event_data) bool
        +set_session(session_id) str
        +close_session() void
        +get_log_file_path() Optional
        -_create_log_file() str
        -_write_json_event(event_data) bool
        -_write_csv_event(event_data) bool
        -_ensure_log_directory() void
        -_initialize_csv_writer() void
    }

    class EventCallback {
        <<interface>>
        +on_violence_start(event_data) void
        +on_violence_end(event_data) void
        +on_violence_ongoing(event_data) void
        +on_normal_state(event_data) void
    }

    EventManager --> EventConfig
    EventManager --> EventData
    EventManager --> EventLogger
    EventManager --> EventType
    EventManager --> EventCallback
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
        +Dict metadata
        +__init__(keypoints, bbox, track_id, score, detection_confidence, metadata)
        +get_keypoint(index) Tuple
        +get_bbox_center() Tuple
        +is_valid() bool
        +to_dict() Dict
        +from_dict(data) PersonPose
        +to_mmaction_format() Dict
        +__str__() str
    }

    class FramePoses {
        +List persons
        +int frame_idx
        +float timestamp
        +Dict video_info
        +Dict metadata
        +__init__(persons, frame_idx, timestamp, video_info, metadata)
        +get_valid_persons() List
        +get_person_by_track_id(track_id) Optional
        +add_person(person) void
        +remove_person(track_id) bool
        +to_dict() Dict
        +from_dict(data) FramePoses
        +to_numpy_array() ndarray
        +to_mmaction_format() Dict
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
        +str model_version
        +Dict metadata
        +__init__(prediction, confidence, probabilities, processing_time, window_id, timestamp, model_version, metadata)
        +get_predicted_class_name(class_names) str
        +get_max_probability() float
        +is_valid_probability_range() bool
        +to_dict() Dict
        +from_dict(data) ClassificationResult
        +to_csv_row() str
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
        +str annotation_format
        +__init__(window_id, start_frame, end_frame, poses_sequence, label, confidence, metadata, annotation_format)
        +get_window_length() int
        +get_frame_at_index(index) Optional
        +add_frame(frame_poses) void
        +to_numpy() ndarray
        +to_dict() Dict
        +from_dict(data) WindowAnnotation
        +to_mmaction_format() Dict
        +validate_structure() bool
    }

    class EvaluationResult {
        +float accuracy
        +float precision
        +float recall
        +float f1_score
        +ndarray confusion_matrix
        +List per_class_metrics
        +Dict metadata
        +str model_version
        +__init__(accuracy, precision, recall, f1_score, confusion_matrix, per_class_metrics, metadata, model_version)
        +to_dict() Dict
        +generate_report() str
        +save_charts(output_dir) void
        +compare_with(other_result) Dict
    }

    FramePoses --> PersonPose
    WindowAnnotation --> FramePoses
    ClassificationResult --> WindowAnnotation
    EvaluationResult --> ClassificationResult
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
        B2[RTMO_ONNX_Estimator]
        B3[ByteTracker]
        B4[RegionBasedScorer]
        B5[SlidingWindowProcessor]
        B6[STGCN_Classifier]
        B7[STGCN_ONNX_Classifier]
    end

    subgraph EventLayer ["Event Management Layer"]
        C[EventManager]
        C1[EventLogger]
        C2[EventConfig]
        C3[EventCallback]
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
        E3[BatchAnalysisProcessor]
    end

    subgraph DataLayer ["Data Layer"]
        F[FramePoses]
        F1[PersonPose]
        F2[ClassificationResult]
        F3[EventData]
        F4[WindowAnnotation]
        F5[EvaluationResult]
    end

    subgraph ONNXLayer ["ONNX Optimization Layer"]
        G[ONNXInferenceBase]
        G1[TemperatureScaling]
        G2[ONNXOptimizer]
    end

    A -->|Frame_Stream| B
    B -->|Classification_Results| C
    B -->|Pose_Data_Results| D
    C -->|Event_Data| D
    C -->|Event_Logging| C1
  
    E -->|Module_Creation| B
    E1 -->|Performance_Monitoring| B
    E2 -->|Configuration| B
    E3 -->|Batch_Processing| B

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
    B6 --> B
    B7 --> B

    C2 --> C
    C3 --> C
    D1 --> D
    D2 --> D

    F1 --> F
    F2 --> F
    F4 --> F
    F5 --> F

    G --> B2
    G --> B7
    G1 --> B7
    G2 --> G
```

---

## 패키지 다이어그램

```mermaid
graph TB
    subgraph recognizer_package ["recognizer"]
        subgraph core_package ["core"]
            C1[inference_modes_py]
            C2[mode_manager_py]
        end
        
        subgraph pipelines_package ["pipelines"]
            subgraph inference_package ["inference"]
                P1[pipeline_py]
            end
            subgraph analysis_package ["analysis"]
                P2[batch_processor_py]
            end
            subgraph base_package ["base"]
                P3[base_pipeline_py]
            end
        end

        subgraph pose_estimation_package ["pose_estimation"]
            subgraph rtmo_package ["rtmo"]
                PE1[rtmo_estimator_py]
                PE2[rtmo_onnx_estimator_py]
                PE3[rtmo_tensorrt_estimator_py]
            end
            PE4[base_py]
            PE5[onnx_inference_base_py]
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
                AC2[stgcn_onnx_classifier_py]
            end
            AC3[base_py]
            AC4[temperature_scaling_py]
        end

        subgraph events_package ["events"]
            E1[event_manager_py]
            E2[event_types_py]
            E3[event_logger_py]
            E4[event_callback_py]
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
            U6[multi_process_splitter_py]
        end

        subgraph evaluation_package ["evaluation"]
            EV1[evaluator_py]
            EV2[metrics_py]
            EV3[report_generator_py]
        end

        subgraph tools_package ["tools"]
            TO1[onnx_optimizer_py]
            TO2[model_converter_py]
        end
    end

    C1 --> P1
    P1 --> PE1
    P1 --> PE2
    P1 --> T1
    P1 --> AC1
    P1 --> AC2
    P1 --> E1
    P1 --> V1
    P1 --> U1
    P1 --> U2
    P1 --> U3

    PE1 --> PE4
    PE2 --> PE4
    PE2 --> PE5
    PE3 --> PE4

    T1 --> T2
    AC1 --> AC3
    AC2 --> AC3
    AC2 --> AC4

    E1 --> E2
    E1 --> E3
    E1 --> E4

    V1 --> V2
    V1 --> U5

    U1 --> U5
    U2 --> U5
    U3 --> U5

    EV1 --> EV2
    EV1 --> EV3

    TO1 --> PE2
    TO1 --> AC2
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

    class ONNXInferenceBase {
        <<abstract>>
        +_load_onnx_model() void
        +_get_providers(device) List
        +warmup() void*
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
        +run_visualize_mode(pkl_path, video_path) bool
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

    class STGCNONNXClassifier {
        +classify_window(window_data) ClassificationResult
        -_preprocess_window(window_data) ndarray
        -_apply_temperature_scaling(raw_scores) ndarray
        -_postprocess_result(probabilities, window_id) ClassificationResult
    }

    BasePipeline <|-- InferencePipeline
    BasePoseEstimator <|-- RTMOPoseEstimator
    BasePoseEstimator <|-- RTMOONNXEstimator
    BasePoseEstimator <|-- RTMOTensorRTEstimator
    ONNXInferenceBase <|-- RTMOONNXEstimator
    ONNXInferenceBase <|-- STGCNONNXClassifier
    BaseTracker <|-- ByteTrackerWrapper
    BaseScorer <|-- RegionBasedScorer
    BaseActionClassifier <|-- STGCNActionClassifier
    BaseActionClassifier <|-- STGCNONNXClassifier
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

    class IONNXOptimizer {
        <<interface>>
        +optimize_model(model_path, target_device) Dict
        +benchmark_configurations(configs) List
        +apply_temperature_scaling(model_config) Dict
    }

    class IEvaluator {
        <<interface>>
        +evaluate_model(model, test_data) EvaluationResult
        +generate_confusion_matrix(predictions, labels) ndarray
        +calculate_metrics(predictions, labels) Dict
    }

    IPoseEstimator <|.. RTMOONNXEstimator
    ITracker <|.. ByteTrackerWrapper
    IActionClassifier <|.. STGCNONNXClassifier
    IEventManager <|.. EventManager
    IVisualizer <|.. RealtimeVisualizer
    ILogger <|.. EventLogger
    IONNXOptimizer <|.. ONNXOptimizer
    IEvaluator <|.. ModelEvaluator
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
    L --> M[STGCNONNXClassifier_classify_window]
    M --> N[TemperatureScaling_apply_scaling]
    N --> O[EventManager_process_classification_result]
    O --> P[EventLogger_log_event]
    O --> Q[EventManager_trigger_callbacks]
  
    C --> R[RealtimeVisualizer_show_frame]
    R --> S[RealtimeVisualizer_draw_event_history]
    R --> T[RealtimeVisualizer_draw_classification_results]

    U[ONNXOptimizer_optimize_model] --> V[RTMOONNXEstimator_benchmark]
    U --> W[STGCNONNXClassifier_find_optimal_temperature]
    W --> X[TemperatureScaling_calibrate_model]
  
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style M fill:#fff3e0
    style N fill:#e8f5e8
    style O fill:#fce4ec
    style U fill:#f1f8e9
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
        +str source_type
    }

    class PoseData {
        +List persons
        +DataFrame frame_info
        +float processing_time
        +str model_version
        +Dict onnx_metrics
    }

    class TrackingData {
        +List tracked_persons
        +List active_tracks
        +List lost_tracks
        +List new_tracks
        +Dict tracking_stats
    }

    class ScoredData {
        +List scored_persons
        +Dict quality_metrics
        +int filtered_count
        +float avg_confidence
    }

    class WindowData {
        +int window_id
        +ndarray sequence
        +int start_frame
        +int end_frame
        +int persons_count
        +str data_format
    }

    class ResultData {
        +ClassificationResult classification
        +Optional event_data
        +Dict performance_stats
        +bool temperature_applied
        +float onnx_inference_time
    }

    class EvaluationData {
        +List results
        +ndarray confusion_matrix
        +Dict metrics
        +str model_comparison
        +Dict temperature_analysis
    }

    DataFrame --> PoseData
    PoseData --> TrackingData
    TrackingData --> ScoredData
    ScoredData --> WindowData
    WindowData --> ResultData
    ResultData --> EvaluationData
```

---

## UML 다이어그램 요약

### 주요 설계 패턴

1. **Strategy Pattern**: 각 모듈(포즈 추정, 추적, 분류)은 교체 가능한 전략으로 구현
2. **Factory Pattern**: ModuleFactory를 통한 모듈 생성 및 관리
3. **Observer Pattern**: EventManager의 콜백 시스템
4. **Template Method**: BasePipeline의 추상 메서드 구조  
5. **Singleton Pattern**: ModuleFactory의 전역 인스턴스 관리
6. **Adapter Pattern**: ONNX 모델과 PyTorch 모델 간의 통합 인터페이스
7. **Chain of Responsibility**: 이벤트 처리 체인 (detection → validation → logging)

### 핵심 아키텍처 특징

1. **계층화된 구조**: 입력, 처리, 이벤트 관리, 시각화 계층 분리
2. **플러그인 아키텍처**: 각 모듈은 독립적으로 교체 가능
3. **비동기 처리**: 분류 작업의 별도 스레드 처리
4. **이벤트 기반**: 결과 처리를 위한 이벤트 시스템
5. **데이터 중심**: 명확한 데이터 구조와 변환 흐름
6. **ONNX 최적화**: 성능 향상을 위한 ONNX 런타임 통합
7. **Temperature Scaling**: ONNX 모델 출력 정규화 및 보정
8. **멀티프로세스 지원**: 대규모 배치 처리를 위한 병렬 실행
9. **Docker 호환성**: 컨테이너 환경에서 안정적 실행

### 확장 포인트

1. **새로운 포즈 추정기**: BasePoseEstimator 상속 또는 ONNXInferenceBase 상속
2. **새로운 분류기**: BaseActionClassifier 상속, ONNX 지원 시 ONNXInferenceBase 상속
3. **새로운 이벤트 타입**: EventType 열거형 확장 및 EventCallback 구현
4. **새로운 시각화**: IVisualizer 인터페이스 구현
5. **새로운 로거**: ILogger 인터페이스 구현
6. **ONNX 최적화 확장**: IONNXOptimizer 인터페이스로 새로운 최적화 전략 추가
7. **평가 메트릭 확장**: IEvaluator 인터페이스로 새로운 평가 방법 추가
8. **Temperature Scaling 전략**: TemperatureScaling 클래스 확장으로 새로운 보정 방법 추가

### 최신 기능 통합 (2025-09-03)

1. **ONNX 런타임 지원**: RTMOONNXEstimator, STGCNONNXClassifier
2. **Temperature Scaling**: 정확한 확률값 출력을 위한 후처리
3. **성능 최적화**: GPU별 자동 최적화 및 벤치마킹
4. **평가 시스템**: 모델 성능 비교 및 분석 도구
5. **멀티프로세스**: annotation 스타일 병렬 처리 지원
6. **Docker 통합**: 컨테이너 환경에서 MMCV 호환성 보장
7. **종합 로깅**: JSON/CSV 형식으로 상세한 이벤트 및 성능 기록