# UML 다이어그램 설계 문서

## 개요

Recognizer 시스템의 포괄적인 UML 다이어그램 모음으로, 시스템의 구조적 관계, 행위적 상호작용, 데이터 플로우를 시각화합니다. 모든 다이어그램은 Mermaid 형식으로 작성되어 GitHub에서 바로 렌더링됩니다.

## 목차

1. [클래스 다이어그램](#클래스-다이어그램)
2. [시퀀스 다이어그램](#시퀀스-다이어그램)
3. [상태 다이어그램](#상태-다이어그램)
4. [컴포넌트 다이어그램](#컴포넌트-다이어그램)
5. [배포 다이어그램](#배포-다이어그램)
6. [액티비티 다이어그램](#액티비티-다이어그램)
7. [유스케이스 다이어그램](#유스케이스-다이어그램)

## 클래스 다이어그램

### 1. 전체 시스템 클래스 구조

```mermaid
classDiagram
    %% 핵심 인프라
    class ModuleFactory {
        <<Singleton>>
        -_pose_estimators: Dict[str, Type]
        -_trackers: Dict[str, Type]
        -_scorers: Dict[str, Type]
        -_classifiers: Dict[str, Type]
        -_instance: ModuleFactory
        +register_pose_estimator(name, class, config): void
        +register_tracker(name, class, config): void
        +register_scorer(name, class, config): void
        +register_classifier(name, class, config): void
        +create_pose_estimator(name, config): BasePoseEstimator
        +create_tracker(name, config): BaseTracker
        +create_scorer(name, config): BaseScorer
        +create_classifier(name, config): BaseClassifier
        +get_instance(): ModuleFactory
        +list_registered_modules(): Dict[str, List[str]]
    }

    %% 기본 모드 클래스
    class BaseMode {
        <<Abstract>>
        #config: Dict[str, Any]
        #mode_config: Dict[str, Any]
        #logger: logging.Logger
        +__init__(config: Dict[str, Any])
        +execute(): bool*
        #_validate_config(required_keys: List[str]): bool
        #_get_mode_config(): Dict[str, Any]
        #_setup_logging(): void
        #_cleanup(): void
    }

    %% 실행 모드들
    class AnnotationMode {
        -current_stage: str
        -multi_process_enabled: bool
        -output_dir: str
        -input_sources: List[str]
        +execute(): bool
        -_execute_stage1(): bool
        -_execute_stage2(): bool
        -_execute_stage3(): bool
        -_setup_multiprocessing(): void
        -_validate_input_sources(): bool
        -_create_output_directories(): void
    }

    class AnalysisMode {
        -dual_service_pipeline: DualServicePipeline
        -evaluation_enabled: bool
        -multi_process_config: Dict[str, Any]
        +execute(): bool
        -_execute_with_dual_pipeline(): bool
        -_run_performance_evaluation(): bool
        -_setup_evaluation_metrics(): void
        -_generate_reports(): void
    }

    class RealtimeMode {
        -pipeline: DualServicePipeline
        -input_manager: RealtimeInputManager
        -visualizer: InferenceVisualizer
        -event_manager: EventManager
        +execute(): bool
        -_process_video_stream(): bool
        -_display_frame(frame, results): void
        -_handle_user_input(): bool
        -_cleanup_resources(): void
    }

    class VisualizeMode {
        -pkl_visualizer: PKLVisualizer
        -stage: str
        -save_mode: bool
        +execute(): bool
        -_visualize_pkl_results(): bool
        -_load_pkl_data(path): VisualizationData
        -_render_overlay_video(): bool
    }

    %% 파이프라인 클래스들
    class BasePipeline {
        <<Abstract>>
        #config: Dict[str, Any]
        #initialized: bool
        #logger: logging.Logger
        +initialize_pipeline(): bool*
        +process_video(video_path): Dict[str, Any]*
        +cleanup(): void*
        #_validate_pipeline_config(): bool
        #_log_performance_stats(): void
    }

    class DualServicePipeline {
        -pose_estimator: BasePoseEstimator
        -tracker: BaseTracker
        -scorers: Dict[str, BaseScorer]
        -classifiers: Dict[str, BaseClassifier]
        -event_manager: EventManager
        -visualizer: InferenceVisualizer
        -window_processor: WindowProcessor
        -services: List[str]
        -dual_service_enabled: bool
        +initialize_pipeline(): bool
        +process_frame(frame): ProcessingResult
        +process_video(video_path): Dict[str, Any]
        +cleanup(): void
        -_initialize_common_modules(): bool
        -_initialize_service_modules(): bool
        -_process_window_classification(window_data): Dict[str, Any]
        -_handle_dual_service_results(results): EventData
    }

    class SeparatedPipeline {
        -current_stage: int
        -stage_configs: Dict[int, Dict[str, Any]]
        +execute_stage1(): bool
        +execute_stage2(): bool
        +execute_stage3(): bool
        -_load_stage_data(stage: int): Any
        -_save_stage_results(stage: int, data: Any): bool
        -_validate_stage_completion(stage: int): bool
    }

    class AnalysisPipeline {
        -evaluation_metrics: Dict[str, Any]
        -report_generator: ReportGenerator
        -batch_processor: BatchProcessor
        +process_batch(video_paths): Dict[str, Any]
        +generate_evaluation_report(): bool
        -_calculate_performance_metrics(results): Dict[str, Any]
        -_create_confusion_matrix(predictions, labels): np.ndarray
        -_generate_charts(): void
    }

    %% 상속 관계
    BaseMode <|-- AnnotationMode
    BaseMode <|-- AnalysisMode
    BaseMode <|-- RealtimeMode
    BaseMode <|-- VisualizeMode

    BasePipeline <|-- DualServicePipeline
    BasePipeline <|-- SeparatedPipeline
    BasePipeline <|-- AnalysisPipeline

    %% 의존 관계
    AnnotationMode --> SeparatedPipeline
    AnalysisMode --> DualServicePipeline
    AnalysisMode --> AnalysisPipeline
    RealtimeMode --> DualServicePipeline
    VisualizeMode --> PKLVisualizer

    DualServicePipeline --> ModuleFactory
```

### 2. 포즈 추정 모듈 클래스 구조

```mermaid
classDiagram
    class BasePoseEstimator {
        <<Abstract>>
        #config: Dict[str, Any]
        #device: str
        #score_threshold: float
        #nms_threshold: float
        #keypoint_threshold: float
        #model_input_size: Tuple[int, int]
        +initialize_model(): bool*
        +estimate_poses(frame: np.ndarray): List[Person]*
        +set_score_threshold(threshold: float): void
        +set_nms_threshold(threshold: float): void
        +get_model_info(): Dict[str, Any]*
        +warmup(num_runs: int): void
        +cleanup(): void*
        #_preprocess_frame(frame: np.ndarray): np.ndarray*
        #_postprocess_results(results: Any): List[Person]*
        #_apply_nms(detections: List): List
    }

    class RTMOEstimator {
        -model: torch.nn.Module
        -device: torch.device
        -cfg: mmengine.Config
        -data_preprocessor: BaseDataPreprocessor
        +initialize_model(): bool
        +estimate_poses(frame: np.ndarray): List[Person]
        +get_model_info(): Dict[str, Any]
        +cleanup(): void
        #_preprocess_frame(frame: np.ndarray): np.ndarray
        #_postprocess_results(results: Any): List[Person]
        -_load_checkpoint(checkpoint_path: str): bool
        -_setup_data_preprocessor(): void
    }

    class RTMOONNXEstimator {
        -session: onnxruntime.InferenceSession
        -input_name: str
        -output_names: List[str]
        -execution_providers: List[str]
        -session_options: onnxruntime.SessionOptions
        -io_binding: Optional[onnxruntime.IOBinding]
        +initialize_model(): bool
        +estimate_poses(frame: np.ndarray): List[Person]
        +get_model_info(): Dict[str, Any]
        +cleanup(): void
        #_preprocess_frame(frame: np.ndarray): np.ndarray
        #_postprocess_results(results: np.ndarray): List[Person]
        -_setup_onnx_session(): bool
        -_optimize_session_options(): void
        -_setup_io_binding(): void
        -_run_inference_with_binding(input_data: np.ndarray): np.ndarray
    }

    class RTMOTensorRTEstimator {
        -engine: tensorrt.ICudaEngine
        -context: tensorrt.IExecutionContext
        -stream: cudart.cudaStream_t
        -input_binding: int
        -output_binding: int
        -input_buffer_gpu: Any
        -output_buffer_gpu: Any
        -input_buffer_cpu: np.ndarray
        -output_buffer_cpu: np.ndarray
        +initialize_model(): bool
        +estimate_poses(frame: np.ndarray): List[Person]
        +get_model_info(): Dict[str, Any]
        +cleanup(): void
        #_preprocess_frame(frame: np.ndarray): np.ndarray
        #_postprocess_results(results: np.ndarray): List[Person]
        -_load_engine(engine_path: str): bool
        -_allocate_buffers(): void
        -_setup_cuda_stream(): void
        -_run_tensorrt_inference(input_data: np.ndarray): np.ndarray
    }

    class EnhancedRTMOExtractor {
        -estimator: BasePoseEstimator
        -batch_size: int
        -use_multi_threading: bool
        -thread_pool: Optional[ThreadPoolExecutor]
        +extract_poses_batch(frames: List[np.ndarray]): List[List[Person]]
        +extract_poses_async(frame: np.ndarray): Future[List[Person]]
        -_batch_inference(frames: List[np.ndarray]): List[List[Person]]
        -_parallel_postprocess(results: List): List[List[Person]]
    }

    %% 상속 관계
    BasePoseEstimator <|-- RTMOEstimator
    BasePoseEstimator <|-- RTMOONNXEstimator
    BasePoseEstimator <|-- RTMOTensorRTEstimator

    %% 조합 관계
    EnhancedRTMOExtractor --> BasePoseEstimator
```

### 3. 추적 및 스코어링 모듈 클래스 구조

```mermaid
classDiagram
    class BaseTracker {
        <<Abstract>>
        #config: Dict[str, Any]
        #frame_id: int
        #active_tracks: Dict[int, Any]
        #lost_tracks: List[Any]
        +initialize(): bool*
        +update_tracks(persons: List[Person]): List[Person]*
        +reset(): void*
        +get_active_tracks(): List[int]
        +get_track_count(): int
        #_convert_to_detections(persons: List[Person]): List[Detection]*
        #_update_persons_with_tracks(persons: List[Person], tracks: List): List[Person]*
    }

    class ByteTrackerWrapper {
        -tracker: BYTETracker
        -track_thresh: float
        -match_thresh: float
        -track_buffer: int
        -frame_rate: int
        -mot20: bool
        +initialize(): bool
        +update_tracks(persons: List[Person]): List[Person]
        +reset(): void
        #_convert_to_detections(persons: List[Person]): List[Detection]
        #_update_persons_with_tracks(persons: List[Person], tracks: List): List[Person]
        -_create_detection(person: Person): Detection
        -_update_person_with_track(person: Person, track: Any): Person
    }

    class BaseScorer {
        <<Abstract>>
        #config: Dict[str, Any]
        #track_histories: Dict[int, List[Dict]]
        #quality_threshold: float
        #min_track_length: int
        +calculate_score(track_data: Dict[str, Any]): float*
        +update_track(person: Person): void*
        +get_track_scores(): Dict[int, float]
        +filter_by_quality(persons: List[Person]): List[Person]
        #_extract_features(person: Person): Dict[str, float]*
        #_calculate_quality_score(features: Dict[str, float]): float*
    }

    class MotionBasedScorer {
        -weights: Dict[str, float]
        -history_length: int
        -movement_threshold: float
        -interaction_threshold: float
        +calculate_score(track_data: Dict[str, Any]): float
        +update_track(person: Person): void
        #_extract_features(person: Person): Dict[str, float]
        #_calculate_quality_score(features: Dict[str, float]): float
        -_calculate_movement_score(history: List[Dict]): float
        -_calculate_interaction_score(person: Person, others: List[Person]): float
        -_calculate_position_score(person: Person): float
        -_calculate_temporal_consistency(history: List[Dict]): float
    }

    class FalldownScorer {
        -height_change_weight: float
        -posture_angle_weight: float
        -movement_intensity_weight: float
        -persistence_weight: float
        -position_weight: float
        +calculate_score(track_data: Dict[str, Any]): float
        +update_track(person: Person): void
        #_extract_features(person: Person): Dict[str, float]
        #_calculate_quality_score(features: Dict[str, float]): float
        -_calculate_height_change(history: List[Dict]): float
        -_calculate_posture_angle(keypoints: np.ndarray): float
        -_calculate_movement_intensity(history: List[Dict]): float
        -_calculate_persistence_score(history: List[Dict]): float
    }

    %% 상속 관계
    BaseTracker <|-- ByteTrackerWrapper
    BaseScorer <|-- MotionBasedScorer
    BaseScorer <|-- FalldownScorer
```

### 4. 동작 분류 모듈 클래스 구조

```mermaid
classDiagram
    class BaseClassifier {
        <<Abstract>>
        #config: Dict[str, Any]
        #model: Any
        #device: str
        #class_names: List[str]
        #num_classes: int
        #confidence_threshold: float
        #window_size: int
        +initialize(): bool*
        +classify_window(window_data: np.ndarray): ClassificationResult*
        +set_confidence_threshold(threshold: float): void
        +get_class_names(): List[str]
        +warmup(num_runs: int): void
        +cleanup(): void*
        #_preprocess_window(window_data: np.ndarray): torch.Tensor*
        #_postprocess_results(output: torch.Tensor): ClassificationResult*
        #_validate_window_data(window_data: np.ndarray): bool
    }

    class STGCNActionClassifier {
        -model: torch.nn.Module
        -cfg: mmengine.Config
        -checkpoint_path: str
        -input_format: str
        -coordinate_dimensions: int
        -expected_keypoint_count: int
        -max_persons: int
        +initialize(): bool
        +classify_window(window_data: np.ndarray): ClassificationResult
        +cleanup(): void
        #_preprocess_window(window_data: np.ndarray): torch.Tensor
        #_postprocess_results(output: torch.Tensor): ClassificationResult
        -_load_model_config(): mmengine.Config
        -_load_checkpoint(): bool
        -_normalize_coordinates(keypoints: np.ndarray): np.ndarray
        -_pad_or_truncate_persons(window_data: np.ndarray): np.ndarray
        -_convert_to_stgcn_format(window_data: np.ndarray): torch.Tensor
    }

    class ClassificationResult {
        +service_type: str
        +class_id: int
        +class_name: str
        +confidence: float
        +probabilities: np.ndarray
        +timestamp: float
        +frame_range: Tuple[int, int]
        +window_id: int
        +processing_time: float
        +metadata: Dict[str, Any]
        +__init__(...)
        +is_valid(): bool
        +get_max_confidence(): float
        +get_predicted_class(): str
        +to_dict(): Dict[str, Any]
        +from_dict(data: Dict[str, Any]): ClassificationResult
    }

    %% 상속 관계
    BaseClassifier <|-- STGCNActionClassifier

    %% 연관 관계
    STGCNActionClassifier --> ClassificationResult : creates
```

### 5. 이벤트 관리 시스템 클래스 구조

```mermaid
classDiagram
    class EventType {
        <<Enumeration>>
        FIGHT_START
        FIGHT_END
        FIGHT_ONGOING
        FALLDOWN_START
        FALLDOWN_END
        FALLDOWN_ONGOING
        NORMAL
        UNKNOWN
        +get_display_name(): str
        +is_alert_event(): bool
        +get_priority(): int
    }

    class EventStatus {
        <<Enumeration>>
        IDLE
        DETECTING
        ACTIVE
        ENDING
        COMPLETED
        CANCELLED
        +is_active_status(): bool
        +can_transition_to(status: EventStatus): bool
    }

    class Event {
        +event_id: str
        +event_type: EventType
        +confidence: float
        +start_time: float
        +end_time: Optional[float]
        +duration: Optional[float]
        +status: EventStatus
        +metadata: Dict[str, Any]
        +participants: List[int]
        +window_ids: List[int]
        +__init__(...)
        +is_active(): bool
        +get_duration(): float
        +update_confidence(confidence: float): void
        +add_participant(track_id: int): void
        +to_dict(): Dict[str, Any]
        +from_dict(data: Dict[str, Any]): Event
    }

    class EventManager {
        -event_configs: Dict[str, Dict[str, Any]]
        -active_events: Dict[str, Dict[str, Any]]
        -event_history: List[Event]
        -logger: EventLogger
        -callbacks: Dict[EventType, List[Callable]]
        -consecutive_counters: Dict[str, Dict[str, int]]
        -last_event_times: Dict[str, float]
        +__init__(config: Dict[str, Any])
        +process_classification_result(result: ClassificationResult): List[Event]
        +add_event_callback(event_type: EventType, callback: Callable): void
        +remove_event_callback(event_type: EventType, callback: Callable): bool
        +get_current_status(): Dict[str, Any]
        +get_event_history(limit: Optional[int]): List[Event]
        +get_active_events(): Dict[str, Any]
        +reset(): void
        -_check_event_triggers(service: str, result: ClassificationResult): Optional[Event]
        -_validate_event_continuity(service: str, result: ClassificationResult): bool
        -_update_event_states(): void
        -_trigger_callbacks(event: Event): void
        -_cleanup_expired_events(): void
    }

    class EventLogger {
        -log_path: str
        -log_format: str
        -enable_logging: bool
        -current_session_id: Optional[str]
        -log_file_handle: Optional[IO]
        -csv_writer: Optional[csv.DictWriter]
        +__init__(log_path: str, log_format: str, enable_logging: bool)
        +log_event(event: Event): bool
        +set_session(session_id: Optional[str]): str
        +close_session(): void
        +get_log_file_path(): Optional[str]
        +flush(): void
        -_create_log_file(): str
        -_write_json_event(event: Event): bool
        -_write_csv_event(event: Event): bool
        -_ensure_log_directory(): void
        -_generate_session_id(): str
    }

    %% 관계
    Event --> EventType
    Event --> EventStatus
    EventManager --> Event
    EventManager --> EventLogger
    EventLogger --> Event
```

### 6. 데이터 구조 클래스 다이어그램

```mermaid
classDiagram
    class Person {
        +person_id: int
        +bbox: Tuple[float, float, float, float]
        +keypoints: np.ndarray
        +score: float
        +track_id: Optional[int]
        +timestamp: float
        +metadata: Dict[str, Any]
        +__init__(...)
        +get_center_point(): Tuple[float, float]
        +get_bbox_area(): float
        +get_keypoint(index: int): Tuple[float, float, float]
        +is_valid(): bool
        +calculate_pose_angles(): Dict[str, float]
        +get_limb_vectors(): Dict[str, np.ndarray]
        +to_dict(): Dict[str, Any]
        +from_dict(data: Dict[str, Any]): Person
    }

    class FramePoses {
        +frame_idx: int
        +persons: List[Person]
        +timestamp: float
        +image_shape: Tuple[int, int]
        +metadata: Dict[str, Any]
        +__init__(...)
        +get_valid_persons(): List[Person]
        +get_person_count(): int
        +get_person_by_track_id(track_id: int): Optional[Person]
        +filter_by_confidence(threshold: float): FramePoses
        +to_dict(): Dict[str, Any]
        +from_dict(data: Dict[str, Any]): FramePoses
    }

    class VisualizationData {
        +video_name: str
        +frame_data: List[FramePoses]
        +stage_info: Dict[str, Any]
        +poses_only: Optional[List[FramePoses]]
        +poses_with_tracking: Optional[List[FramePoses]]
        +tracking_info: Optional[Dict[str, Any]]
        +poses_with_scores: Optional[List[FramePoses]]
        +scoring_info: Optional[Dict[str, Any]]
        +classification_results: Optional[List[ClassificationResult]]
        +__init__(...)
        +get_stage_data(stage: str): Optional[List[FramePoses]]
        +get_total_frames(): int
        +get_video_duration(): float
        +to_dict(): Dict[str, Any]
        +from_dict(data: Dict[str, Any]): VisualizationData
    }

    class WindowAnnotation {
        +window_id: str
        +start_frame: int
        +end_frame: int
        +keypoint_sequence: np.ndarray
        +score_sequence: np.ndarray
        +labels: Dict[str, int]
        +metadata: Dict[str, Any]
        +__init__(...)
        +get_duration(): float
        +get_person_count(): int
        +get_frame_count(): int
        +normalize_keypoints(): np.ndarray
        +to_stgcn_format(): Tuple[np.ndarray, np.ndarray]
        +to_dict(): Dict[str, Any]
        +from_dict(data: Dict[str, Any]): WindowAnnotation
    }

    class ProcessingResult {
        +frame_poses: FramePoses
        +classification_results: Dict[str, ClassificationResult]
        +events: List[Event]
        +performance_metrics: Dict[str, float]
        +overlay_info: Dict[str, Any]
        +timestamp: float
        +__init__(...)
        +has_events(): bool
        +get_primary_classification(): Optional[ClassificationResult]
        +to_dict(): Dict[str, Any]
    }

    %% 관계
    FramePoses --> Person
    VisualizationData --> FramePoses
    WindowAnnotation --> FramePoses
    ProcessingResult --> FramePoses
    ProcessingResult --> ClassificationResult
    ProcessingResult --> Event
```

### 7. 유틸리티 클래스 다이어그램

```mermaid
classDiagram
    class ConfigLoader {
        <<Singleton>>
        -_instance: Optional[ConfigLoader]
        -_config_cache: Dict[str, Dict[str, Any]]
        +load_config(config_path: str): Dict[str, Any]
        +validate_config(config: Dict[str, Any]): bool
        +get_mode_config(config: Dict[str, Any], mode: str): Dict[str, Any]
        +merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]): Dict[str, Any]
        +save_config(config: Dict[str, Any], path: str): bool
        -_validate_required_keys(config: Dict[str, Any], required_keys: List[str]): bool
        -_expand_environment_variables(config: Dict[str, Any]): Dict[str, Any]
    }

    class WindowProcessor {
        -window_size: int
        -window_stride: int
        -max_persons: int
        -coordinate_dimensions: int
        -frames_buffer: List[FramePoses]
        -current_window_id: int
        +__init__(window_size: int, window_stride: int, max_persons: int, coordinate_dimensions: int)
        +add_frame_data(frame_poses: FramePoses): void
        +is_ready(): bool
        +get_window_data(): Tuple[np.ndarray, int]
        +get_buffer_size(): int
        +reset(): void
        +get_current_window_id(): int
        -_convert_to_numpy(frame_poses: FramePoses): np.ndarray
        -_pad_or_truncate_persons(frame_data: np.ndarray): np.ndarray
        -_create_window_tensor(): np.ndarray
    }

    class MultiProcessSplitter {
        -num_processes: int
        -gpus: List[int]
        -chunk_strategy: str
        -process_pool: Optional[ProcessPoolExecutor]
        +__init__(num_processes: int, gpus: List[int], chunk_strategy: str)
        +split_work(work_items: List[Any]): List[List[Any]]
        +execute_parallel(work_function: Callable, work_items: List[Any], **kwargs): List[Any]
        +cleanup(): void
        -_split_by_count(items: List[Any]): List[List[Any]]
        -_split_by_size(items: List[Any]): List[List[Any]]
        -_assign_gpu(process_index: int): int
        -_worker_function(work_chunk: List[Any], gpu_id: int, work_function: Callable, **kwargs): Any
    }

    class ResultSaver {
        -base_output_dir: str
        -naming_config: Dict[str, str]
        -supported_formats: List[str]
        +__init__(base_output_dir: str, naming_config: Dict[str, str])
        +save_stage_result(stage: str, data: Any, video_name: str, config_name: str): str
        +save_classification_results(results: List[ClassificationResult], output_path: str): bool
        +save_evaluation_report(report_data: Dict[str, Any], output_path: str): bool
        +create_directory_structure(dataset_name: str): Dict[str, str]
        -_generate_filename(video_name: str, stage: str, config_name: str, extension: str): str
        -_ensure_directory_exists(directory_path: str): void
        -_save_pkl(data: Any, filepath: str): bool
        -_save_json(data: Dict[str, Any], filepath: str): bool
    }

    class PerformanceTracker {
        -stage_timings: Dict[str, List[float]]
        -frame_counts: Dict[str, int]
        -start_times: Dict[str, float]
        -memory_usage: List[float]
        -gpu_usage: List[float]
        +__init__()
        +start_timing(stage: str): void
        +end_timing(stage: str): float
        +add_frame_count(stage: str, count: int): void
        +track_memory_usage(): void
        +track_gpu_usage(): void
        +get_fps_stats(): Dict[str, float]
        +get_timing_stats(): Dict[str, Dict[str, float]]
        +get_resource_stats(): Dict[str, float]
        +reset(): void
        +generate_report(): Dict[str, Any]
    }

    %% 특별한 관계들
    ConfigLoader --> Dict
    WindowProcessor --> FramePoses
    ResultSaver --> ClassificationResult
    MultiProcessSplitter --> ProcessPoolExecutor
```

## 시퀀스 다이어그램

### 1. Annotation 파이프라인 실행 시퀀스

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant AnnotationMode
    participant SeparatedPipeline
    participant PoseEstimator
    participant Tracker
    participant DatasetCreator
    participant MultiProcessSplitter
    participant ResultSaver

    User->>Main: python main.py --mode annotation.stage1
    Main->>AnnotationMode: execute()

    AnnotationMode->>AnnotationMode: _validate_config()
    AnnotationMode->>MultiProcessSplitter: split_work(video_files)
    MultiProcessSplitter-->>AnnotationMode: work_chunks

    loop For each chunk in parallel
        AnnotationMode->>SeparatedPipeline: execute_stage1(chunk)
        SeparatedPipeline->>PoseEstimator: initialize_model()
        PoseEstimator-->>SeparatedPipeline: model_ready

        loop For each video in chunk
            SeparatedPipeline->>PoseEstimator: estimate_poses(frame)
            PoseEstimator-->>SeparatedPipeline: persons_list
            SeparatedPipeline->>SeparatedPipeline: collect_frame_poses()
        end

        SeparatedPipeline->>ResultSaver: save_stage_result(stage1_data)
        ResultSaver-->>SeparatedPipeline: saved_path
        SeparatedPipeline-->>AnnotationMode: chunk_completed
    end

    AnnotationMode-->>Main: stage1_completed

    Note over User: Stage2 execution
    User->>Main: python main.py --mode annotation.stage2
    Main->>AnnotationMode: execute()

    AnnotationMode->>SeparatedPipeline: execute_stage2()

    loop For each stage1 result
        SeparatedPipeline->>ResultSaver: load_stage_result(stage1_path)
        ResultSaver-->>SeparatedPipeline: stage1_data
        SeparatedPipeline->>Tracker: initialize()

        loop For each frame_poses
            SeparatedPipeline->>Tracker: update_tracks(persons)
            Tracker-->>SeparatedPipeline: tracked_persons
        end

        SeparatedPipeline->>ResultSaver: save_stage_result(stage2_data)
    end

    Note over User: Stage3 execution
    User->>Main: python main.py --mode annotation.stage3
    AnnotationMode->>DatasetCreator: create_dataset()

    DatasetCreator->>DatasetCreator: load_all_stage2_results()
    DatasetCreator->>DatasetCreator: convert_to_tensors()
    DatasetCreator->>DatasetCreator: split_dataset(train/val/test)
    DatasetCreator->>ResultSaver: save_final_dataset()

    AnnotationMode-->>Main: annotation_completed
    Main-->>User: All stages completed
```

### 2. 실시간 추론 파이프라인 시퀀스

```mermaid
sequenceDiagram
    participant User
    participant RealtimeMode
    participant DualServicePipeline
    participant InputManager
    participant PoseEstimator
    participant Tracker
    participant WindowProcessor
    participant Classifier
    participant EventManager
    participant Visualizer
    participant EventLogger

    User->>RealtimeMode: execute()
    RealtimeMode->>DualServicePipeline: initialize_pipeline()

    par Initialize modules in parallel
        DualServicePipeline->>PoseEstimator: initialize_model()
        DualServicePipeline->>Tracker: initialize()
        DualServicePipeline->>Classifier: initialize()
        DualServicePipeline->>EventManager: initialize()
    end

    DualServicePipeline->>InputManager: start(input_source)
    InputManager-->>DualServicePipeline: stream_ready

    DualServicePipeline->>Visualizer: start_display()

    loop Main processing loop
        InputManager->>DualServicePipeline: get_frame()
        DualServicePipeline->>PoseEstimator: estimate_poses(frame)
        PoseEstimator-->>DualServicePipeline: persons_list

        DualServicePipeline->>Tracker: update_tracks(persons)
        Tracker-->>DualServicePipeline: tracked_persons

        DualServicePipeline->>WindowProcessor: add_frame_data(tracked_persons)

        alt Window ready for classification
            WindowProcessor-->>DualServicePipeline: window_ready
            DualServicePipeline->>WindowProcessor: get_window_data()
            WindowProcessor-->>DualServicePipeline: window_tensor

            par Classify for each service
                DualServicePipeline->>Classifier: classify_window(fight_data)
                Classifier-->>DualServicePipeline: fight_result
                DualServicePipeline->>Classifier: classify_window(falldown_data)
                Classifier-->>DualServicePipeline: falldown_result
            end

            DualServicePipeline->>EventManager: process_results(classification_results)

            opt Event triggered
                EventManager->>EventLogger: log_event(event_data)
                EventManager->>EventManager: trigger_callbacks()
            end

            EventManager-->>DualServicePipeline: event_data
        end

        DualServicePipeline->>Visualizer: show_frame(frame, poses, results, events)
        Visualizer-->>DualServicePipeline: continue_processing

        alt User requests stop
            DualServicePipeline->>InputManager: stop()
            DualServicePipeline->>Visualizer: stop_display()
            break
        end
    end

    DualServicePipeline->>DualServicePipeline: cleanup()
    DualServicePipeline-->>RealtimeMode: processing_completed
    RealtimeMode-->>User: execution_finished
```

### 3. 분석 모드 배치 처리 시퀀스

```mermaid
sequenceDiagram
    participant User
    participant AnalysisMode
    participant DualServicePipeline
    participant BatchProcessor
    participant EvaluationMetrics
    participant ReportGenerator
    participant MultiProcessSplitter

    User->>AnalysisMode: execute()
    AnalysisMode->>AnalysisMode: _validate_config()
    AnalysisMode->>DualServicePipeline: initialize_pipeline()

    AnalysisMode->>BatchProcessor: prepare_batch(video_files)
    BatchProcessor-->>AnalysisMode: batch_ready

    alt Multi-process enabled
        AnalysisMode->>MultiProcessSplitter: split_work(video_files)
        MultiProcessSplitter-->>AnalysisMode: work_chunks

        par Process chunks in parallel
            loop For each chunk
                AnalysisMode->>DualServicePipeline: process_video_batch(chunk)
                DualServicePipeline-->>AnalysisMode: batch_results
            end
        end
    else Single process
        loop For each video
            AnalysisMode->>DualServicePipeline: process_video(video_path)
            DualServicePipeline-->>AnalysisMode: video_results
        end
    end

    AnalysisMode->>EvaluationMetrics: calculate_metrics(all_results)
    EvaluationMetrics-->>AnalysisMode: performance_metrics

    opt Evaluation enabled
        AnalysisMode->>EvaluationMetrics: create_confusion_matrix()
        AnalysisMode->>EvaluationMetrics: calculate_classification_report()
        AnalysisMode->>ReportGenerator: generate_charts()
        AnalysisMode->>ReportGenerator: generate_final_report()
    end

    AnalysisMode->>ResultSaver: save_analysis_results()
    AnalysisMode-->>User: analysis_completed
```

### 4. 이벤트 관리 상세 시퀀스

```mermaid
sequenceDiagram
    participant Classifier
    participant EventManager
    participant EventLogger
    participant CallbackHandler
    participant AlertSystem

    Classifier->>EventManager: process_classification_result(fight_result)

    EventManager->>EventManager: _check_event_triggers("fight", result)

    alt High confidence detection
        EventManager->>EventManager: increment_consecutive_counter("fight")

        alt Minimum consecutive reached
            EventManager->>EventManager: create_event(FIGHT_START)
            EventManager->>EventLogger: log_event(fight_start_event)
            EventManager->>CallbackHandler: trigger_callbacks(FIGHT_START)
            CallbackHandler->>AlertSystem: send_alert(fight_detected)

            EventManager->>EventManager: set_event_status(ACTIVE)
        else Not enough consecutive
            EventManager->>EventManager: continue_monitoring()
        end

    else Low confidence or normal
        alt Event currently active
            EventManager->>EventManager: increment_normal_counter()

            alt Enough normal frames
                EventManager->>EventManager: create_event(FIGHT_END)
                EventManager->>EventManager: calculate_event_duration()
                EventManager->>EventLogger: log_event(fight_end_event)
                EventManager->>CallbackHandler: trigger_callbacks(FIGHT_END)
                CallbackHandler->>AlertSystem: send_alert(event_resolved)

                EventManager->>EventManager: set_event_status(COMPLETED)
                EventManager->>EventManager: start_cooldown_period()
            else Continue monitoring
                EventManager->>EventManager: maintain_active_status()
            end
        end
    end

    EventManager->>EventManager: _cleanup_expired_events()
    EventManager-->>Classifier: event_processing_complete
```

## 상태 다이어그램

### 1. 이벤트 상태 전이도

```mermaid
stateDiagram-v2
    [*] --> IDLE : System Start

    IDLE --> DETECTING : Classification Confidence > Alert Threshold

    DETECTING --> ACTIVE : Consecutive Detections >= Min Count
    DETECTING --> IDLE : Confidence < Normal Threshold
    DETECTING --> DETECTING : Confidence Fluctuating

    ACTIVE --> ENDING : Confidence < Normal Threshold
    ACTIVE --> ACTIVE : Confidence Maintains High
    ACTIVE --> COMPLETED : Max Duration Reached

    ENDING --> ACTIVE : Confidence > Alert Threshold Again
    ENDING --> COMPLETED : Normal Count >= Min Consecutive

    COMPLETED --> IDLE : Cooldown Period Ends
    COMPLETED --> CANCELLED : System Reset

    state IDLE {
        [*] --> Monitoring
        Monitoring --> [*]
        note right of Monitoring : No events detected\nContinuous monitoring active
    }

    state DETECTING {
        [*] --> CountingConsecutive
        CountingConsecutive --> ValidationCheck
        ValidationCheck --> CountingConsecutive : Not enough consecutive
        ValidationCheck --> [*] : Threshold met
        note right of CountingConsecutive : Building confidence\nTracking consecutive detections
    }

    state ACTIVE {
        [*] --> AlertSent
        AlertSent --> OngoingMonitoring
        OngoingMonitoring --> OngoingAlert : Interval reached
        OngoingAlert --> OngoingMonitoring
        OngoingMonitoring --> [*]
        note right of AlertSent : Event confirmed\nAlerts active\nLogging enabled
    }

    state ENDING {
        [*] --> NormalDetection
        NormalDetection --> ValidationNormal
        ValidationNormal --> NormalDetection : Continue normal
        ValidationNormal --> [*] : Sufficient normal frames
        note right of NormalDetection : Event ending\nVerifying resolution
    }

    state COMPLETED {
        [*] --> EventClosed
        EventClosed --> CooldownActive
        CooldownActive --> [*] : Cooldown finished
        note right of EventClosed : Event completed\nIn cooldown period
    }
```

### 2. 파이프라인 실행 상태

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING : Start Pipeline

    INITIALIZING --> READY : All Modules Loaded Successfully
    INITIALIZING --> ERROR : Initialization Failed

    READY --> PROCESSING : Start Video Processing
    READY --> SHUTDOWN : User Stop Request

    PROCESSING --> PROCESSING : Process Next Frame
    PROCESSING --> PAUSED : Pause Request
    PROCESSING --> COMPLETED : Video End Reached
    PROCESSING --> ERROR : Processing Error Occurred

    PAUSED --> PROCESSING : Resume Request
    PAUSED --> SHUTDOWN : Stop Request

    COMPLETED --> READY : Process Next Video
    COMPLETED --> SHUTDOWN : No More Videos

    ERROR --> RECOVERING : Error Recovery Initiated
    ERROR --> SHUTDOWN : Critical Error / User Abort

    RECOVERING --> PROCESSING : Recovery Successful
    RECOVERING --> ERROR : Recovery Failed
    RECOVERING --> SHUTDOWN : Max Retries Exceeded

    SHUTDOWN --> [*] : Resources Cleaned Up

    state INITIALIZING {
        [*] --> LoadingModels
        LoadingModels --> AllocatingMemory
        AllocatingMemory --> ValidatingConfig
        ValidatingConfig --> [*]
        note right of LoadingModels : Loading pose estimation\nInitializing trackers\nSetting up classifiers
    }

    state PROCESSING {
        [*] --> FrameCapture
        FrameCapture --> PoseEstimation
        PoseEstimation --> Tracking
        Tracking --> Classification
        Classification --> EventProcessing
        EventProcessing --> Visualization
        Visualization --> [*]
        note right of FrameCapture : Real-time frame processing\nEnd-to-end pipeline execution
    }

    state ERROR {
        [*] --> ErrorAnalysis
        ErrorAnalysis --> LoggingError
        LoggingError --> DetermineRecovery
        DetermineRecovery --> [*]
        note right of ErrorAnalysis : Analyzing failure cause\nDetermining recovery strategy
    }
```

### 3. 모듈 생명주기 상태

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED : Module Created

    UNINITIALIZED --> INITIALIZING : Initialize Called

    INITIALIZING --> READY : Initialization Success
    INITIALIZING --> FAILED : Initialization Failed

    READY --> PROCESSING : Process Request
    READY --> CONFIGURING : Config Update
    READY --> TERMINATING : Cleanup Called

    PROCESSING --> READY : Processing Complete
    PROCESSING --> ERROR : Processing Failed

    CONFIGURING --> READY : Config Applied
    CONFIGURING --> FAILED : Invalid Config

    ERROR --> RECOVERING : Auto Recovery
    ERROR --> FAILED : Recovery Impossible
    ERROR --> TERMINATING : Manual Shutdown

    RECOVERING --> READY : Recovery Success
    RECOVERING --> FAILED : Recovery Failed

    FAILED --> INITIALIZING : Reinitialize
    FAILED --> TERMINATING : Give Up

    TERMINATING --> TERMINATED : Cleanup Complete

    TERMINATED --> [*] : Module Destroyed

    state PROCESSING {
        state choice_state <<choice>>
        [*] --> choice_state
        choice_state --> CPUProcessing : CPU Task
        choice_state --> GPUProcessing : GPU Task
        choice_state --> AsyncProcessing : Async Task

        CPUProcessing --> [*]
        GPUProcessing --> [*]
        AsyncProcessing --> [*]
    }
```

## 컴포넌트 다이어그램

### 1. 전체 시스템 컴포넌트 구조

```mermaid
graph TB
    subgraph "Application Layer"
        A1[Main Application]
        A2[Mode Controllers]
        A3[Configuration Manager]
    end

    subgraph "Pipeline Layer"
        P1[Dual Service Pipeline]
        P2[Separated Pipeline]
        P3[Analysis Pipeline]
    end

    subgraph "Processing Layer"
        PR1[Pose Estimation]
        PR2[Object Tracking]
        PR3[Action Classification]
        PR4[Scoring & Filtering]
    end

    subgraph "Event Management Layer"
        E1[Event Manager]
        E2[Event Logger]
        E3[Alert System]
    end

    subgraph "Data Layer"
        D1[Data Structures]
        D2[Result Storage]
        D3[Configuration Data]
    end

    subgraph "Utility Layer"
        U1[Multi-processing]
        U2[Performance Tracking]
        U3[Visualization]
        U4[File I/O]
    end

    subgraph "External Dependencies"
        EXT1[OpenCV]
        EXT2[PyTorch/ONNX]
        EXT3[MMPose/MMAction]
        EXT4[NumPy/SciPy]
    end

    %% Connections
    A1 --> A2
    A2 --> P1
    A2 --> P2
    A2 --> P3
    A3 --> A2

    P1 --> PR1
    P1 --> PR2
    P1 --> PR3
    P1 --> PR4
    P1 --> E1

    P2 --> PR1
    P2 --> PR2
    P2 --> D2

    P3 --> P1
    P3 --> U2

    E1 --> E2
    E1 --> E3

    PR1 --> D1
    PR2 --> D1
    PR3 --> D1
    PR4 --> D1

    U1 --> P2
    U1 --> P3
    U3 --> P1
    U4 --> D2

    PR1 --> EXT1
    PR1 --> EXT2
    PR1 --> EXT3
    PR2 --> EXT3
    PR3 --> EXT2
    PR3 --> EXT3

    D1 --> EXT4
    D2 --> EXT4
    U2 --> EXT4
```

### 2. 실시간 처리 컴포넌트 상세

```mermaid
graph TB
    subgraph "Input Management"
        IM1[Video Capture]
        IM2[RTSP Client]
        IM3[Frame Buffer]
        IM4[Input Manager]
    end

    subgraph "Core Processing Pipeline"
        CP1[Pose Estimator]
        CP2[Tracker]
        CP3[Window Processor]
        CP4[Classifier Queue]
        CP5[Async Classifier]
    end

    subgraph "Event Processing"
        EP1[Event Manager]
        EP2[Event State Machine]
        EP3[Alert Generator]
        EP4[Event Logger]
    end

    subgraph "Output & Visualization"
        OV1[Real-time Visualizer]
        OV2[Overlay Renderer]
        OV3[Display Manager]
        OV4[Performance HUD]
    end

    subgraph "Resource Management"
        RM1[GPU Memory Pool]
        RM2[Thread Pool]
        RM3[Performance Monitor]
        RM4[Resource Scheduler]
    end

    %% Data Flow
    IM1 --> IM3
    IM2 --> IM3
    IM3 --> IM4
    IM4 --> CP1

    CP1 --> CP2
    CP2 --> CP3
    CP3 --> CP4
    CP4 --> CP5

    CP5 --> EP1
    EP1 --> EP2
    EP2 --> EP3
    EP3 --> EP4

    CP1 --> OV1
    CP2 --> OV1
    EP1 --> OV1
    OV1 --> OV2
    OV2 --> OV3
    RM3 --> OV4
    OV4 --> OV3

    %% Resource Management
    RM1 --> CP1
    RM1 --> CP5
    RM2 --> CP5
    RM3 --> RM4
    RM4 --> RM1
    RM4 --> RM2
```

### 3. 데이터 처리 파이프라인 컴포넌트

```mermaid
graph LR
    subgraph "Stage 1: Pose Extraction"
        S1_1[Video Loader]
        S1_2[Frame Preprocessor]
        S1_3[RTMO Estimator]
        S1_4[Pose Postprocessor]
        S1_5[PKL Saver]
    end

    subgraph "Stage 2: Tracking & Scoring"
        S2_1[PKL Loader]
        S2_2[ByteTracker]
        S2_3[Motion Scorer]
        S2_4[Quality Filter]
        S2_5[Tracking PKL Saver]
    end

    subgraph "Stage 3: Dataset Creation"
        S3_1[Multi PKL Loader]
        S3_2[Tensor Converter]
        S3_3[Dataset Splitter]
        S3_4[STGCN Formatter]
        S3_5[Dataset Saver]
    end

    subgraph "Support Components"
        SC1[Multi-process Manager]
        SC2[GPU Distributor]
        SC3[Progress Tracker]
        SC4[Error Handler]
        SC5[Config Validator]
    end

    %% Stage 1 Flow
    S1_1 --> S1_2
    S1_2 --> S1_3
    S1_3 --> S1_4
    S1_4 --> S1_5

    %% Stage 2 Flow
    S1_5 --> S2_1
    S2_1 --> S2_2
    S2_2 --> S2_3
    S2_3 --> S2_4
    S2_4 --> S2_5

    %% Stage 3 Flow
    S2_5 --> S3_1
    S3_1 --> S3_2
    S3_2 --> S3_3
    S3_3 --> S3_4
    S3_4 --> S3_5

    %% Support Connections
    SC1 --> S1_1
    SC1 --> S2_1
    SC1 --> S3_1
    SC2 --> S1_3
    SC3 --> S1_5
    SC3 --> S2_5
    SC3 --> S3_5
    SC4 --> S1_3
    SC4 --> S2_2
    SC4 --> S3_2
    SC5 --> S1_1
    SC5 --> S2_1
    SC5 --> S3_1
```

## 배포 다이어그램

### 1. Docker 컨테이너 배포 구조

```mermaid
graph TB
    subgraph "Host Environment"
        subgraph "Docker Container: mmlabs"
            subgraph "Workspace"
                WS1[/workspace/recognizer]
                WS2[/workspace/mmaction2]
                WS3[/workspace/mmpose]
            end

            subgraph "Python Environment"
                PE1[PyTorch 2.0+]
                PE2[MMPose Framework]
                PE3[MMAction2 Framework]
                PE4[ONNX Runtime]
                PE5[TensorRT]
            end

            subgraph "System Libraries"
                SL1[CUDA 11.8]
                SL2[cuDNN]
                SL3[OpenCV]
                SL4[FFmpeg]
            end
        end

        subgraph "Host Resources"
            HR1[NVIDIA GPU]
            HR2[CPU Cores]
            HR3[System Memory]
            HR4[Storage]
        end

        subgraph "Data Volumes"
            DV1[Input Videos]
            DV2[Model Checkpoints]
            DV3[Output Results]
            DV4[Configuration Files]
        end
    end

    %% Connections
    WS1 --> PE1
    WS1 --> PE2
    WS1 --> PE3
    WS2 --> PE3
    WS3 --> PE2

    PE1 --> SL1
    PE2 --> SL1
    PE3 --> SL1
    PE4 --> SL1
    PE5 --> SL1

    SL1 --> HR1
    SL2 --> HR1
    SL3 --> HR2
    SL4 --> HR2

    WS1 --> DV1
    WS1 --> DV2
    WS1 --> DV3
    WS1 --> DV4

    style "Docker Container: mmlabs" fill:#e1f5fe
    style "Host Environment" fill:#f3e5f5
```

### 2. 멀티 GPU 분산 처리 구조

```mermaid
graph TB
    subgraph "Master Process"
        MP1[Main Controller]
        MP2[Task Distributor]
        MP3[Result Aggregator]
    end

    subgraph "GPU 0 Process"
        GP1_1[Process Manager 0]
        GP1_2[RTMO Estimator]
        GP1_3[STGCN Classifier]
        GP1_4[GPU Memory Manager]
    end

    subgraph "GPU 1 Process"
        GP2_1[Process Manager 1]
        GP2_2[RTMO Estimator]
        GP2_3[STGCN Classifier]
        GP2_4[GPU Memory Manager]
    end

    subgraph "Shared Resources"
        SR1[Video Queue]
        SR2[Result Queue]
        SR3[Configuration]
        SR4[Progress Monitor]
    end

    subgraph "Hardware Layer"
        HW1[NVIDIA GPU 0]
        HW2[NVIDIA GPU 1]
        HW3[System Memory]
        HW4[NVMe Storage]
    end

    %% Process Communication
    MP1 --> MP2
    MP2 --> GP1_1
    MP2 --> GP2_1
    GP1_1 --> MP3
    GP2_1 --> MP3

    %% Shared Resource Access
    MP2 --> SR1
    GP1_1 --> SR1
    GP2_1 --> SR1
    GP1_1 --> SR2
    GP2_1 --> SR2
    MP3 --> SR2

    %% Hardware Mapping
    GP1_2 --> HW1
    GP1_3 --> HW1
    GP1_4 --> HW1
    GP2_2 --> HW2
    GP2_3 --> HW2
    GP2_4 --> HW2

    %% Storage Access
    SR1 --> HW4
    SR2 --> HW4
    SR3 --> HW3
    SR4 --> HW3
```

### 3. 실시간 스트리밍 배포 구조

```mermaid
graph TB
    subgraph "Input Sources"
        IS1[IP Camera 1]
        IS2[IP Camera 2]
        IS3[RTSP Server]
        IS4[Local Files]
    end

    subgraph "Processing Node"
        subgraph "Input Layer"
            IL1[RTSP Client]
            IL2[Frame Buffer]
            IL3[Load Balancer]
        end

        subgraph "Inference Layer"
            INF1[Pose Pipeline]
            INF2[Classification Pipeline]
            INF3[Event Pipeline]
        end

        subgraph "Output Layer"
            OL1[Real-time Viewer]
            OL2[Event Logger]
            OL3[Alert System]
        end
    end

    subgraph "Storage & Monitoring"
        SM1[Event Database]
        SM2[Performance Metrics]
        SM3[Log Files]
        SM4[Model Checkpoints]
    end

    subgraph "External Systems"
        ES1[Notification Service]
        ES2[Dashboard Web UI]
        ES3[Mobile App]
        ES4[Security System]
    end

    %% Input Connections
    IS1 --> IL1
    IS2 --> IL1
    IS3 --> IL1
    IS4 --> IL2

    %% Processing Flow
    IL1 --> IL2
    IL2 --> IL3
    IL3 --> INF1
    INF1 --> INF2
    INF2 --> INF3

    %% Output Flow
    INF1 --> OL1
    INF2 --> OL1
    INF3 --> OL2
    INF3 --> OL3

    %% Storage Connections
    OL2 --> SM1
    INF1 --> SM2
    INF2 --> SM2
    OL2 --> SM3
    INF1 --> SM4
    INF2 --> SM4

    %% External Integrations
    OL3 --> ES1
    OL1 --> ES2
    ES1 --> ES3
    OL3 --> ES4
```

## 액티비티 다이어그램

### 1. Annotation 모드 전체 워크플로우

```mermaid
graph TD
    Start([시작]) --> LoadConfig[설정 파일 로드]
    LoadConfig --> ValidateConfig{설정 검증}
    ValidateConfig -->|실패| ConfigError[설정 오류 출력]
    ConfigError --> End([종료])
    ValidateConfig -->|성공| CheckMode{모드 확인}

    CheckMode -->|annotation.stage1| Stage1Flow
    CheckMode -->|annotation.stage2| Stage2Flow
    CheckMode -->|annotation.stage3| Stage3Flow
    CheckMode -->|annotation.visualize| VisualizeFlow

    subgraph Stage1Flow [Stage1: 포즈 추정]
        S1_Start[Stage1 시작] --> S1_ScanVideos[비디오 파일 스캔]
        S1_ScanVideos --> S1_CheckMultiProcess{멀티프로세싱 활성화?}
        S1_CheckMultiProcess -->|예| S1_SplitWork[작업 분할]
        S1_SplitWork --> S1_ParallelProcess[병렬 처리]
        S1_CheckMultiProcess -->|아니오| S1_SequentialProcess[순차 처리]
        S1_ParallelProcess --> S1_Aggregate[결과 집계]
        S1_SequentialProcess --> S1_Aggregate
        S1_Aggregate --> S1_SaveResults[Stage1 결과 저장]
        S1_SaveResults --> S1_End[Stage1 완료]
    end

    subgraph Stage2Flow [Stage2: 추적 및 스코어링]
        S2_Start[Stage2 시작] --> S2_LoadStage1[Stage1 결과 로드]
        S2_LoadStage1 --> S2_InitTracker[추적기 초기화]
        S2_InitTracker --> S2_ProcessFrames[프레임별 추적]
        S2_ProcessFrames --> S2_ScoreMotion[모션 스코어링]
        S2_ScoreMotion --> S2_SaveResults[Stage2 결과 저장]
        S2_SaveResults --> S2_End[Stage2 완료]
    end

    subgraph Stage3Flow [Stage3: 데이터셋 생성]
        S3_Start[Stage3 시작] --> S3_LoadStage2[Stage2 결과 로드]
        S3_LoadStage2 --> S3_ConvertTensor[텐서 변환]
        S3_ConvertTensor --> S3_SplitDataset[데이터셋 분할]
        S3_SplitDataset --> S3_SaveDataset[최종 데이터셋 저장]
        S3_SaveDataset --> S3_End[Stage3 완료]
    end

    subgraph VisualizeFlow [Visualize: 결과 시각화]
        V_Start[Visualize 시작] --> V_SelectStage{Stage 선택}
        V_SelectStage -->|stage1| V_LoadStage1[Stage1 PKL 로드]
        V_SelectStage -->|stage2| V_LoadStage2[Stage2 PKL 로드]
        V_LoadStage1 --> V_RenderVideo[오버레이 비디오 렌더링]
        V_LoadStage2 --> V_RenderVideo
        V_RenderVideo --> V_SaveOrDisplay{저장/표시 모드}
        V_SaveOrDisplay -->|저장| V_SaveVideo[비디오 파일 저장]
        V_SaveOrDisplay -->|표시| V_DisplayVideo[실시간 표시]
        V_SaveVideo --> V_End[Visualize 완료]
        V_DisplayVideo --> V_End
    end

    Stage1Flow --> CheckContinue1{다음 단계 실행?}
    CheckContinue1 -->|예| Stage2Flow
    CheckContinue1 -->|아니오| End

    Stage2Flow --> CheckContinue2{다음 단계 실행?}
    CheckContinue2 -->|예| Stage3Flow
    CheckContinue2 -->|아니오| End

    Stage3Flow --> End
    VisualizeFlow --> End
```

### 2. 실시간 처리 상세 워크플로우

```mermaid
graph TD
    Start([시작]) --> InitPipeline[파이프라인 초기화]
    InitPipeline --> LoadModels[모델 로드]
    LoadModels --> StartInput[입력 스트림 시작]
    StartInput --> StartDisplay[화면 표시 시작]
    StartDisplay --> MainLoop{메인 루프}

    MainLoop --> GetFrame[프레임 가져오기]
    GetFrame --> FrameValid{프레임 유효?}
    FrameValid -->|아니오| CheckStop{종료 요청?}
    FrameValid -->|예| EstimatePose[포즈 추정]

    EstimatePose --> PoseValid{포즈 유효?}
    PoseValid -->|아니오| SkipFrame[프레임 스킵]
    SkipFrame --> DisplayFrame
    PoseValid -->|예| TrackObjects[객체 추적]

    TrackObjects --> ScoreMotion[모션 스코어링]
    ScoreMotion --> UpdateWindow[윈도우 업데이트]
    UpdateWindow --> WindowReady{윈도우 준비됨?}

    WindowReady -->|아니오| DisplayFrame[프레임 표시]
    WindowReady -->|예| ClassifyAction[동작 분류]

    ClassifyAction --> ProcessEvents[이벤트 처리]
    ProcessEvents --> LogEvents{이벤트 발생?}
    LogEvents -->|예| TriggerAlert[알림 발생]
    LogEvents -->|아니오| DisplayFrame
    TriggerAlert --> DisplayFrame

    DisplayFrame --> CheckStop
    CheckStop -->|아니오| MainLoop
    CheckStop -->|예| Cleanup[정리 작업]
    Cleanup --> SaveLogs[로그 저장]
    SaveLogs --> End([종료])

    subgraph "병렬 분류 처리"
        ClassifyAction --> QueueWindow[윈도우 큐에 추가]
        QueueWindow --> AsyncClassify[비동기 분류]
        AsyncClassify --> ReturnResult[결과 반환]
        ReturnResult --> ProcessEvents
    end
```

### 3. 이벤트 처리 상세 워크플로우

```mermaid
graph TD
    Start([분류 결과 수신]) --> CheckConfidence{신뢰도 검사}
    CheckConfidence -->|낮음| UpdateNormalCounter[정상 카운터 증가]
    CheckConfidence -->|높음| UpdateViolenceCounter[폭력 카운터 증가]

    UpdateViolenceCounter --> CheckConsecutive{연속 감지 충분?}
    CheckConsecutive -->|아니오| ResetNormalCounter[정상 카운터 리셋]
    ResetNormalCounter --> EndProcess([처리 종료])
    CheckConsecutive -->|예| CheckEventActive{이벤트 활성?}

    CheckEventActive -->|아니오| CreateEvent[새 이벤트 생성]
    CreateEvent --> SetEventActive[이벤트 활성화]
    SetEventActive --> LogEventStart[이벤트 시작 로그]
    LogEventStart --> TriggerCallbacks[콜백 실행]
    TriggerCallbacks --> SendAlert[알림 전송]
    SendAlert --> EndProcess

    CheckEventActive -->|예| UpdateEventConfidence[이벤트 신뢰도 업데이트]
    UpdateEventConfidence --> CheckOngoingAlert{진행중 알림 필요?}
    CheckOngoingAlert -->|예| SendOngoingAlert[진행중 알림 전송]
    CheckOngoingAlert -->|아니오| EndProcess
    SendOngoingAlert --> EndProcess

    UpdateNormalCounter --> CheckEventActive2{이벤트 활성?}
    CheckEventActive2 -->|아니오| EndProcess
    CheckEventActive2 -->|예| CheckNormalConsecutive{연속 정상 충분?}
    CheckNormalConsecutive -->|아니오| EndProcess
    CheckNormalConsecutive -->|예| EndEvent[이벤트 종료]

    EndEvent --> CalculateDuration[지속 시간 계산]
    CalculateDuration --> LogEventEnd[이벤트 종료 로그]
    LogEventEnd --> TriggerEndCallbacks[종료 콜백 실행]
    TriggerEndCallbacks --> StartCooldown[쿨다운 시작]
    StartCooldown --> ResetCounters[카운터 리셋]
    ResetCounters --> EndProcess

    subgraph "이벤트 상태 관리"
        SetEventActive --> MonitorDuration[지속시간 모니터링]
        MonitorDuration --> CheckMaxDuration{최대 시간 초과?}
        CheckMaxDuration -->|예| ForceEndEvent[강제 종료]
        ForceEndEvent --> EndEvent
        CheckMaxDuration -->|아니오| ContinueMonitoring[모니터링 계속]
        ContinueMonitoring --> MonitorDuration
    end
```

## 유스케이스 다이어그램

### 1. 전체 시스템 유스케이스

```mermaid
graph LR
    subgraph "사용자 유형"
        Researcher[연구자]
        Developer[개발자]
        Operator[운영자]
        Analyst[분석가]
    end

    subgraph "Recognizer 시스템"
        subgraph "데이터 준비"
            UC1[비디오 포즈 추정]
            UC2[객체 추적]
            UC3[학습 데이터셋 생성]
            UC4[결과 시각화]
        end

        subgraph "실시간 처리"
            UC5[실시간 동작 감지]
            UC6[이벤트 모니터링]
            UC7[알림 관리]
            UC8[성능 모니터링]
        end

        subgraph "배치 분석"
            UC9[비디오 배치 분석]
            UC10[성능 평가]
            UC11[분석 보고서 생성]
            UC12[결과 시각화]
        end

        subgraph "시스템 관리"
            UC13[설정 관리]
            UC14[모델 관리]
            UC15[로그 관리]
            UC16[시스템 모니터링]
        end
    end

    %% 연구자 관련 유스케이스
    Researcher --> UC1
    Researcher --> UC2
    Researcher --> UC3
    Researcher --> UC10
    Researcher --> UC11

    %% 개발자 관련 유스케이스
    Developer --> UC1
    Developer --> UC2
    Developer --> UC3
    Developer --> UC4
    Developer --> UC13
    Developer --> UC14
    Developer --> UC15

    %% 운영자 관련 유스케이스
    Operator --> UC5
    Operator --> UC6
    Operator --> UC7
    Operator --> UC8
    Operator --> UC13
    Operator --> UC16

    %% 분석가 관련 유스케이스
    Analyst --> UC9
    Analyst --> UC10
    Analyst --> UC11
    Analyst --> UC12
    Analyst --> UC15

    %% 유스케이스 간 관계
    UC1 --> UC2 : includes
    UC2 --> UC3 : includes
    UC3 --> UC4 : includes
    UC5 --> UC6 : includes
    UC6 --> UC7 : includes
    UC9 --> UC10 : includes
    UC10 --> UC11 : includes
    UC11 --> UC12 : includes
```

### 2. 실시간 처리 상세 유스케이스

```mermaid
graph TB
    subgraph "Primary Actors"
        SecurityOperator[보안 운영자]
        SystemAdmin[시스템 관리자]
    end

    subgraph "Secondary Actors"
        Camera[IP 카메라]
        AlertSystem[알림 시스템]
        Database[데이터베이스]
        LoggingSystem[로깅 시스템]
    end

    subgraph "Real-time Processing Use Cases"
        UC1[스트림 입력 설정]
        UC2[실시간 포즈 추정]
        UC3[객체 추적 수행]
        UC4[동작 분류 실행]
        UC5[이벤트 감지]
        UC6[실시간 알림 전송]
        UC7[화면 표시]
        UC8[이벤트 로깅]
        UC9[성능 모니터링]
        UC10[시스템 설정 변경]
        UC11[처리 중단/재시작]
        UC12[결과 내보내기]
    end

    %% Primary Actor 관계
    SecurityOperator --> UC1
    SecurityOperator --> UC5
    SecurityOperator --> UC6
    SecurityOperator --> UC7
    SecurityOperator --> UC8
    SecurityOperator --> UC11
    SecurityOperator --> UC12

    SystemAdmin --> UC9
    SystemAdmin --> UC10
    SystemAdmin --> UC11

    %% Secondary Actor 관계
    Camera --> UC1
    AlertSystem --> UC6
    Database --> UC8
    LoggingSystem --> UC8
    LoggingSystem --> UC9

    %% Include 관계
    UC1 -.-> UC2 : includes
    UC2 -.-> UC3 : includes
    UC3 -.-> UC4 : includes
    UC4 -.-> UC5 : includes
    UC5 -.-> UC6 : includes
    UC5 -.-> UC8 : includes

    %% Extend 관계
    UC6 -.-> UC6_EXT[다중 채널 알림] : extends
    UC8 -.-> UC8_EXT[상세 로그 분석] : extends
    UC9 -.-> UC9_EXT[자동 최적화] : extends

    style UC6_EXT fill:#e8f4f8
    style UC8_EXT fill:#e8f4f8
    style UC9_EXT fill:#e8f4f8
```

### 3. 데이터 처리 파이프라인 유스케이스

```mermaid
graph TB
    subgraph "Primary Actors"
        DataScientist[데이터 과학자]
        MLEngineer[ML 엔지니어]
        ResearchStudent[연구생]
    end

    subgraph "Data Processing Use Cases"
        subgraph "Stage 1 Use Cases"
            UC1_1[비디오 데이터 로드]
            UC1_2[배치 포즈 추정]
            UC1_3[멀티프로세싱 실행]
            UC1_4[GPU 분산 처리]
            UC1_5[Stage1 결과 저장]
        end

        subgraph "Stage 2 Use Cases"
            UC2_1[Stage1 결과 로드]
            UC2_2[객체 추적 실행]
            UC2_3[모션 스코어링]
            UC2_4[품질 필터링]
            UC2_5[Stage2 결과 저장]
        end

        subgraph "Stage 3 Use Cases"
            UC3_1[Stage2 결과 집계]
            UC3_2[텐서 형식 변환]
            UC3_3[데이터셋 분할]
            UC3_4[STGCN 호환 형식 생성]
            UC3_5[최종 데이터셋 저장]
        end

        subgraph "Visualization Use Cases"
            UC4_1[PKL 결과 로드]
            UC4_2[오버레이 생성]
            UC4_3[시각화 렌더링]
            UC4_4[비디오 저장/표시]
        end

        subgraph "Quality Assurance Use Cases"
            UC5_1[처리 상태 모니터링]
            UC5_2[오류 감지 및 복구]
            UC5_3[품질 검증]
            UC5_4[성능 최적화]
        end
    end

    %% Actor 관계
    DataScientist --> UC1_1
    DataScientist --> UC2_1
    DataScientist --> UC3_1
    DataScientist --> UC4_1
    DataScientist --> UC5_3

    MLEngineer --> UC1_3
    MLEngineer --> UC1_4
    MLEngineer --> UC3_4
    MLEngineer --> UC5_1
    MLEngineer --> UC5_4

    ResearchStudent --> UC1_1
    ResearchStudent --> UC4_1
    ResearchStudent --> UC4_3
    ResearchStudent --> UC5_1

    %% 순차적 관계
    UC1_1 --> UC1_2
    UC1_2 --> UC1_5
    UC1_5 --> UC2_1
    UC2_1 --> UC2_2
    UC2_2 --> UC2_3
    UC2_3 --> UC2_4
    UC2_4 --> UC2_5
    UC2_5 --> UC3_1
    UC3_1 --> UC3_2
    UC3_2 --> UC3_3
    UC3_3 --> UC3_4
    UC3_4 --> UC3_5

    %% Include 관계
    UC1_2 -.-> UC1_3 : includes
    UC1_3 -.-> UC1_4 : includes
    UC4_1 -.-> UC4_2 : includes
    UC4_2 -.-> UC4_3 : includes

    %% Extend 관계
    UC5_2 -.-> UC5_2_EXT[자동 재시작] : extends
    UC5_4 -.-> UC5_4_EXT[하이퍼파라미터 튜닝] : extends

    style UC5_2_EXT fill:#fff2cc
    style UC5_4_EXT fill:#fff2cc
```

이 포괄적인 UML 다이어그램 문서는 Recognizer 시스템의 모든 측면을 시각적으로 문서화합니다. 각 다이어그램은 시스템의 다른 관점을 제공하며, 개발자들이 시스템을 이해하고 확장하는 데 도움이 됩니다.

## 주요 특징

1. **포괄적 커버리지**: 클래스 구조부터 배포까지 모든 아키텍처 측면 포함
2. **Mermaid 형식**: GitHub에서 바로 렌더링 가능한 다이어그램
3. **계층적 구조**: 전체 시스템부터 세부 모듈까지 점진적 상세화
4. **실제 코드 반영**: 실제 구현된 클래스와 메서드명 사용
5. **확장 가능성**: 새로운 모듈 추가 시 참조 가능한 패턴 제공