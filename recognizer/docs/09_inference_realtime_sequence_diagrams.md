# Inference.Realtime 모드 시퀀스 다이어그램

## 개요

이 문서는 inference.realtime 모드의 상세한 시퀀스 다이어그램을 ONNX와 PyTorch 적용 케이스로 구분하여 제공합니다.

## ONNX 적용 시 시퀀스 다이어그램

### 1. 시스템 초기화 및 파이프라인 설정

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Main as main.py
    participant ModeManager as ModeManager
    participant RealtimeMode as RealtimeMode
    participant DualPipeline as DualServicePipeline
    participant Factory as ModuleFactory
    participant RTMOEstimator as RTMOONNXEstimator
    participant ByteTracker as ByteTracker
    participant WindowProcessor as SlidingWindowProcessor
    participant FightClassifier as STGCNClassifier(Fight)
    participant FalldownClassifier as STGCNClassifier(Falldown)

    User->>Main: python main.py --mode inference.realtime
    Main->>ModeManager: execute("inference.realtime")
    ModeManager->>RealtimeMode: execute()

    Note over RealtimeMode: Config 검증 및 설정 로드
    RealtimeMode->>RealtimeMode: _validate_config(['input'])
    RealtimeMode->>RealtimeMode: dual_config = config.get('dual_service')

    RealtimeMode->>DualPipeline: create_dual_service_pipeline(config)
    DualPipeline->>DualPipeline: __init__(config)
    DualPipeline->>DualPipeline: initialize_pipeline()

    Note over DualPipeline: 공통 모듈 초기화
    DualPipeline->>DualPipeline: _initialize_common_modules()
    DualPipeline->>Factory: create_pose_estimator('rtmo_onnx', config)
    Factory->>RTMOEstimator: RTMOONNXEstimator(config)
    RTMOEstimator->>RTMOEstimator: initialize_model()
    RTMOEstimator-->>Factory: estimator 객체
    Factory-->>DualPipeline: pose_estimator

    DualPipeline->>Factory: create_tracker('bytetrack', config)
    Factory->>ByteTracker: ByteTracker(config)
    ByteTracker->>ByteTracker: initialize_tracker()
    ByteTracker-->>Factory: tracker 객체
    Factory-->>DualPipeline: tracker

    DualPipeline->>Factory: create_window_processor('sliding_window', config)
    Factory->>WindowProcessor: SlidingWindowProcessor(config)
    WindowProcessor-->>Factory: processor 객체
    Factory-->>DualPipeline: window_processor

    Note over DualPipeline: 서비스별 모듈 초기화
    DualPipeline->>DualPipeline: _initialize_service_modules()

    loop services in ['fight', 'falldown']
        DualPipeline->>Factory: create_scorer(service, config)
        Factory-->>DualPipeline: scorer[service]

        DualPipeline->>Factory: create_classifier('stgcn', config)
        Factory->>FightClassifier: STGCNClassifier(fight_config) [ONNX]
        FightClassifier->>FightClassifier: initialize_model()
        FightClassifier->>FightClassifier: load_onnx_model()
        FightClassifier-->>Factory: fight_classifier
        Factory-->>DualPipeline: classifiers['fight']

        DualPipeline->>Factory: create_classifier('stgcn', config)
        Factory->>FalldownClassifier: STGCNClassifier(falldown_config) [ONNX]
        FalldownClassifier->>FalldownClassifier: initialize_model()
        FalldownClassifier->>FalldownClassifier: load_onnx_model()
        FalldownClassifier-->>Factory: falldown_classifier
        Factory-->>DualPipeline: classifiers['falldown']
    end

    DualPipeline-->>RealtimeMode: pipeline 객체
```

### 2. 실시간 처리 루프 (ONNX 기반)

```mermaid
sequenceDiagram
    participant RealtimeMode as RealtimeMode
    participant DualPipeline as DualServicePipeline
    participant InputManager as RealtimeInputManager
    participant RTMOEstimator as RTMOONNXEstimator
    participant ONNXRuntime as ONNX Runtime
    participant ByteTracker as ByteTracker
    participant WindowProcessor as SlidingWindowProcessor
    participant FightClassifier as STGCNClassifier(Fight)
    participant FalldownClassifier as STGCNClassifier(Falldown)
    participant Visualizer as InferenceVisualizer
    participant CV2 as OpenCV Display

    RealtimeMode->>DualPipeline: start_realtime_display()
    DualPipeline->>InputManager: RealtimeInputManager(input_source)
    DualPipeline->>InputManager: start()
    InputManager-->>DualPipeline: capture 시작 완료

    DualPipeline->>Visualizer: InferenceVisualizer()
    DualPipeline->>CV2: cv2.namedWindow()

    loop 실시간 프레임 처리
        DualPipeline->>InputManager: get_latest_frame()
        InputManager-->>DualPipeline: frame, frame_idx, timestamp

        Note over DualPipeline: 프레임 처리 시작
        DualPipeline->>DualPipeline: frame_idx += 1

        Note over DualPipeline,ONNXRuntime: POSE ESTIMATION (ONNX)
        DualPipeline->>RTMOEstimator: process_frame(frame, frame_idx)
        RTMOEstimator->>RTMOEstimator: preprocess_frame(frame)
        RTMOEstimator->>ONNXRuntime: session.run(input_data)
        ONNXRuntime-->>RTMOEstimator: pose_predictions
        RTMOEstimator->>RTMOEstimator: postprocess_results(predictions)
        RTMOEstimator->>RTMOEstimator: convert_to_frame_poses()
        RTMOEstimator-->>DualPipeline: FramePoses 객체

        Note over DualPipeline: TRACKING
        DualPipeline->>ByteTracker: track_frame_poses(frame_poses)
        ByteTracker->>ByteTracker: update_tracks(detections)
        ByteTracker->>ByteTracker: assign_track_ids()
        ByteTracker-->>DualPipeline: tracked_frame_poses

        Note over DualPipeline: WINDOW PROCESSING
        DualPipeline->>WindowProcessor: add_frame(tracked_frame_poses)
        WindowProcessor->>WindowProcessor: check_window_ready()

        alt 윈도우 준비 완료
            WindowProcessor->>WindowProcessor: create_window_annotation()
            WindowProcessor-->>DualPipeline: ready_windows[]

            loop window in ready_windows
                Note over DualPipeline: DUAL SERVICE CLASSIFICATION
                DualPipeline->>DualPipeline: process_window_dual_service(window)

                Note over DualPipeline: Fight Service Processing
                DualPipeline->>DualPipeline: _extract_tracked_poses_from_window()
                DualPipeline->>DualPipeline: calculate_scores(tracked_poses)
                DualPipeline->>DualPipeline: _apply_scores_to_window()

                DualPipeline->>FightClassifier: classify_window(scored_window)
                FightClassifier->>FightClassifier: preprocess_window_data()
                FightClassifier->>ONNXRuntime: session.run(stgcn_input) [Fight ONNX]
                ONNXRuntime-->>FightClassifier: fight_predictions
                FightClassifier->>FightClassifier: postprocess_predictions()
                FightClassifier-->>DualPipeline: fight_classification_result

                Note over DualPipeline: Falldown Service Processing
                DualPipeline->>DualPipeline: _extract_tracked_poses_from_window()
                DualPipeline->>DualPipeline: calculate_scores(tracked_poses)
                DualPipeline->>DualPipeline: _apply_scores_to_window()

                DualPipeline->>FalldownClassifier: classify_window(scored_window)
                FalldownClassifier->>FalldownClassifier: preprocess_window_data()
                FalldownClassifier->>ONNXRuntime: session.run(stgcn_input) [Falldown ONNX]
                ONNXRuntime-->>FalldownClassifier: falldown_predictions
                FalldownClassifier->>FalldownClassifier: postprocess_predictions()
                FalldownClassifier-->>DualPipeline: falldown_classification_result

                DualPipeline->>DualPipeline: create_dual_classification_result()
                DualPipeline->>DualPipeline: latest_classification_result = result
            end
        else 윈도우 미준비
            WindowProcessor-->>DualPipeline: []
        end

        Note over DualPipeline: VISUALIZATION
        DualPipeline->>Visualizer: visualize_frame(frame, poses, classification)
        Visualizer->>Visualizer: draw_keypoints()
        Visualizer->>Visualizer: draw_bboxes()
        Visualizer->>Visualizer: draw_tracking_ids()
        Visualizer->>Visualizer: draw_classification_overlay()
        Visualizer-->>DualPipeline: display_frame

        DualPipeline->>CV2: cv2.imshow(window_name, display_frame)
        DualPipeline->>CV2: cv2.waitKey(1)

        alt 비디오 저장 모드
            DualPipeline->>CV2: video_writer.write(display_frame)
        end

        alt ESC 키 입력 또는 영상 종료
            break 루프 종료
        end
    end

    DualPipeline->>InputManager: stop()
    DualPipeline->>CV2: cv2.destroyAllWindows()
    DualPipeline-->>RealtimeMode: 처리 완료
```

### 3. ONNX 모델 추론 상세 과정

```mermaid
sequenceDiagram
    participant DualPipeline as DualServicePipeline
    participant RTMOEstimator as RTMOONNXEstimator
    participant ONNXSession as ONNX Session
    participant FightClassifier as STGCNClassifier(Fight)
    participant FalldownClassifier as STGCNClassifier(Falldown)

    Note over DualPipeline: 포즈 추정 (RTMO ONNX)
    DualPipeline->>RTMOEstimator: process_frame(frame, frame_idx)
    RTMOEstimator->>RTMOEstimator: validate_frame(frame)
    RTMOEstimator->>RTMOEstimator: preprocess_image(frame)
    Note right of RTMOEstimator: 이미지 리사이징<br/>정규화<br/>채널 순서 변경<br/>(BGR→RGB)
    RTMOEstimator->>RTMOEstimator: prepare_onnx_input(preprocessed)
    Note right of RTMOEstimator: Batch 차원 추가<br/>[1, 3, H, W] 형태로 변환

    RTMOEstimator->>ONNXSession: run(input_feed, output_names)
    Note right of ONNXSession: RTMO ONNX 모델 실행<br/>Bottom-up 포즈 추정
    ONNXSession-->>RTMOEstimator: raw_outputs

    RTMOEstimator->>RTMOEstimator: postprocess_outputs(raw_outputs)
    Note right of RTMOEstimator: NMS 적용<br/>키포인트 좌표 변환<br/>신뢰도 필터링
    RTMOEstimator->>RTMOEstimator: create_person_poses(processed_outputs)
    RTMOEstimator-->>DualPipeline: FramePoses

    Note over DualPipeline: 행동 분류 (ST-GCN++ ONNX)
    DualPipeline->>FightClassifier: classify_window(window_data)
    FightClassifier->>FightClassifier: extract_keypoint_features(window)
    Note right of FightClassifier: 윈도우 데이터에서<br/>키포인트 시퀀스 추출<br/>[M, T, V, C] 형태 생성
    FightClassifier->>FightClassifier: normalize_features(features)
    Note right of FightClassifier: 키포인트 정규화<br/>시간축 패딩/자르기

    FightClassifier->>ONNXSession: run(stgcn_input, output_names)
    Note right of ONNXSession: ST-GCN++ Fight 모델 실행<br/>시공간 그래프 컨볼루션
    ONNXSession-->>FightClassifier: classification_logits

    FightClassifier->>FightClassifier: apply_softmax(logits)
    FightClassifier->>FightClassifier: create_classification_result()
    FightClassifier-->>DualPipeline: ClassificationResult(Fight)

    DualPipeline->>FalldownClassifier: classify_window(window_data)
    FalldownClassifier->>FalldownClassifier: extract_keypoint_features(window)
    FalldownClassifier->>FalldownClassifier: normalize_features(features)

    FalldownClassifier->>ONNXSession: run(stgcn_input, output_names)
    Note right of ONNXSession: ST-GCN++ Falldown 모델 실행
    ONNXSession-->>FalldownClassifier: classification_logits

    FalldownClassifier->>FalldownClassifier: apply_softmax(logits)
    FalldownClassifier->>FalldownClassifier: create_classification_result()
    FalldownClassifier-->>DualPipeline: ClassificationResult(Falldown)
```

## PyTorch 적용 시 시퀀스 다이어그램

### 1. 시스템 초기화 및 파이프라인 설정 (PyTorch)

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Main as main.py
    participant ModeManager as ModeManager
    participant RealtimeMode as RealtimeMode
    participant DualPipeline as DualServicePipeline
    participant Factory as ModuleFactory
    participant RTMOEstimator as RTMOPoseEstimator
    participant MMPose as MMPose Framework
    participant ByteTracker as ByteTracker
    participant WindowProcessor as SlidingWindowProcessor
    participant FightClassifier as STGCNClassifier(Fight)
    participant FalldownClassifier as STGCNClassifier(Falldown)
    participant PyTorch as PyTorch Framework

    User->>Main: python main.py --mode inference.realtime
    Main->>ModeManager: execute("inference.realtime")
    ModeManager->>RealtimeMode: execute()

    Note over RealtimeMode: Config 검증 및 설정 로드
    RealtimeMode->>RealtimeMode: _validate_config(['input'])
    RealtimeMode->>RealtimeMode: dual_config = config.get('dual_service')

    RealtimeMode->>DualPipeline: create_dual_service_pipeline(config)
    DualPipeline->>DualPipeline: __init__(config)
    DualPipeline->>DualPipeline: initialize_pipeline()

    Note over DualPipeline: 공통 모듈 초기화 (PyTorch)
    DualPipeline->>DualPipeline: _initialize_common_modules()
    DualPipeline->>Factory: create_pose_estimator('rtmo', config)
    Factory->>RTMOEstimator: RTMOPoseEstimator(config)
    RTMOEstimator->>RTMOEstimator: initialize_model()
    RTMOEstimator->>MMPose: init_model(config_file, checkpoint_file)
    MMPose->>PyTorch: torch.load(checkpoint_file)
    PyTorch-->>MMPose: model weights
    MMPose->>PyTorch: model.to(device)
    PyTorch-->>MMPose: GPU 모델
    MMPose-->>RTMOEstimator: initialized model
    RTMOEstimator-->>Factory: estimator 객체
    Factory-->>DualPipeline: pose_estimator

    DualPipeline->>Factory: create_tracker('bytetrack', config)
    Factory->>ByteTracker: ByteTracker(config)
    ByteTracker->>ByteTracker: initialize_tracker()
    ByteTracker-->>Factory: tracker 객체
    Factory-->>DualPipeline: tracker

    DualPipeline->>Factory: create_window_processor('sliding_window', config)
    Factory->>WindowProcessor: SlidingWindowProcessor(config)
    WindowProcessor-->>Factory: processor 객체
    Factory-->>DualPipeline: window_processor

    Note over DualPipeline: 서비스별 모듈 초기화 (PyTorch)
    DualPipeline->>DualPipeline: _initialize_service_modules()

    loop services in ['fight', 'falldown']
        DualPipeline->>Factory: create_scorer(service, config)
        Factory-->>DualPipeline: scorer[service]

        DualPipeline->>Factory: create_classifier('stgcn', config)
        Factory->>FightClassifier: STGCNClassifier(fight_config) [PyTorch]
        FightClassifier->>FightClassifier: initialize_model()
        FightClassifier->>PyTorch: torch.load(model_path)
        PyTorch-->>FightClassifier: model weights
        FightClassifier->>PyTorch: model.to(device)
        FightClassifier->>PyTorch: model.eval()
        PyTorch-->>FightClassifier: initialized model
        FightClassifier-->>Factory: fight_classifier
        Factory-->>DualPipeline: classifiers['fight']

        DualPipeline->>Factory: create_classifier('stgcn', config)
        Factory->>FalldownClassifier: STGCNClassifier(falldown_config) [PyTorch]
        FalldownClassifier->>FalldownClassifier: initialize_model()
        FalldownClassifier->>PyTorch: torch.load(model_path)
        PyTorch-->>FalldownClassifier: model weights
        FalldownClassifier->>PyTorch: model.to(device)
        FalldownClassifier->>PyTorch: model.eval()
        PyTorch-->>FalldownClassifier: initialized model
        FalldownClassifier-->>Factory: falldown_classifier
        Factory-->>DualPipeline: classifiers['falldown']
    end

    DualPipeline-->>RealtimeMode: pipeline 객체
```

### 2. 실시간 처리 루프 (PyTorch 기반)

```mermaid
sequenceDiagram
    participant RealtimeMode as RealtimeMode
    participant DualPipeline as DualServicePipeline
    participant InputManager as RealtimeInputManager
    participant RTMOEstimator as RTMOPoseEstimator
    participant MMPose as MMPose Framework
    participant PyTorch as PyTorch Framework
    participant ByteTracker as ByteTracker
    participant WindowProcessor as SlidingWindowProcessor
    participant FightClassifier as STGCNClassifier(Fight)
    participant FalldownClassifier as STGCNClassifier(Falldown)
    participant Visualizer as InferenceVisualizer
    participant CV2 as OpenCV Display

    RealtimeMode->>DualPipeline: start_realtime_display()
    DualPipeline->>InputManager: RealtimeInputManager(input_source)
    DualPipeline->>InputManager: start()
    InputManager-->>DualPipeline: capture 시작 완료

    DualPipeline->>Visualizer: InferenceVisualizer()
    DualPipeline->>CV2: cv2.namedWindow()

    loop 실시간 프레임 처리
        DualPipeline->>InputManager: get_latest_frame()
        InputManager-->>DualPipeline: frame, frame_idx, timestamp

        Note over DualPipeline: 프레임 처리 시작
        DualPipeline->>DualPipeline: frame_idx += 1

        Note over DualPipeline,PyTorch: POSE ESTIMATION (PyTorch)
        DualPipeline->>RTMOEstimator: process_frame(frame, frame_idx)
        RTMOEstimator->>RTMOEstimator: validate_frame(frame)
        RTMOEstimator->>MMPose: inference_bottomup(model, frame)
        MMPose->>MMPose: preprocess_image(frame)
        Note right of MMPose: 이미지 전처리<br/>정규화 및 크기 조정
        MMPose->>PyTorch: model.forward(preprocessed_frame)
        Note right of PyTorch: GPU에서 forward pass<br/>CUDA 연산 수행
        PyTorch-->>MMPose: pose_predictions
        MMPose->>MMPose: postprocess_results(predictions)
        Note right of MMPose: NMS 적용<br/>키포인트 좌표 변환<br/>신뢰도 필터링
        MMPose-->>RTMOEstimator: pose_results
        RTMOEstimator->>RTMOEstimator: convert_to_frame_poses(pose_results)
        RTMOEstimator-->>DualPipeline: FramePoses 객체

        Note over DualPipeline: TRACKING
        DualPipeline->>ByteTracker: track_frame_poses(frame_poses)
        ByteTracker->>ByteTracker: update_tracks(detections)
        ByteTracker->>ByteTracker: assign_track_ids()
        ByteTracker-->>DualPipeline: tracked_frame_poses

        Note over DualPipeline: WINDOW PROCESSING
        DualPipeline->>WindowProcessor: add_frame(tracked_frame_poses)
        WindowProcessor->>WindowProcessor: check_window_ready()

        alt 윈도우 준비 완료
            WindowProcessor->>WindowProcessor: create_window_annotation()
            WindowProcessor-->>DualPipeline: ready_windows[]

            loop window in ready_windows
                Note over DualPipeline: DUAL SERVICE CLASSIFICATION
                DualPipeline->>DualPipeline: process_window_dual_service(window)

                Note over DualPipeline: Fight Service Processing (PyTorch)
                DualPipeline->>DualPipeline: _extract_tracked_poses_from_window()
                DualPipeline->>DualPipeline: calculate_scores(tracked_poses)
                DualPipeline->>DualPipeline: _apply_scores_to_window()

                DualPipeline->>FightClassifier: classify_window(scored_window)
                FightClassifier->>FightClassifier: preprocess_window_data()
                FightClassifier->>PyTorch: torch.tensor(window_data)
                PyTorch-->>FightClassifier: input_tensor
                FightClassifier->>PyTorch: input_tensor.to(device)
                FightClassifier->>PyTorch: model(input_tensor) [Fight PyTorch]
                Note right of PyTorch: GPU에서 ST-GCN++ 실행<br/>그래디언트 계산 비활성화
                PyTorch-->>FightClassifier: output_tensor
                FightClassifier->>PyTorch: torch.softmax(output_tensor)
                PyTorch-->>FightClassifier: probabilities
                FightClassifier->>FightClassifier: create_classification_result()
                FightClassifier-->>DualPipeline: fight_classification_result

                Note over DualPipeline: Falldown Service Processing (PyTorch)
                DualPipeline->>DualPipeline: _extract_tracked_poses_from_window()
                DualPipeline->>DualPipeline: calculate_scores(tracked_poses)
                DualPipeline->>DualPipeline: _apply_scores_to_window()

                DualPipeline->>FalldownClassifier: classify_window(scored_window)
                FalldownClassifier->>FalldownClassifier: preprocess_window_data()
                FalldownClassifier->>PyTorch: torch.tensor(window_data)
                PyTorch-->>FalldownClassifier: input_tensor
                FalldownClassifier->>PyTorch: input_tensor.to(device)
                FalldownClassifier->>PyTorch: model(input_tensor) [Falldown PyTorch]
                Note right of PyTorch: GPU에서 ST-GCN++ 실행
                PyTorch-->>FalldownClassifier: output_tensor
                FalldownClassifier->>PyTorch: torch.softmax(output_tensor)
                PyTorch-->>FalldownClassifier: probabilities
                FalldownClassifier->>FalldownClassifier: create_classification_result()
                FalldownClassifier-->>DualPipeline: falldown_classification_result

                DualPipeline->>DualPipeline: create_dual_classification_result()
                DualPipeline->>DualPipeline: latest_classification_result = result
            end
        else 윈도우 미준비
            WindowProcessor-->>DualPipeline: []
        end

        Note over DualPipeline: VISUALIZATION
        DualPipeline->>Visualizer: visualize_frame(frame, poses, classification)
        Visualizer->>Visualizer: draw_keypoints()
        Visualizer->>Visualizer: draw_bboxes()
        Visualizer->>Visualizer: draw_tracking_ids()
        Visualizer->>Visualizer: draw_classification_overlay()
        Visualizer-->>DualPipeline: display_frame

        DualPipeline->>CV2: cv2.imshow(window_name, display_frame)
        DualPipeline->>CV2: cv2.waitKey(1)

        alt 비디오 저장 모드
            DualPipeline->>CV2: video_writer.write(display_frame)
        end

        alt ESC 키 입력 또는 영상 종료
            break 루프 종료
        end
    end

    DualPipeline->>InputManager: stop()
    DualPipeline->>CV2: cv2.destroyAllWindows()
    DualPipeline-->>RealtimeMode: 처리 완료
```

### 3. PyTorch 모델 추론 상세 과정

```mermaid
sequenceDiagram
    participant DualPipeline as DualServicePipeline
    participant RTMOEstimator as RTMOPoseEstimator
    participant MMPose as MMPose Framework
    participant PyTorch as PyTorch Framework
    participant CUDA as CUDA Runtime
    participant FightClassifier as STGCNClassifier(Fight)
    participant FalldownClassifier as STGCNClassifier(Falldown)

    Note over DualPipeline: 포즈 추정 (RTMO PyTorch)
    DualPipeline->>RTMOEstimator: process_frame(frame, frame_idx)
    RTMOEstimator->>RTMOEstimator: validate_frame(frame)
    RTMOEstimator->>MMPose: inference_bottomup(model, frame)

    MMPose->>MMPose: preprocess_data(frame)
    Note right of MMPose: 이미지 리사이징<br/>정규화 적용<br/>데이터 증강 변환
    MMPose->>PyTorch: torch.from_numpy(preprocessed)
    MMPose->>PyTorch: data.to(device)

    MMPose->>PyTorch: with torch.no_grad():
    Note right of PyTorch: 그래디언트 계산 비활성화<br/>추론 모드 설정
    MMPose->>PyTorch: model.forward(input_data)
    PyTorch->>CUDA: 커널 실행 (GPU)
    Note right of CUDA: GPU 메모리 할당<br/>CUDA 커널 실행<br/>병렬 행렬 연산
    CUDA-->>PyTorch: feature_maps
    PyTorch-->>MMPose: model_outputs

    MMPose->>MMPose: postprocess_result(model_outputs)
    Note right of MMPose: 히트맵 디코딩<br/>키포인트 좌표 추출<br/>신뢰도 계산<br/>NMS 적용
    MMPose->>MMPose: format_pose_results(processed)
    MMPose-->>RTMOEstimator: pose_results

    RTMOEstimator->>RTMOEstimator: convert_mmpose_to_frame_poses()
    RTMOEstimator-->>DualPipeline: FramePoses

    Note over DualPipeline: 행동 분류 (ST-GCN++ PyTorch)
    DualPipeline->>FightClassifier: classify_window(window_data)
    FightClassifier->>FightClassifier: extract_keypoint_sequence(window)
    Note right of FightClassifier: 윈도우에서 키포인트 시퀀스 추출<br/>[M, T, V, C] 형태 생성<br/>M: 사람 수, T: 프레임 수<br/>V: 관절 수, C: 좌표+신뢰도

    FightClassifier->>FightClassifier: normalize_keypoints(sequence)
    FightClassifier->>PyTorch: torch.from_numpy(normalized_seq)
    FightClassifier->>PyTorch: input_tensor.to(device)
    PyTorch->>CUDA: 텐서 GPU 메모리 복사

    FightClassifier->>PyTorch: with torch.no_grad():
    FightClassifier->>PyTorch: model.forward(input_tensor)
    Note right of PyTorch: ST-GCN++ 모델 실행
    PyTorch->>CUDA: 그래프 컨볼루션 연산
    Note right of CUDA: 시공간 그래프 처리<br/>다중 스케일 특징 추출<br/>어텐션 메커니즘 적용
    CUDA-->>PyTorch: classification_logits
    PyTorch-->>FightClassifier: output_tensor

    FightClassifier->>PyTorch: torch.softmax(output_tensor, dim=1)
    PyTorch->>CUDA: 소프트맥스 계산
    CUDA-->>PyTorch: probabilities
    PyTorch-->>FightClassifier: softmax_probs

    FightClassifier->>PyTorch: torch.argmax(softmax_probs)
    PyTorch-->>FightClassifier: predicted_class
    FightClassifier->>FightClassifier: create_classification_result()
    FightClassifier-->>DualPipeline: ClassificationResult(Fight)

    DualPipeline->>FalldownClassifier: classify_window(window_data)
    FalldownClassifier->>FalldownClassifier: extract_keypoint_sequence(window)
    FalldownClassifier->>FalldownClassifier: normalize_keypoints(sequence)
    FalldownClassifier->>PyTorch: torch.from_numpy(normalized_seq)
    FalldownClassifier->>PyTorch: input_tensor.to(device)

    FalldownClassifier->>PyTorch: with torch.no_grad():
    FalldownClassifier->>PyTorch: model.forward(input_tensor)
    PyTorch->>CUDA: 그래프 컨볼루션 연산 (Falldown)
    CUDA-->>PyTorch: classification_logits
    PyTorch-->>FalldownClassifier: output_tensor

    FalldownClassifier->>PyTorch: torch.softmax(output_tensor, dim=1)
    PyTorch-->>FalldownClassifier: softmax_probs
    FalldownClassifier->>PyTorch: torch.argmax(softmax_probs)
    PyTorch-->>FalldownClassifier: predicted_class
    FalldownClassifier->>FalldownClassifier: create_classification_result()
    FalldownClassifier-->>DualPipeline: ClassificationResult(Falldown)
```

## ONNX vs PyTorch 비교 분석

### 성능 특성 비교

| 구분 | ONNX | PyTorch |
|------|------|---------|
| **초기화 시간** | 빠름 (모델 로딩만) | 느림 (프레임워크 초기화 + 모델 로딩) |
| **메모리 사용량** | 낮음 (최적화된 런타임) | 높음 (프레임워크 오버헤드) |
| **추론 속도** | 빠름 (최적화된 연산) | 중간 (동적 그래프 오버헤드) |
| **GPU 활용** | 최적화됨 | 우수 (네이티브 CUDA 지원) |
| **배포 용이성** | 매우 좋음 | 보통 (의존성 많음) |

### 기술적 차이점

#### ONNX 기반 처리
- **정적 그래프**: 컴파일 타임에 그래프 최적화
- **전용 런타임**: ONNX Runtime을 통한 최적화된 실행
- **플랫폼 독립성**: 다양한 하드웨어에서 일관된 성능
- **메모리 효율성**: 최소한의 메모리 오버헤드

#### PyTorch 기반 처리
- **동적 그래프**: 런타임에 그래프 구성 및 실행
- **프레임워크 통합**: MMPose와의 네이티브 통합
- **디버깅 용이성**: 풍부한 디버깅 도구와 프로파일링
- **확장성**: 새로운 모델과 기능의 쉬운 통합

### 사용 시나리오 권장사항

#### ONNX 권장 상황
- 프로덕션 환경 배포
- 제한된 하드웨어 리소스
- 높은 처리량이 필요한 경우
- 크로스 플랫폼 호환성이 중요한 경우

#### PyTorch 권장 상황
- 개발 및 실험 단계
- 모델 커스터마이징이 빈번한 경우
- 풍부한 디버깅 정보가 필요한 경우
- 새로운 기능 개발 및 테스트

## 결론

inference.realtime 모드는 ONNX와 PyTorch 두 가지 백엔드를 지원하여 다양한 배포 요구사항을 충족할 수 있습니다. ONNX는 프로덕션 환경에서의 성능과 효율성에 중점을 두고, PyTorch는 개발 단계에서의 유연성과 확장성을 제공합니다. 시스템의 전체적인 아키텍처는 동일하지만, 각 백엔드의 특성에 따라 모델 로딩, 추론 과정, 메모리 관리 방식에서 차이를 보입니다.