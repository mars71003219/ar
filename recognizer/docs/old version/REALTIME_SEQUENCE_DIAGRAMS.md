# 실시간 추론 시스템 시퀀스 다이어그램

## 개요

본 문서는 Violence Detection 실시간 추론 시스템의 상세한 시퀀스 다이어그램과 상호작용 흐름을 제공한다.

---

## 1. 전체 시스템 초기화 시퀀스

```mermaid
sequenceDiagram
    participant Main as main.py
    participant MM as ModeManager
    participant IP as InferencePipeline
    participant MF as ModuleFactory
    participant PE as RTMO PoseEstimator
    participant BT as ByteTracker
    participant SC as RegionBasedScorer
    participant AC as STGCN Classifier
    participant EM as EventManager
    participant RV as RealtimeVisualizer

    Main->>MM: ModeManager(config)
    MM->>IP: InferencePipeline(config)
    
    IP->>IP: _initialize_event_manager(config)
    IP->>EM: EventManager(event_config)
    EM-->>IP: ✓ initialized
    
    IP->>IP: initialize_pipeline()
    
    IP->>MF: create_pose_estimator('rtmo_onnx', config)
    MF->>PE: RTMOONNXEstimator(config)
    PE-->>IP: ✓ pose_estimator
    
    IP->>MF: create_tracker('bytetrack', config)
    MF->>BT: ByteTrackerWrapper(config)
    BT-->>IP: ✓ tracker
    
    IP->>MF: create_scorer('region_based', config)
    MF->>SC: RegionBasedScorer(config)
    SC-->>IP: ✓ scorer
    
    IP->>MF: create_classifier('stgcn', config)
    MF->>AC: STGCNActionClassifier(config)
    AC->>AC: warmup()
    AC-->>IP: ✓ classifier
    
    IP->>IP: _start_classification_thread()
    IP-->>MM: ✓ pipeline_ready
    MM-->>Main: ✓ ready
```

---

## 2. 실시간 처리 메인 루프

```mermaid
sequenceDiagram
    participant IP as InferencePipeline
    participant RIM as RealtimeInputManager
    participant PE as PoseEstimator
    participant BT as ByteTracker
    participant SC as RegionScorer
    participant WP as WindowProcessor
    participant CQ as ClassificationQueue
    participant CW as ClassificationWorker
    participant AC as ActionClassifier
    participant EM as EventManager
    participant RV as RealtimeVisualizer

    loop 실시간 처리 루프
        IP->>RIM: get_frame()
        RIM-->>IP: frame, frame_idx
        
        IP->>IP: process_frame(frame, frame_idx)
        
        Note over IP,PE: 1. 포즈 추정
        IP->>PE: estimate_poses(frame)
        PE->>PE: ONNX inference
        PE-->>IP: raw_poses
        
        Note over IP,BT: 2. 다중 객체 추적
        IP->>BT: track(raw_poses)
        BT->>BT: kalman_filter + data_association
        BT-->>IP: tracked_poses
        
        Note over IP,SC: 3. 포즈 점수화 및 필터링
        IP->>SC: score_poses(tracked_poses)
        SC->>SC: quality_assessment + filtering
        SC-->>IP: scored_poses
        
        Note over IP,WP: 4. 윈도우 데이터 관리
        IP->>WP: add_frame_data(scored_poses)
        WP->>WP: sliding_window_update
        
        alt 윈도우가 준비됨
            WP-->>IP: window_ready=True
            IP->>WP: get_window_data()
            WP-->>IP: window_data
            IP->>CQ: put((window_data, window_id))
        end
        
        Note over IP,RV: 5. 시각화 업데이트
        IP->>RV: show_frame(frame, poses, overlay_data)
        RV->>RV: draw_poses + draw_classification + draw_events
        RV-->>IP: continue=True/False
    end
```

---

## 3. 비동기 분류 처리 시퀀스

```mermaid
sequenceDiagram
    participant CW as ClassificationWorker
    participant CQ as ClassificationQueue
    participant AC as ActionClassifier
    participant EM as EventManager
    participant EL as EventLogger
    participant RV as RealtimeVisualizer

    loop 분류 워커 루프
        CW->>CQ: get(timeout=0.5)
        
        alt 작업 있음
            CQ-->>CW: (window_data, window_id)
            
            Note over CW,AC: ST-GCN++ 추론
            CW->>AC: classify_window(window_data)
            AC->>AC: preprocess + forward + postprocess
            AC-->>CW: ClassificationResult
            
            Note over CW,EM: 이벤트 처리
            CW->>CW: convert_to_event_format(result)
            CW->>EM: process_classification_result(event_data)
            
            alt 이벤트 발생
                EM->>EM: check_thresholds + update_counters
                EM->>EM: create_event_data()
                EM->>EL: log_event(event_data)
                EL->>EL: write_to_file(json/csv)
                EM-->>CW: event_data
                
                CW->>RV: update_event_history(event_data)
            else 이벤트 없음
                EM-->>CW: None
            end
            
            CW->>CQ: task_done()
            
        else 타임아웃
            CW->>CW: continue
        end
    end
```

---

## 4. 이벤트 생명주기 상세 시퀀스

```mermaid
sequenceDiagram
    participant EM as EventManager
    participant EC as EventConfig
    participant EL as EventLogger
    participant CB as EventCallbacks

    Note over EM: 분류 결과 수신
    EM->>EM: process_classification_result(result)
    
    alt Violence 예측 (confidence ≥ 0.7)
        EM->>EM: consecutive_violence++
        EM->>EM: consecutive_normal = 0
        
        alt 연속 탐지 조건 만족 (≥ 3회)
            EM->>EM: current_event_active = True
            EM->>EM: create_event_data(VIOLENCE_START)
            EM->>EL: log_event(event_data)
            EM->>CB: call_callbacks(VIOLENCE_START)
            
        else 조건 미만족
            EM->>EM: 카운터만 증가
        end
        
    else Normal 예측 (confidence ≥ 0.5)
        EM->>EM: consecutive_normal++
        EM->>EM: consecutive_violence = 0
        
        alt 이벤트 활성 상태 + 연속 정상 조건 (≥ 5회)
            EM->>EM: current_event_active = False
            EM->>EM: calculate_duration()
            EM->>EM: create_event_data(VIOLENCE_END)
            EM->>EL: log_event(event_data)
            EM->>CB: call_callbacks(VIOLENCE_END)
            
        else 조건 미만족
            alt 이벤트 활성 상태
                EM->>EM: check_ongoing_alert_interval()
                alt 30초 경과
                    EM->>EM: create_event_data(VIOLENCE_ONGOING)
                    EM->>EL: log_event(event_data)
                    EM->>CB: call_callbacks(VIOLENCE_ONGOING)
                end
            end
        end
        
    else 낮은 신뢰도
        EM->>EM: 카운터 유지 (변화 없음)
    end
```

---

## 5. 시각화 업데이트 시퀀스

```mermaid
sequenceDiagram
    participant IP as InferencePipeline
    participant RV as RealtimeVisualizer
    participant PV as PoseVisualizer
    participant CV as OpenCV

    IP->>RV: show_frame(frame, poses, additional_info, overlay_data)
    
    RV->>RV: resize_frame(frame)
    RV->>RV: scale_poses_to_display(poses)
    
    Note over RV,PV: 포즈 시각화
    RV->>PV: visualize_frame(frame, scaled_poses)
    PV->>PV: draw_keypoints + draw_limbs + draw_tracking_ids
    PV-->>RV: frame_with_poses
    
    Note over RV: 분류 결과 표시
    RV->>RV: draw_classification_results(frame)
    loop 윈도우 히스토리
        RV->>RV: draw_window_info(window_data)
        RV->>RV: apply_confidence_threshold_coloring()
    end
    
    Note over RV: 추가 정보 오버레이
    RV->>RV: add_overlay_info(frame, additional_info)
    RV->>RV: draw_fps_info()
    RV->>RV: draw_performance_stats()
    
    Note over RV: 이벤트 상태 표시
    RV->>RV: _draw_event_history(frame)
    RV->>RV: determine_current_event_status()
    RV->>RV: draw_event_box(status, color, details)
    
    Note over RV,CV: 화면 출력
    RV->>CV: imshow(window_name, final_frame)
    RV->>CV: waitKey(1)
    CV-->>RV: key_pressed
    
    alt 'q' 또는 ESC
        RV-->>IP: False (종료)
    else 기타 키
        RV-->>IP: True (계속)
    end
```

---

## 6. 오류 처리 및 복구 시퀀스

```mermaid
sequenceDiagram
    participant IP as InferencePipeline
    participant PE as PoseEstimator
    participant AC as ActionClassifier
    participant EM as EventManager
    participant LOG as Logger

    Note over IP: 오류 상황 발생
    
    alt 포즈 추정 실패
        IP->>PE: estimate_poses(frame)
        PE-->>IP: Exception/None
        IP->>LOG: log_error("Pose estimation failed")
        IP->>IP: use_empty_poses()
        IP->>IP: continue_processing()
        
    else 분류 실패
        IP->>AC: classify_window(window_data)
        AC-->>IP: Exception/None
        IP->>LOG: log_error("Classification failed")
        IP->>IP: skip_classification_result()
        
    else 이벤트 처리 실패
        IP->>EM: process_classification_result(result)
        EM-->>IP: Exception
        IP->>LOG: log_error("Event processing failed")
        IP->>IP: continue_without_event_update()
        
    else 메모리 부족
        IP->>IP: detect_memory_pressure()
        IP->>IP: clear_old_buffers()
        IP->>IP: reduce_queue_sizes()
        IP->>LOG: log_warning("Memory optimization applied")
        
    else GPU 오류
        IP->>PE: estimate_poses(frame)
        PE-->>IP: CUDA_ERROR
        IP->>PE: fallback_to_cpu()
        IP->>LOG: log_warning("Fallback to CPU processing")
        IP->>IP: continue_with_cpu()
    end
    
    Note over IP: 시스템 복구 완료
    IP->>IP: resume_normal_operation()
```

---

## 7. 성능 모니터링 시퀀스

```mermaid
sequenceDiagram
    participant IP as InferencePipeline
    participant PT as PerformanceTracker
    participant STATS as Statistics
    participant LOG as Logger

    loop 성능 측정 주기 (매 프레임)
        Note over IP: 처리 시간 측정
        IP->>PT: start_timing("pose_estimation")
        IP->>IP: pose_estimation_process()
        IP->>PT: end_timing("pose_estimation")
        
        IP->>PT: start_timing("tracking")
        IP->>IP: tracking_process()
        IP->>PT: end_timing("tracking")
        
        IP->>PT: start_timing("classification")
        IP->>IP: classification_process()
        IP->>PT: end_timing("classification")
        
        IP->>PT: update_frame_count()
    end
    
    loop 통계 업데이트 주기 (매 10초)
        IP->>PT: calculate_stage_fps()
        PT-->>IP: fps_stats
        
        IP->>PT: get_processing_time_stats()
        PT-->>IP: timing_stats
        
        IP->>STATS: update_performance_stats(fps_stats, timing_stats)
        
        alt 성능 저하 감지
            STATS->>STATS: detect_performance_degradation()
            STATS->>LOG: log_warning("Performance degradation detected")
            STATS->>IP: suggest_optimization()
        end
        
        IP->>LOG: log_info(f"Performance: {fps_stats}")
    end
```

---

## 8. 설정 로딩 및 모듈 팩토리 시퀀스

```mermaid
sequenceDiagram
    participant Main as main.py
    participant CL as ConfigLoader
    participant MF as ModuleFactory
    participant REG as ModuleRegistry

    Main->>Main: register_modules()
    
    Note over Main,REG: 모듈 등록
    Main->>MF: register_pose_estimator('rtmo', RTMOPoseEstimator)
    MF->>REG: add_pose_estimator('rtmo', class, config)
    
    Main->>MF: register_pose_estimator('rtmo_onnx', RTMOONNXEstimator)
    MF->>REG: add_pose_estimator('rtmo_onnx', class, config)
    
    Main->>MF: register_tracker('bytetrack', ByteTrackerWrapper)
    MF->>REG: add_tracker('bytetrack', class, config)
    
    Main->>MF: register_classifier('stgcn', STGCNActionClassifier)
    MF->>REG: add_classifier('stgcn', class, config)
    
    Note over Main,CL: 설정 로딩
    Main->>CL: load_config('config.yaml')
    CL->>CL: parse_yaml()
    CL->>CL: validate_config()
    CL->>CL: create_unified_config()
    CL-->>Main: config_object
    
    Note over Main,MF: 모듈 생성
    Main->>MF: create_pose_estimator('rtmo_onnx', pose_config)
    MF->>REG: get_registered_class('rtmo_onnx')
    REG-->>MF: RTMOONNXEstimator
    MF->>MF: instantiate(RTMOONNXEstimator, pose_config)
    MF-->>Main: pose_estimator_instance
```

---

## 9. 메모리 관리 및 리소스 정리 시퀀스

```mermaid
sequenceDiagram
    participant IP as InferencePipeline
    participant MM as MemoryManager
    participant GC as GarbageCollector
    participant GPU as GPUMemory

    loop 메모리 관리 주기
        IP->>MM: check_memory_usage()
        MM->>MM: get_system_memory_info()
        MM->>GPU: get_gpu_memory_info()
        
        alt 메모리 사용률 > 80%
            MM->>IP: trigger_memory_cleanup()
            
            IP->>IP: clear_old_frame_buffers()
            IP->>IP: clear_classification_results_cache()
            IP->>IP: clear_event_history_old_items()
            
            IP->>GC: collect()
            GC->>GC: python_garbage_collection()
            
            IP->>GPU: empty_cache()
            GPU->>GPU: cuda_empty_cache()
            
            MM->>MM: verify_memory_freed()
            MM-->>IP: memory_optimization_complete
            
        else 메모리 정상
            MM-->>IP: memory_status_ok
        end
    end
    
    Note over IP: 시스템 종료 시
    IP->>IP: cleanup_resources()
    IP->>IP: stop_classification_thread()
    IP->>IP: release_input_manager()
    IP->>IP: cleanup_visualizer()
    IP->>GPU: release_gpu_memory()
    IP->>MM: final_cleanup()
```

---

## 시퀀스 다이어그램 요약

### 주요 상호작용 패턴

1. **초기화 단계**: 순차적 모듈 로딩 및 설정
2. **실시간 처리**: 파이프라인 병렬 처리
3. **비동기 분류**: 큐 기반 백그라운드 처리
4. **이벤트 관리**: 상태 기반 이벤트 생명주기
5. **오류 처리**: 단계별 fallback 및 복구
6. **성능 모니터링**: 주기적 통계 수집 및 최적화

### 핵심 설계 원칙

- **비동기 처리**: 분류 작업의 독립적 실행
- **모듈 분리**: 각 컴포넌트의 독립성 보장
- **상태 관리**: 이벤트 및 성능 상태의 체계적 관리
- **오류 복구**: 단계별 fallback 메커니즘
- **리소스 최적화**: 동적 메모리 관리 및 정리