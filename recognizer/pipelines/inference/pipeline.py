"""
실시간 추론 파이프라인 메인 클래스
"""

import time
import cv2
import numpy as np
import logging
import threading
from queue import Queue, Empty
from typing import Dict, Any, List, Optional, Callable, Union

import sys
from pathlib import Path

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

# from .config import RealtimeConfig, Dict[str, Any]
from ..base import BasePipeline, ModuleInitializer, PerformanceTracker
from utils.data_structure import PersonPose, FramePoses, WindowAnnotation, ClassificationResult
from utils.realtime_input import RealtimeInputManager
from visualization.realtime_visualizer import RealtimeVisualizer
from events.event_manager import EventManager, EventConfig


class InferencePipeline(BasePipeline):
    """실시간 추론 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 실행 모드 설정 - config에서 mode 추출
        full_mode = config.get('mode', 'inference.analysis')
        if '.' in full_mode:
            _, self.mode = full_mode.split('.', 1)
        else:
            self.mode = full_mode
        
        logging.info(f"InferencePipeline initialized with mode: {self.mode}")
        
        # 모듈 인스턴스들
        self.pose_estimator = None
        self.tracker = None
        self.scorer = None
        self.classifier = None
        self.window_processor = None
        
        # 설정에서 큐 크기 가져오기
        max_queue_size = config.get('max_queue_size', 200) if isinstance(config, dict) else 200
        
        # 실시간 처리용 결과 큐
        self.result_queue = Queue(maxsize=100)
        
        # 프레임 버퍼
        self.frame_buffer: List[FramePoses] = []
        
        # 실시간 디스플레이용 최근 분류 결과 유지
        self.latest_classification = None
        
        # 성능 추적
        self.performance_tracker = PerformanceTracker()
        self.performance_stats = {
            'windows_classified': 0,
            'total_alerts': 0,
            'frames_skipped': 0
        }
        
        # 구간별 성능 측정
        self.stage_timings = {
            'pose_estimation': [],
            'tracking': [],
            'scoring': [],
            'classification': []
        }
        
        # FPS 계산을 위한 윈도우 크기
        self.fps_window_size = 30
        
        # 분류 결과 저장
        self.classification_results = []
        
        # 포즈 결과 저장 (시각화용)
        self.frame_poses_results = []  # 트래킹 및 스코어링 완료된 데이터
        self.rtmo_poses_results = []   # RTMO 원본 포즈 데이터
        
        # 비동기 분류 처리
        self.classification_queue = Queue(maxsize=10)  # 분류 대기 큐
        self.classification_thread = None
        self.classification_running = False
        
        # 이벤트 관리자 초기화
        self.event_manager = None
        self._initialize_event_manager(config)
        
        # 처리 모드 설정
        if hasattr(config, 'processing_mode'):
            self.processing_mode = config.processing_mode
        elif hasattr(config, 'get'):
            self.processing_mode = config.get('processing_mode', 'realtime')
        else:
            self.processing_mode = 'realtime'
    
    def _initialize_event_manager(self, config: Dict[str, Any]):
        """이벤트 관리자 초기화"""
        try:
            # 이벤트 설정 추출
            event_config_dict = config.get('events', {})
            
            event_config = EventConfig(
                alert_threshold=event_config_dict.get('alert_threshold', 0.7),
                min_consecutive_detections=event_config_dict.get('min_consecutive_detections', 3),
                normal_threshold=event_config_dict.get('normal_threshold', 0.5),
                min_consecutive_normal=event_config_dict.get('min_consecutive_normal', 5),
                min_event_duration=event_config_dict.get('min_event_duration', 2.0),
                max_event_duration=event_config_dict.get('max_event_duration', 300.0),
                cooldown_duration=event_config_dict.get('cooldown_duration', 10.0),
                enable_ongoing_alerts=event_config_dict.get('enable_ongoing_alerts', True),
                ongoing_alert_interval=event_config_dict.get('ongoing_alert_interval', 30.0),
                save_event_log=event_config_dict.get('save_event_log', True),
                event_log_format=event_config_dict.get('event_log_format', 'json'),
                event_log_path=event_config_dict.get('event_log_path', 'output/event_logs')
            )
            
            self.event_manager = EventManager(event_config)
            
            # 이벤트 콜백 등록
            from events.event_types import EventType
            self.event_manager.add_event_callback(EventType.VIOLENCE_START, self._on_violence_start)
            self.event_manager.add_event_callback(EventType.VIOLENCE_END, self._on_violence_end)
            self.event_manager.add_event_callback(EventType.VIOLENCE_ONGOING, self._on_violence_ongoing)
            
            logging.info(f"Event manager initialized with config: {event_config}")
            
        except Exception as e:
            logging.error(f"Failed to initialize event manager: {e}")
            self.event_manager = None
    
    def _on_violence_start(self, event_data):
        """폭력 시작 이벤트 콜백"""
        logging.warning(f"[ALERT] VIOLENCE DETECTED! Window: {event_data.window_id}, "
                       f"Confidence: {event_data.confidence:.3f}, "
                       f"Consecutive: {event_data.metadata.get('consecutive_detections', 0)}")
        
        # 성능 통계 업데이트
        self.performance_stats['total_alerts'] = self.performance_stats.get('total_alerts', 0) + 1
    
    def _on_violence_end(self, event_data):
        """폭력 종료 이벤트 콜백"""
        duration = event_data.duration or 0.0
        reason = event_data.metadata.get('reason', 'normal_detection')
        
        logging.info(f"[ALERT] Violence ended. Window: {event_data.window_id}, "
                    f"Duration: {duration:.1f}s, Reason: {reason}")
    
    def _on_violence_ongoing(self, event_data):
        """폭력 진행 중 이벤트 콜백"""
        duration = event_data.duration or 0.0
        logging.info(f"[ALERT] Violence ongoing. Window: {event_data.window_id}, "
                    f"Duration: {duration:.1f}s")
    
    def reset_pipeline_state(self):
        """파이프라인 상태 초기화 (폴더 처리 시 각 비디오마다 호출)"""
        logging.info("Resetting pipeline state for new video")
        
        # 프레임 버퍼 초기화
        self.frame_buffer = []
        
        # 윈도우 프로세서 초기화
        if hasattr(self, 'window_processor') and self.window_processor:
            self.window_processor.reset()
        
        # 분류 결과 초기화
        self.classification_results = []
        
        # 포즈 결과 초기화
        self.frame_poses_results = []
        self.rtmo_poses_results = []
        
        # 이벤트 관리자 리셋
        if self.event_manager:
            self.event_manager.reset()
        
        # 성능 통계 초기화
        self.performance_stats = {
            'total_frames': 0,
            'pose_estimation_time': 0.0,
            'tracking_time': 0.0,
            'scoring_time': 0.0,
            'classification_time': 0.0,
            'visualization_time': 0.0,
            'windows_classified': 0
        }
        
        # 시간 측정 초기화
        self.stage_timings = {
            'pose_estimation': [],
            'tracking': [],
            'scoring': [],
            'classification': [],
            'visualization': []
        }
    
    def initialize_pipeline(self) -> bool:
        """파이프라인 모듈 초기화"""
        try:
            logging.info("Initializing realtime inference pipeline")
            
            # UnifiedConfig 객체에서 각 모듈별 설정 구성
            if hasattr(self.config, 'pose_model'):
                # UnifiedConfig 객체인 경우
                pose_config = {
                    'model_name': self.config.pose_model,
                    'config_file': self.config.pose_config,
                    'checkpoint_path': self.config.pose_checkpoint,
                    'device': self.config.pose_device,
                    'score_threshold': self.config.pose_score_threshold,
                    'input_size': self.config.pose_input_size
                }
                tracking_config = {
                    'tracker_name': self.config.tracker_name,
                    'track_thresh': self.config.track_thresh,
                    'track_buffer': self.config.track_buffer,
                    'match_thresh': self.config.match_thresh
                }
                scoring_config = {
                    'scorer_name': self.config.scorer_name,
                    'quality_threshold': self.config.scoring_quality_threshold,
                    'min_track_length': self.config.min_track_length,
                    'img_width': 640,
                    'img_height': 640
                }
                classification_config = {
                    'model_name': self.config.classifier_name,
                    'config_file': self.config.classifier_config,
                    'checkpoint_path': self.config.classifier_checkpoint,
                    'device': self.config.classifier_device,
                    'window_size': self.config.classifier_window_size,
                    'confidence_threshold': self.config.classifier_confidence_threshold,
                    'class_names': self.config.class_names
                }
                window_size = self.config.window_size
                inference_stride = self.config.inference_stride
            else:
                # 딕셔너리인 경우 - 새로운 YAML 구조에서 설정 추출
                models = self.config.get('models', {})
                performance = self.config.get('performance', {})
                
                # 포즈 추정 설정 - inference_mode에 따라 적절한 설정 섹션 선택
                pose_estimation = models.get('pose_estimation', {})
                inference_mode = pose_estimation.get('inference_mode', 'pth')
                mode_config = pose_estimation.get(inference_mode, {})
                
                
                pose_config = {
                    'inference_mode': inference_mode,  # 추론 모드 정보 추가
                    'model_name': mode_config.get('model_name', 'rtmo'),
                    'config_file': mode_config.get('config_file', ''),
                    'checkpoint_path': mode_config.get('checkpoint_path', ''),
                    'model_path': mode_config.get('model_path', ''),  # ONNX/TensorRT용
                    'device': mode_config.get('device', 'cuda:0'),
                    'score_threshold': mode_config.get('score_threshold', 0.3),
                    'input_size': mode_config.get('input_size', [640, 640]),
                    # 각 inference_mode별 전체 설정도 포함
                    inference_mode: mode_config  # 'onnx': {...}, 'pth': {...}, 'tensorrt': {...}
                }
                
                # 다른 inference_mode 설정들도 포함 (factory.py에서 접근할 수 있도록)
                for mode_name in ['onnx', 'pth', 'tensorrt']:
                    if mode_name in pose_estimation:
                        pose_config[mode_name] = pose_estimation[mode_name]
                
                # 트래킹 설정
                tracking = models.get('tracking', {})
                tracking_config = {
                    'tracker_name': tracking.get('tracker_name', 'bytetrack'),
                    'track_thresh': tracking.get('track_thresh', 0.4),
                    'track_buffer': tracking.get('track_buffer', 50),
                    'match_thresh': tracking.get('match_thresh', 0.4)
                }
                
                # 스코어링 설정
                scoring = models.get('scoring', {})
                scoring_config = {
                    'scorer_name': scoring.get('scorer_name', 'region_based'),
                    'quality_threshold': scoring.get('quality_threshold', 0.3),
                    'min_track_length': scoring.get('min_track_length', 10),
                    'img_width': 640,
                    'img_height': 640
                }
                
                # 분류 설정
                classification = models.get('action_classification', {})
                classification_config = {
                    'model_name': classification.get('model_name', 'stgcn'),
                    'config_file': classification.get('config_file', ''),
                    'checkpoint_path': classification.get('checkpoint_path', ''),
                    'device': classification.get('device', 'cuda:0'),
                    'window_size': classification.get('window_size', 100),
                    'confidence_threshold': classification.get('confidence_threshold', 0.4),
                    'class_names': classification.get('class_names', ['NonFight', 'Fight'])
                }
                
                window_size = performance.get('window_size', 100)
                inference_stride = performance.get('window_stride', 50)
                
                # 추가 실시간 처리 설정
                self.config['skip_frames'] = 1  # 기본값 설정
                self.config['target_fps'] = 30.0
                self.config['max_queue_size'] = 200
            
            self.pose_estimator = ModuleInitializer.init_pose_estimator(
                self.factory, pose_config
            )
            self.tracker = ModuleInitializer.init_tracker(
                self.factory, tracking_config
            )
            self.scorer = ModuleInitializer.init_scorer(
                self.factory, scoring_config
            )
            self.classifier = ModuleInitializer.init_classifier(
                self.factory, classification_config
            )
            # 윈도우 프로세서 설정 구성 (모드별 설정 포함)
            def get_config_value(key, default):
                if hasattr(self.config, key):
                    return getattr(self.config, key)
                elif hasattr(self.config, 'get'):
                    return self.config.get(key, default)
                else:
                    return default
            
            # realtime_mode 설정을 window_size와 stride에 우선 적용
            realtime_window_size = get_config_value('window_size', window_size)
            realtime_stride = get_config_value('window_stride', inference_stride)
            
            window_processor_config = {
                'window_size': realtime_window_size,
                'window_stride': realtime_stride,
                'processing_mode': get_config_value('processing_mode', 'realtime'),
                # 자동 계산된 값들 사용
                'buffer_size': get_config_value('buffer_size', realtime_window_size + realtime_stride),
                'classification_delay': get_config_value('classification_delay', realtime_window_size),
                'show_keypoints': get_config_value('show_keypoints', True),
                'show_tracking_ids': get_config_value('show_tracking_ids', True),
                'show_composite_score': get_config_value('show_composite_score', False)
            }
            
            self.window_processor = ModuleInitializer.init_window_processor(
                self.factory, window_processor_config
            )
            
            # 모든 모델 초기화 검증
            logging.info("Verifying model initialization...")
            
            if not self.pose_estimator.ensure_initialized():
                logging.error("Failed to initialize pose estimator")
                return False
                
            if not self.classifier.ensure_initialized():
                logging.error("Failed to initialize action classifier")
                return False
            
            # 비동기 분류 스레드 시작
            self._start_classification_thread()
            
            self._initialized = True
            logging.info("Pipeline initialization completed")
            return True
            
        except Exception as e:
            logging.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _start_classification_thread(self):
        """비동기 분류 처리 스레드 시작"""
        if self.classification_thread is None or not self.classification_thread.is_alive():
            self.classification_running = True
            self.classification_thread = threading.Thread(
                target=self._classification_worker, 
                daemon=True,
                name="ClassificationWorker"
            )
            self.classification_thread.start()
            logging.info("Classification worker thread started")
    
    def _stop_classification_thread(self):
        """비동기 분류 처리 스레드 중지"""
        try:
            self.classification_running = False
            if self.classification_thread and self.classification_thread.is_alive():
                # 스레드 종료 대기를 위한 더미 작업 추가
                try:
                    self.classification_queue.put(None, timeout=0.1)
                except:
                    pass
                
                self.classification_thread.join(timeout=2.0)
                
                if self.classification_thread.is_alive():
                    logging.warning("Classification thread did not stop gracefully")
                else:
                    logging.info("Classification worker thread stopped")
            
        except Exception as e:
            logging.error(f"Error stopping classification thread: {e}")
    
    def _classification_worker(self):
        """비동기 분류 처리 워커"""
        logging.info("Classification worker started")
        
        while self.classification_running:
            try:
                # 분류 작업 대기 (타임아웃으로 주기적 체크)
                task = self.classification_queue.get(timeout=0.5)
                
                if task is None:  # 종료 신호
                    break
                
                window_data, window_id = task
                
                # 실제 분류 수행
                classification_start = time.time()
                result = self.classifier.classify_window(window_data)
                classification_time = time.time() - classification_start
                
                # 결과 검증
                if result and hasattr(result, 'prediction') and hasattr(result, 'confidence'):
                    # 타이밍 정보를 메인 스레드의 통계에 추가
                    self._add_stage_timing('classification', classification_time)
                    
                    # 결과를 메인 스레드로 전달
                    timestamp = time.time()
                    self.classification_results.append({
                        'window_id': window_id,
                        'result': result,
                        'processing_time': classification_time,
                        'timestamp': timestamp,
                        'processed': False  # 아직 처리되지 않음
                    })
                    
                    # 이벤트 처리
                    if self.event_manager:
                        # 이벤트 시스템용 데이터 변환
                        # result.prediction은 클래스 인덱스 (0: NonFight, 1: Fight)
                        prediction_name = 'violence' if result.prediction == 1 else 'normal'
                        
                        event_result_data = {
                            'window_id': window_id,
                            'prediction': prediction_name,  # 문자열로 변환
                            'confidence': result.confidence,
                            'timestamp': timestamp,
                            'class_index': result.prediction,  # 원본 인덱스도 보존
                            'probabilities': result.probabilities  # 확률 정보 추가
                        }
                        event_data = self.event_manager.process_classification_result(event_result_data)
                        if event_data:
                            logging.info(f"Event generated: {event_data.event_type.value}")
                            # 시각화 모듈에 이벤트 전달
                            if hasattr(self, '_current_visualizer') and self._current_visualizer:
                                self._current_visualizer.update_event_history(event_data.to_dict())
                    
                    logging.info(f"Async Window {window_id} classified: {result.prediction} ({result.confidence:.3f}) in {classification_time*1000:.1f}ms")
                else:
                    logging.warning(f"Invalid classification result for window {window_id}: {result}")
                
                self.classification_queue.task_done()
                
            except Empty:
                continue  # 타임아웃, 계속 대기
            except Exception as e:
                logging.error(f"Classification worker error: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
        
        logging.info("Classification worker terminated")
    
    def _add_stage_timing(self, stage_name: str, timing: float, max_history: int = 30):
        """스테이지 타이밍 추가 (메모리 효율화)"""
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
        
        self.stage_timings[stage_name].append(timing)
        
        # 최근 N개만 유지하여 메모리 절약
        if len(self.stage_timings[stage_name]) > max_history:
            self.stage_timings[stage_name].pop(0)

    def _log_detailed_performance_analysis(self):
        """상세 성능 분석 로깅"""
        try:
            stage_fps = self.get_stage_fps()
            
            logging.info("=== DETAILED PERFORMANCE ANALYSIS ===")
            for stage_name, fps_value in stage_fps.items():
                if stage_name in self.stage_timings and self.stage_timings[stage_name]:
                    recent_times = self.stage_timings[stage_name][-10:]  # 최근 10개
                    avg_time = sum(recent_times) / len(recent_times)
                    min_time = min(recent_times)
                    max_time = max(recent_times)
                    
                    logging.info(f"{stage_name.upper()}: {fps_value:.1f} FPS | "
                               f"Avg: {avg_time*1000:.2f}ms | "
                               f"Min: {min_time*1000:.2f}ms | "
                               f"Max: {max_time*1000:.2f}ms | "
                               f"Count: {len(self.stage_timings[stage_name])}")
            
            # 큐 상태 정보
            logging.info(f"Classification Queue Size: {self.classification_queue.qsize()}")
            logging.info(f"Classification Results Count: {len(self.classification_results)}")
            logging.info("============================================")
            
        except Exception as e:
            logging.error(f"Error in detailed performance analysis: {e}")
    
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int):
        """단일 프레임 처리"""
        frame_start_time = time.time()
        
        try:
            # 포즈 추정
            pose_start = time.time()
            frame_poses = self.pose_estimator.process_frame(frame, frame_idx)
            pose_time = time.time() - pose_start
            self._add_stage_timing('pose_estimation', pose_time)
            if frame_idx < 5:  # 처음 5프레임만 로깅
                logging.info(f"After pose estimation - frame {frame_idx}: {len(frame_poses.persons)} persons")
            
            # RTMO 원본 데이터 저장 (트래킹 전) - 최적화된 복사
            copy_start = time.time()
            import copy
            rtmo_original_poses = copy.deepcopy(frame_poses)
            copy_time = time.time() - copy_start
            self._add_stage_timing('deepcopy', copy_time)
            
            # 메모리 효율화 - 실시간 모드에서는 최근 100프레임만 유지
            if self.mode == 'realtime':
                self.rtmo_poses_results.append(rtmo_original_poses)
                if len(self.rtmo_poses_results) > 100:
                    self.rtmo_poses_results.pop(0)
            else:
                self.rtmo_poses_results.append(rtmo_original_poses)
            
            # 트래킹
            if self.tracker:
                track_start = time.time()
                frame_poses = self.tracker.track_frame_poses(frame_poses)
                track_time = time.time() - track_start
                self._add_stage_timing('tracking', track_time)
                if frame_idx < 5:  # 처음 5프레임만 로깅
                    logging.info(f"After tracking - frame {frame_idx}: {len(frame_poses.persons)} persons")
            
            # 스코어링
            if self.scorer:
                score_start = time.time()
                frame_poses = self.scorer.score_frame_poses(frame_poses)
                score_time = time.time() - score_start
                self._add_stage_timing('scoring', score_time)
                if frame_idx < 5:  # 처음 5프레임만 로깅
                    logging.info(f"After scoring - frame {frame_idx}: {len(frame_poses.persons)} persons")
            
            # 포즈 결과 저장 (시각화용) - 메모리 효율화
            if self.mode == 'realtime':
                # 실시간 모드에서는 최근 100프레임만 유지 (메모리 절약)
                self.frame_poses_results.append(frame_poses)
                if len(self.frame_poses_results) > 100:
                    self.frame_poses_results.pop(0)
            else:
                # 분석 모드에서는 전체 저장
                self.frame_poses_results.append(frame_poses)
            
            # 윈도우 처리기에 프레임 추가하고 새 윈도우들 가져오기  
            new_windows = self.window_processor.add_frame(frame_poses)
            
            # 새로 생성된 윈도우들 즉시 처리
            for window in new_windows:
                self._process_window(window, frame_idx)
            
            
            # 성능 통계 업데이트
            processing_time = time.time() - frame_start_time
            self.performance_tracker.update(processing_time)
            
            # 상세 성능 분석 (매 60프레임마다 로깅 - 오버헤드 감소)
            if frame_idx % 60 == 0:
                self._log_detailed_performance_analysis()
            
        except Exception as e:
            import traceback
            logging.error(f"Frame processing error: {e}")
            logging.error(f"Detailed traceback: {traceback.format_exc()}")
    
    def _process_window(self, window: 'WindowAnnotation', frame_idx: int):
        """단일 윈도우 처리 - 비동기 분류 활용"""
        try:
            # 비동기 분류 스레드가 활성화된 경우 큐에 작업 추가
            if self.classification_thread and self.classification_thread.is_alive():
                try:
                    # 윈도우 ID 생성 (고유 식별자)
                    window_id = getattr(window, 'window_idx', frame_idx)
                    
                    # 분류 작업을 큐에 추가 (논블로킹)
                    self.classification_queue.put((window, window_id), timeout=0.01)
                    logging.info(f"[ASYNC] Window {window_id} queued for async classification")
                    
                except Exception as e:
                    logging.warning(f"Failed to queue window for async classification: {e}, falling back to sync")
                    # 큐가 가득 찬 경우 동기 방식으로 폴백
                    self._process_window_sync(window, frame_idx)
            else:
                # 비동기 스레드가 없으면 동기 방식 사용
                self._process_window_sync(window, frame_idx)
                
        except Exception as e:
            logging.error(f"Window processing error: {e}")
    
    def _process_window_sync(self, window: 'WindowAnnotation', frame_idx: int):
        """동기 방식 윈도우 처리 (폴백용)"""
        try:
            # 분류 수행
            class_start = time.time()
            result = self.classifier.classify_window(window)
            class_time = time.time() - class_start
            self._add_stage_timing('classification', class_time)
            
            if result:
                # STGCN 내부에서 관리하는 윈도우 번호 가져오기
                display_id = getattr(result, 'metadata', {}).get('display_id', 0)
                
                # 분류 결과 저장
                classification_data = {
                    'display_id': display_id,  # STGCN에서 관리하는 윈도우 번호
                    'timestamp': time.time(),
                    'frame_idx': frame_idx,
                    'window_start': window.start_frame if hasattr(window, 'start_frame') else 0,
                    'window_end': window.end_frame if hasattr(window, 'end_frame') else 0,
                    'prediction': result.prediction,
                    'predicted_class': result.get_predicted_label(),
                    'confidence': result.confidence,
                    'probabilities': result.probabilities,
                    'processed': False  # 시각화 처리용
                }
                
                self.classification_results.append(classification_data)
                self.performance_stats['windows_classified'] += 1
                
                # 분석 모드에서만 window_processor에 분류 결과 추가
                # 실시간 모드에서는 메인 루프에서 이미 처리함 (중복 방지)
                if self.processing_mode == 'analysis':
                    self.window_processor.add_window_result(display_id, classification_data)
                
                logging.info(f"Sync Pipeline processed window {display_id}: {result.get_predicted_label()} (confidence: {result.confidence:.3f})")
            
        except Exception as e:
            logging.error(f"Sync window processing error: {e}")
    
    
    
    
    def _clear_queues(self):
        """결과 큐 초기화"""
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
    
    def get_latest_results(self, max_count: int = 10) -> List[ClassificationResult]:
        """최신 결과 가져오기"""
        results = []
        for _ in range(min(max_count, self.result_queue.qsize())):
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                break
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_tracker.get_stats()
        
        # 구간별 성능 계산
        stage_stats = {}
        for stage, timings in self.stage_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                stage_fps = 1.0 / avg_time if avg_time > 0 else 0.0
                stage_stats[f'{stage}_avg_time'] = avg_time
                stage_stats[f'{stage}_fps'] = stage_fps
                stage_stats[f'{stage}_total_calls'] = len(timings)
            else:
                stage_stats[f'{stage}_avg_time'] = 0.0
                stage_stats[f'{stage}_fps'] = 0.0
                stage_stats[f'{stage}_total_calls'] = 0
        
        # 전체 파이프라인 FPS 계산
        pipeline_fps = 1.0 / stats['avg_time'] if stats['avg_time'] > 0 else 0.0
        target_fps = getattr(self.config, 'target_fps', 30.0)
        
        # 이벤트 상태 추가
        event_stats = {}
        if self.event_manager:
            event_status = self.event_manager.get_event_status()
            event_stats = {
                'event_active': event_status.get('event_active', False),
                'event_duration': event_status.get('event_duration'),
                'total_events': event_status.get('total_events', 0),
                'consecutive_violence': event_status.get('consecutive_violence', 0),
                'consecutive_normal': event_status.get('consecutive_normal', 0)
            }
        
        stats.update({
            'frames_processed': stats['total_processed'],
            'fps': min(pipeline_fps, target_fps * 3),  # 파이프라인 전체 FPS
            'pipeline_fps': pipeline_fps,
            'target_fps': target_fps,
            'windows_classified': self.performance_stats['windows_classified'],
            'total_alerts': self.performance_stats['total_alerts'],
            'classification_count': len(self.classification_results),
            **stage_stats,  # 구간별 통계 추가
            **event_stats   # 이벤트 통계 추가
        })
        return stats
    
    def get_classification_results(self) -> List[Dict[str, Any]]:
        """분류 결과 반환"""
        return self.classification_results.copy()
    
    def get_frame_poses_results(self) -> List[FramePoses]:
        """포즈 결과 반환 (시각화용)"""
        return self.frame_poses_results.copy()
    
    def get_rtmo_poses_results(self) -> List[FramePoses]:
        """RTMO 원본 포즈 결과 반환 (트래킹 전)"""
        return self.rtmo_poses_results.copy()
    
    
    def start_realtime_display(self, 
                             input_source: Union[str, int],
                             display_width: int = 1280,
                             display_height: int = 720,
                             save_output: bool = False,
                             output_path: Optional[str] = None) -> bool:
        """
        실시간 디스플레이 모드 시작
        
        Args:
            input_source: 입력 소스 (파일, RTSP URL, 웹캠 인덱스)
            display_width: 디스플레이 창 너비
            display_height: 디스플레이 창 높이
            save_output: 실시간 결과를 비디오로 저장할지 여부
            output_path: 저장할 비디오 파일 경로
        
        Returns:
            성공 여부
        """
        try:
            # 실시간 입력 관리자 초기화
            input_manager = RealtimeInputManager(
                input_source=input_source,
                buffer_size=5,
                target_fps=getattr(self.config, 'target_fps', 30)
            )
            
            # 모드별 시각화 도구 초기화
            max_persons = getattr(self.config, 'max_persons', 4)
            processing_mode = getattr(self.config, 'processing_mode', 'realtime')
            
            window_title = f"Violence Detection - {processing_mode.upper()}"
            
            # 설정에서 confidence_threshold 가져오기
            if hasattr(self.config, 'classifier_confidence_threshold'):
                confidence_threshold = self.config.classifier_confidence_threshold
            else:
                # 딕셔너리 형태 설정에서 가져오기
                action_config = self.config.get('models', {}).get('action_classification', {})
                confidence_threshold = action_config.get('confidence_threshold', 0.4)
            
            visualizer = RealtimeVisualizer(
                window_name=window_title,
                display_width=display_width,
                display_height=display_height,
                fps_limit=getattr(self.config, 'target_fps', 30),
                save_output=save_output,
                output_path=output_path,
                max_persons=max_persons,
                processing_mode=processing_mode,
                confidence_threshold=confidence_threshold
            )
            
            # 시각화 모듈 참조 저장 (이벤트 전달용)
            self._current_visualizer = visualizer
            
            logging.info(f"Visualizer initialized with {processing_mode} mode")
            
            # 입력 소스 시작
            if not input_manager.start():
                logging.error("Failed to start input manager")
                return False
            
            # 파이프라인 초기화
            if not self.initialize_pipeline():
                logging.error("Failed to initialize pipeline")
                input_manager.stop()
                return False
            
            # 디스플레이 창 시작
            visualizer.start_display()
            
            logging.info("Realtime display mode started")
            
            # 메인 처리 루프
            try:
                self._realtime_processing_loop(input_manager, visualizer)
            finally:
                # 비디오 끝날 때 마지막 윈도우 처리
                if len(self.frame_buffer) > 0:
                    window_size = 100
                    stride = 50
                    
                    # 마지막 완전한 윈도우가 생성되지 않은 경우에만 처리
                    last_window_start = (len(self.classification_results) - 1) * stride if self.classification_results else 0
                    remaining_frames = len(self.frame_buffer) - last_window_start - window_size
                    
                    # 40% 이상의 프레임이 남아있으면 마지막 윈도우 생성 (16프레임 이상)
                    if remaining_frames >= window_size * 0.4:
                        window_number = len(self.classification_results) + 1
                        logging.info(f"Processing final window {window_number} with remaining frames")
                        
                        # 마지막 프레임들을 100프레임으로 패딩
                        final_frames = self.frame_buffer.copy()
                        if len(final_frames) < window_size:
                            last_frame = final_frames[-1]
                            while len(final_frames) < window_size:
                                import copy
                                padded_frame = copy.deepcopy(last_frame)
                                padded_frame.frame_idx = final_frames[-1].frame_idx + 1
                                final_frames.append(padded_frame)
                        
                        self._process_classification_window(final_frames[-window_size:], window_number, visualizer)
                
                # 정리
                visualizer.stop_display()
                input_manager.stop()
                logging.info("Realtime display mode stopped")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in realtime display mode: {e}")
            return False
    
    def _realtime_processing_loop(self, 
                                input_manager: RealtimeInputManager,
                                visualizer: RealtimeVisualizer):
        """실시간 처리 메인 루프"""
        frame_count = 0
        stats_frame_count = 0  # 통계용 별도 프레임 카운터
        last_stats_time = time.time()
        stats_interval = 3.0  # 3초마다 통계 출력
        
        while input_manager.is_alive():
            # 프레임 가져오기
            frame_data = input_manager.get_latest_frame()
            if frame_data is None:
                time.sleep(0.001)  # 1ms 대기
                continue
            
            frame, frame_number, timestamp = frame_data
            frame_count += 1
            stats_frame_count += 1
            
            # 주요 프레임에서 디버깅 로깅 - 더 많은 프레임 추가
            if frame_count % 25 == 0 or frame_count in [140, 145, 148, 149, 150, 151, 152, 155, 160, 200, 210]:
                logging.info(f"DEBUG: Processing frame {frame_count}, classification_results: {len(self.classification_results)}")
            
            # 프레임 처리 - 전체 타이밍 측정
            process_start = time.time()
            poses = None
            classification = None
            
            # 각 단계별 세밀한 타이밍 측정
            timing_breakdown = {}
            
            try:
                # 1. 포즈 추정
                pose_start = time.time()
                pose_result = self.pose_estimator.process_frame(frame, frame_count)
                pose_time = time.time() - pose_start
                timing_breakdown['pose_estimation'] = pose_time
                self.stage_timings['pose_estimation'].append(pose_time)
                
                # 2. 포즈 결과 변환
                convert_start = time.time()
                if pose_result:
                    poses = self._convert_pose_result(pose_result, frame_count)
                convert_time = time.time() - convert_start
                timing_breakdown['pose_conversion'] = convert_time
                
                # 3. 트래킹
                track_start = time.time()
                if poses and self.tracker:
                    poses = self.tracker.track_frame_poses(poses)
                track_time = time.time() - track_start
                timing_breakdown['tracking'] = track_time
                self.stage_timings['tracking'].append(track_time)
                
                # 4. 스코어링
                score_start = time.time()
                if poses and self.scorer:
                    poses = self.scorer.score_frame_poses(poses)
                score_time = time.time() - score_start
                timing_breakdown['scoring'] = score_time
                self.stage_timings['scoring'].append(score_time)
                
                # 5. 윈도우 처리 및 버퍼 관리
                buffer_start = time.time()
                if poses:
                    # 프레임 버퍼에 추가 (포즈 추정 결과)
                    self._add_frame_to_buffer(poses)
                    
                    # 윈도우 크기(100)만큼 쌓이면 분류 처리
                    window_size = 100
                    stride = 50
                    
                    # 디버깅 로그
                    if frame_count % 20 == 0:  # 20프레임마다 로그
                        logging.info(f"Frame {frame_count}: buffer_size={len(self.frame_buffer)}, "
                                   f"classification_count={len(self.classification_results)}")
                    
                    # 첫 번째 윈도우: 0-99 프레임 (100프레임에서 생성)
                    if len(self.frame_buffer) >= window_size and len(self.classification_results) == 0:
                        logging.info(f"Creating first window at frame {frame_count} with {len(self.frame_buffer)} frames")
                        first_window_frames = self.frame_buffer[:window_size]  # 처음 100프레임 사용
                        self._process_classification_window(first_window_frames, 1, visualizer)
                    
                    # 이후 윈도우들: stride(50) 간격으로만 생성 (140+, 190+, 240+...)
                    elif frame_count >= window_size + stride * len(self.classification_results) - 10:  # 10프레임 일찍 시작
                        # 윈도우 생성 시점: frame_count 기준으로 100 + 50*n - 10 (140+, 190+, 240+...)
                        expected_frame_count = window_size + stride * len(self.classification_results)
                        current_window_num = len(self.classification_results) + 1
                        
                        # 아직 해당 윈도우가 생성되지 않았고, 충분한 프레임이 있으면 생성
                        if len(self.frame_buffer) >= window_size:
                            logging.info(f"Creating window {current_window_num} at frame {frame_count} (expected around {expected_frame_count})")
                            recent_frames = self.frame_buffer[-window_size:]
                            self._process_classification_window(recent_frames, current_window_num, visualizer)
                        else:
                            logging.warning(f"Not enough frames in buffer for window {current_window_num}: {len(self.frame_buffer)}")
                            
                        # 디버깅용 로그 (한 번만 출력)
                        if frame_count == window_size + stride * len(self.classification_results) - 10:
                            logging.info(f"DEBUG: Window {current_window_num} creation attempt - frame_count={frame_count}, expected={expected_frame_count}, buffer={len(self.frame_buffer)}")
                
                buffer_time = time.time() - buffer_start
                timing_breakdown['buffer_management'] = buffer_time
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")
            
            # 6. 비동기 분류 결과 처리
            async_start = time.time()
            processed_classifications = self._process_async_classification_results(visualizer)
            async_time = time.time() - async_start
            timing_breakdown['async_classification'] = async_time
            
            pure_processing_time = time.time() - process_start  # 시각화 제외한 순수 처리 시간
            
            # 상세 타이밍 로깅 (50프레임마다)
            if frame_count % 50 == 0 and timing_breakdown:
                logging.info(f"=== Frame {frame_count} Detailed Timing ===")
                for stage, duration in timing_breakdown.items():
                    logging.info(f"  {stage}: {duration*1000:.2f}ms")
                logging.info(f"  TOTAL: {pure_processing_time*1000:.2f}ms ({1.0/pure_processing_time:.1f} FPS)")
                logging.info("=====================================")
            
            # 추가 정보 구성 (시각화 제외한 순수 FPS 계산)
            pure_fps = 1.0 / pure_processing_time if pure_processing_time > 0 else 0
            
            # 단계별 FPS 계산
            stage_fps = self.get_stage_fps()
            
            additional_info = {
                'fps': pure_fps,  # 오버레이 표시용 순수 FPS (시각화 제외)
                'processing_fps': pure_fps,  # 시각화 제외한 파이프라인 FPS
                'pure_processing_time': pure_processing_time,  # 순수 처리 시간
                'total_persons': len(poses.persons) if poses and poses.persons else 0,
                'buffer_size': len(self.frame_buffer),
                'queue_size': input_manager.get_queue_size(),
                # 단계별 FPS 추가
                'pose_estimation_fps': stage_fps.get('pose_estimation', 0),
                'tracking_fps': stage_fps.get('tracking', 0),
                'scoring_fps': stage_fps.get('scoring', 0),
                'classification_fps': stage_fps.get('classification', 0)
            }
            
            # 오버레이 데이터 구성 (분류 결과 포함)
            overlay_data = None
            if self.classification_results:
                window_results = []
                for result in self.classification_results:
                    if isinstance(result, dict):
                        # 기존 동기 결과 형태
                        if 'display_id' in result and 'predicted_class' in result:
                            window_results.append({
                                'window_id': result['display_id'],
                                'classification': {
                                    'predicted_class': result['predicted_class'],
                                    'confidence': result['confidence'],
                                    'probabilities': result.get('probabilities', [])
                                }
                            })
                        # 비동기 결과 형태는 _process_async_classification_results에서 처리됨
                
                if window_results:
                    overlay_data = {
                        'show_keypoints': True,
                        'show_tracking_ids': True,
                        'show_classification': True,
                        'window_results': window_results
                    }
            
            if not visualizer.show_frame(frame, poses, None, additional_info, overlay_data):
                logging.info("Display terminated by user")
                break
            
            # 주기적 통계 출력
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                self._log_realtime_stats(input_manager, visualizer, stats_frame_count, current_time - last_stats_time)
                last_stats_time = current_time
                stats_frame_count = 0  # 통계용 카운터만 리셋
    
    def _log_realtime_stats(self, 
                          input_manager: RealtimeInputManager,
                          visualizer: RealtimeVisualizer,
                          frames_processed: int,
                          interval: float):
        """실시간 통계 로깅"""
        input_stats = input_manager.get_statistics()
        viz_stats = visualizer.get_statistics()
        
        processing_fps = frames_processed / interval if interval > 0 else 0
        
        # 상세 성능 통계 가져오기
        detailed_stats = self.get_detailed_performance_stats()
        stage_fps = detailed_stats['stage_fps']
        
        logging.info(f"=== Realtime Performance Stats ===")
        logging.info(f"Overall Processing FPS: {processing_fps:.1f} (전체 파이프라인)")
        logging.info(f"Pure Processing FPS: {detailed_stats['processing_fps']:.1f} (시각화 제외)")
        logging.info(f"Display FPS: {viz_stats['current_fps']:.1f}")
        logging.info(f"Capture FPS: {input_stats['capture_fps']:.1f}")
        logging.info(f"=== Stage-wise FPS ===")
        logging.info(f"Pose Estimation: {stage_fps.get('pose_estimation', 0):.1f} FPS")
        logging.info(f"Tracking: {stage_fps.get('tracking', 0):.1f} FPS") 
        logging.info(f"Scoring: {stage_fps.get('scoring', 0):.1f} FPS")
        logging.info(f"Classification: {stage_fps.get('classification', 0):.1f} FPS")
        logging.info(f"=== Other Stats ===")
        logging.info(f"Dropped frames: {input_stats['dropped_frames']} ({input_stats['drop_rate']:.1%})")
        logging.info(f"Queue size: {input_stats['queue_size']}")
        logging.info(f"Total classifications: {len(self.classification_results)}")
        logging.info(f"Buffer size: {len(self.frame_buffer)}")
        logging.info("====================")
    
    def _convert_pose_result(self, pose_result: FramePoses, frame_count: int) -> FramePoses:
        """포즈 결과를 FramePoses 형태로 변환"""
        # pose_result가 이미 FramePoses 형태라면 그대로 반환
        if isinstance(pose_result, FramePoses):
            return pose_result
        
        # 변환이 필요한 경우 여기에 추가 구현
        return pose_result
    
    def _add_frame_to_buffer(self, poses: FramePoses):
        """프레임을 버퍼에 추가"""
        self.frame_buffer.append(poses)
        
        # 버퍼 크기 제한 (메모리 관리) - 최대 200프레임 유지
        max_buffer_size = 200
        if len(self.frame_buffer) > max_buffer_size:
            self.frame_buffer.pop(0)  # 가장 오래된 프레임 제거
    
    def _process_classification_window(self, frames: List[FramePoses], window_number: int, visualizer):
        """윈도우 분류 처리 - 비동기 처리로 변경"""
        try:
            logging.info(f"Processing window {window_number} with {len(frames)} frames")
            
            # 1. 복합점수 계산
            from utils.pipeline_common import PipelineCommonUtils
            scored_frames = PipelineCommonUtils.apply_composite_scores(frames, self.scorer)
            
            # 2. 윈도우 어노테이션 생성 (MMAction2 표준 형식)
            window_annotation = PipelineCommonUtils.create_window_annotation(scored_frames, window_number)
            
            # 3. 비동기 분류 요청 (기존 동기 처리를 비동기로 변경)
            if self.classifier:
                try:
                    # 비동기 큐에 작업 추가 (논블로킹)
                    self.classification_queue.put_nowait((window_annotation, window_number))
                    logging.info(f"Window {window_number} queued for async classification")
                except:
                    # 큐가 가득찬 경우 가장 오래된 작업 제거 후 추가
                    try:
                        self.classification_queue.get_nowait()  # 가장 오래된 작업 제거
                        self.classification_queue.put_nowait((window_annotation, window_number))
                        logging.warning(f"Classification queue full, dropped oldest task for window {window_number}")
                    except:
                        logging.warning(f"Failed to queue window {window_number} for classification")
            else:
                logging.warning(f"No classifier available for window {window_number}")
                
        except Exception as e:
            import traceback
            logging.error(f"Error processing window {window_number}: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
    
    def _process_async_classification_results(self, visualizer=None):
        """비동기 분류 결과 처리"""
        processed_results = []
        
        # 새로운 분류 결과들을 처리
        for i in range(len(self.classification_results) - 1, -1, -1):
            result_data = self.classification_results[i]
            
            # 이미 처리된 결과인지 확인 (기존 동기 결과와의 호환성)
            if isinstance(result_data, dict):
                if result_data.get('processed', False):
                    continue
                    
                # 새로운 비동기 결과 형태인지 확인
                if 'window_id' in result_data and 'result' in result_data:
                    # 비동기 결과 처리
                    window_id = result_data['window_id']
                    result = result_data['result']
                    
                    # display_id 확보 - STGCN metadata에서 가져오거나 window_id 사용
                    display_id = window_id
                    if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                        display_id = result.metadata.get('display_id', window_id)
                    
                    classification = {
                        'display_id': display_id,
                        'predicted_class': result.get_predicted_label(),
                        'confidence': result.confidence,
                        'probabilities': result.probabilities,
                        'window_start': display_id * 50,
                        'window_end': display_id * 50 + 100,
                        'timestamp': result_data.get('timestamp', time.time()),
                        'processing_time': result_data.get('processing_time', 0)
                    }
                    
                    # 시각화 처리
                    if self.mode == "realtime" and visualizer:
                        visualizer.update_classification_history(classification)
                        logging.info(f"[ASYNC] Window {display_id} result processed for visualization")
                    
                    # 처리 완료 마킹
                    result_data['processed'] = True
                    processed_results.append(classification)
                
                elif 'display_id' in result_data:
                    # 기존 동기 결과 (이미 처리된 형태)
                    if self.mode == "realtime" and visualizer and not result_data.get('vis_processed', False):
                        visualizer.update_classification_history(result_data)
                        logging.info(f"[SYNC] Window {result_data['display_id']} result processed for visualization")
                        result_data['vis_processed'] = True
                        processed_results.append(result_data)
                    continue
                
                else:
                    # 알 수 없는 결과 형태 - 경고 로그
                    logging.warning(f"Unknown classification result format: {list(result_data.keys())}")
                    result_data['processed'] = True  # 무한 처리 방지
        
        return processed_results
    
    
    def process_video_analysis_mode(self, video_path: str) -> Dict[str, Any]:
        """분석 모드 전용 비디오 처리 - 한 번에 모든 프레임 처리"""
        logging.info(f"Starting analysis mode processing for: {video_path}")
        
        try:
            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video: {video_path}")
                return {'success': False, 'error': 'Cannot open video'}
            
            # 비디오 정보
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logging.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
            
            # 성능 추적 초기화
            self.performance_tracker.reset()
            
            # 1단계: 모든 프레임 포즈 추정 및 트래킹
            all_frame_poses = []
            raw_pose_data = []  # 원본 포즈 데이터 저장
            frame_idx = 0
            
            logging.info("Phase 1: Pose estimation and tracking")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # 포즈 추정
                pose_start = time.time()
                pose_results = self.pose_estimator.extract_poses(frame, frame_idx)
                pose_time = time.time() - pose_start
                
                # 원본 포즈 데이터 저장 (PKL용)
                raw_pose_frame = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'pose_results': pose_results,
                    'image_shape': frame.shape[:2]
                }
                raw_pose_data.append(raw_pose_frame)
                
                # FramePoses 생성 (트래킹과 스코어링 이전)
                frame_poses = FramePoses(
                    frame_idx=frame_idx,
                    persons=pose_results,
                    timestamp=frame_idx / fps,
                    image_shape=frame.shape[:2]
                )
                
                # 트래킹
                track_start = time.time()
                tracked_frame_poses = self.tracker.track_frame_poses(frame_poses)
                track_time = time.time() - track_start
                
                # 스코어링
                score_start = time.time()
                scored_frame_poses = self.scorer.score_frame_poses(tracked_frame_poses)
                score_time = time.time() - score_start
                
                # 최종 FramePoses
                frame_poses = scored_frame_poses
                
                all_frame_poses.append(frame_poses)
                
                # 성능 추적
                total_time = time.time() - start_time
                self.performance_tracker.update(total_time)
                
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logging.info(f"Processed {frame_idx}/{total_frames} frames")
            
            cap.release()
            logging.info(f"Phase 1 completed: {len(all_frame_poses)} frames processed")
            
            # 2단계: 윈도우 생성 및 분류
            logging.info("Phase 2: Window creation and classification")
            
            # 윈도우 프로세서 설정을 분석 모드로 변경
            self.window_processor.set_processing_mode('analysis')
            
            # 새로운 분석 모드 메서드 사용
            windows = self.window_processor.process_all_frames_analysis_mode(all_frame_poses)
            
            logging.info(f"Created {len(windows)} windows for classification")
            
            # 3단계: 윈도우 분류
            classification_results = []
            analysis_events = []  # 분석 모드에서 발생한 이벤트들
            
            for i, window in enumerate(windows):
                try:
                    class_start = time.time()
                    result = self.classifier.classify_window(window)
                    class_time = time.time() - class_start
                    
                    # 성능 추적 (분류 시간은 별도로 기록하지 않고 전체 통계에 포함)
                    
                    if result:
                        classification_results.append(result)
                        self.window_processor.add_window_result(i, result.to_dict())
                        
                        # 분석 모드에서도 이벤트 처리
                        if self.event_manager:
                            # 이벤트 시스템용 데이터 변환
                            prediction_name = 'violence' if result.prediction == 1 else 'normal'
                            
                            event_result_data = {
                                'window_id': i,
                                'prediction': prediction_name,  # 문자열로 변환
                                'confidence': result.confidence,
                                'timestamp': class_start + class_time/2,  # 윈도우 중간 시점으로 추정
                                'class_index': result.prediction,  # 원본 인덱스도 보존
                                'probabilities': getattr(result, 'probabilities', [])  # 확률 정보 추가
                            }
                            event_data = self.event_manager.process_classification_result(event_result_data)
                            if event_data:
                                analysis_events.append(event_data)
                        
                        logging.info(f"Window {i} classified: {result.get_predicted_label()} ({result.confidence:.3f})")
                    
                except Exception as e:
                    logging.error(f"Error classifying window {i}: {e}")
            
            # 성능 통계 완료
            performance_stats = self.performance_tracker.get_stats()
            
            # 이벤트 관련 통계 추가
            event_stats = {
                'total_alerts': 0,
                'total_events': len(analysis_events),
                'event_history': [event.to_dict() for event in analysis_events]
            }
            
            if self.event_manager:
                event_status = self.event_manager.get_event_status()
                event_stats.update({
                    'total_alerts': event_status.get('total_events', 0),
                    'final_event_active': event_status.get('event_active', False)
                })
            
            # 추가 성능 통계 계산
            performance_stats.update({
                'frames_processed': total_frames,
                'windows_classified': len(classification_results),
                'fps': performance_stats['total_processed'] / performance_stats['total_time'] if performance_stats['total_time'] > 0 else 0,
                'pipeline_fps': performance_stats['total_processed'] / performance_stats['total_time'] if performance_stats['total_time'] > 0 else 0,
                'target_fps': 30.0,
                'classification_count': len(classification_results),
                **event_stats
            })
            
            # 결과 정리
            result = {
                'success': True,
                'total_frames': total_frames,
                'windows_created': len(windows),
                'classifications': [r.to_dict() for r in classification_results],
                'performance_stats': performance_stats,
                'video_info': {
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration': total_frames / fps if fps > 0 else 0
                },
                # PKL 파일용 데이터 추가
                'raw_pose_results': raw_pose_data,  # 포즈추정 원본 데이터
                'processed_frame_poses': all_frame_poses  # 트래킹+복합점수 계산 후 데이터
            }
            
            logging.info(f"Analysis mode completed successfully: {len(classification_results)} classifications")
            return result
            
        except Exception as e:
            logging.error(f"Error in analysis mode processing: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_stage_fps(self) -> Dict[str, float]:
        """각 처리 단계별 FPS 계산 (최근 30프레임 기준)"""
        stage_fps = {}
        
        try:
            for stage_name, timings in self.stage_timings.items():
                if len(timings) > 0:
                    # 최근 fps_window_size개 타이밍만 사용
                    recent_timings = timings[-self.fps_window_size:]
                    if len(recent_timings) > 0:
                        avg_time = sum(recent_timings) / len(recent_timings)
                        stage_fps[stage_name] = 1.0 / avg_time if avg_time > 0 else 0.0
                    else:
                        stage_fps[stage_name] = 0.0
                else:
                    stage_fps[stage_name] = 0.0
            
            # 디버깅용 로그
            logging.debug(f"Stage timings lengths: {[(k, len(v)) for k, v in self.stage_timings.items()]}")
            
        except Exception as e:
            logging.error(f"Error calculating stage FPS: {e}")
            stage_fps = {k: 0.0 for k in self.stage_timings.keys()}
        
        return stage_fps
    
    def get_detailed_performance_stats(self) -> Dict[str, Any]:
        """상세 성능 통계 반환"""
        stage_fps = self.get_stage_fps()
        
        # 전체 처리 시간 계산 (시각화 제외)
        total_processing_times = []
        min_length = min([len(timings) for timings in self.stage_timings.values() if len(timings) > 0] + [0])
        
        for i in range(max(0, min_length - self.fps_window_size), min_length):
            frame_total = 0
            for stage_timings in self.stage_timings.values():
                if i < len(stage_timings):
                    frame_total += stage_timings[i]
            if frame_total > 0:
                total_processing_times.append(frame_total)
        
        # 전체 처리 FPS (시각화 제외)
        if total_processing_times:
            avg_total_time = sum(total_processing_times) / len(total_processing_times)
            processing_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0.0
        else:
            processing_fps = 0.0
        
        return {
            'stage_fps': stage_fps,
            'processing_fps': processing_fps,  # 시각화 제외한 순수 처리 FPS
            'pipeline_fps': self.performance_tracker.get_fps(),  # 전체 pipeline FPS (기존)
            'stage_counts': {k: len(v) for k, v in self.stage_timings.items()}
        }
    
    def cleanup(self):
        """파이프라인 정리"""
        try:
            logging.info("Cleaning up inference pipeline...")
            
            # 비동기 분류 스레드 중지
            self._stop_classification_thread()
            
            # 모듈별 정리
            if hasattr(self.classifier, 'cleanup'):
                self.classifier.cleanup()
            
            if hasattr(self.pose_estimator, 'cleanup'):
                self.pose_estimator.cleanup()
                
            # 버퍼와 큐 정리
            self.frame_buffer.clear()
            self.classification_results.clear()
            
            # 큐 정리
            while not self.classification_queue.empty():
                try:
                    self.classification_queue.get_nowait()
                except:
                    break
            
            # GPU 메모리 정리
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logging.info("GPU memory cleared")
            
            logging.info("Pipeline cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during pipeline cleanup: {e}")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            self.cleanup()
        except:
            pass