"""
실시간 추론 파이프라인 메인 클래스
"""

import time
import threading
import cv2
import numpy as np
import logging
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
        
        # 실시간 처리용 큐들
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.pose_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=100)
        
        # 윈도우 버퍼
        self.window_buffer: List[FramePoses] = []
        self.frame_buffer: List[FramePoses] = []  # 실시간 처리용 프레임 버퍼
        self.last_inference_frame = 0
        
        # 실시간 디스플레이용 최근 분류 결과 유지
        self.latest_classification = None
        
        # 스레드 관리
        self.processing_thread = None
        self.is_running = False
        
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
        
        # 분류 결과 저장
        self.classification_results = []
        
        # 포즈 결과 저장 (시각화용)
        self.frame_poses_results = []  # 트래킹 및 스코어링 완료된 데이터
        self.rtmo_poses_results = []   # RTMO 원본 포즈 데이터
        
        # 콜백 함수들
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.frame_callbacks: List[Callable[[np.ndarray, List[PersonPose]], None]] = []
        
        # 윈도우 크기 설정
        self.window_size = 100  # 기본값
        
        # 처리 모드 설정
        if hasattr(config, 'processing_mode'):
            self.processing_mode = config.processing_mode
        elif hasattr(config, 'get'):
            self.processing_mode = config.get('processing_mode', 'realtime')
        else:
            self.processing_mode = 'realtime'
    
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
                
                # 포즈 추정 설정
                pose_estimation = models.get('pose_estimation', {})
                pose_config = {
                    'model_name': pose_estimation.get('model_name', 'rtmo'),
                    'config_file': pose_estimation.get('config_file', ''),
                    'checkpoint_path': pose_estimation.get('checkpoint_path', ''),
                    'device': pose_estimation.get('device', 'cuda:0'),
                    'score_threshold': pose_estimation.get('score_threshold', 0.3),
                    'input_size': pose_estimation.get('input_size', [640, 640])
                }
                
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
            
            self._initialized = True
            logging.info("Pipeline initialization completed")
            return True
            
        except Exception as e:
            logging.error(f"Pipeline initialization failed: {e}")
            return False
    
    def start_realtime_processing(self, source: Union[str, int] = 0):
        """실시간 처리 시작"""
        if self.is_running:
            logging.warning("Pipeline is already running")
            return
        
        if not self.initialize_pipeline():
            raise RuntimeError("Failed to initialize pipeline")
        
        self.is_running = True
        self._clear_queues()
        
        # 백그라운드 스레드에서 처리 시작
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(source,),
            daemon=True
        )
        self.processing_thread.start()
        
        logging.info(f"Realtime processing started with source: {source}")
    
    def stop_realtime_processing(self):
        """실시간 처리 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self._clear_queues()
        logging.info("Realtime processing stopped")
    
    def _processing_loop(self, source: Union[str, int]):
        """메인 처리 루프"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logging.error(f"Failed to open video source: {source}")
            self.is_running = False
            return
        
        frame_idx = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 스킵 처리
                skip_frames = self.config.get('skip_frames', 1) if isinstance(self.config, dict) else getattr(self.config, 'skip_frames', 1)
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue
                
                # 입력 리사이즈
                resize_input = self.config.get('resize_input') if isinstance(self.config, dict) else getattr(self.config, 'resize_input', None)
                if resize_input:
                    frame = cv2.resize(frame, resize_input)
                
                # 프레임 처리
                self._process_single_frame(frame, frame_idx)
                frame_idx += 1
                
                # FPS 제어
                target_fps = self.config.get('target_fps', 30.0) if isinstance(self.config, dict) else getattr(self.config, 'target_fps', 30.0)
                time.sleep(1.0 / target_fps)
                
        except Exception as e:
            logging.error(f"Processing loop error: {e}")
        finally:
            # 마지막 윈도우 처리 (패딩 포함)
            try:
                final_windows = self.window_processor.finalize_processing()
                for window in final_windows:
                    self._process_window(window, frame_idx)
                    logging.info(f"Processed final padded window: {window.window_idx}")
            except Exception as e:
                logging.error(f"Error processing final windows: {e}")
            
            cap.release()
            self.is_running = False
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int):
        """단일 프레임 처리"""
        frame_start_time = time.time()
        
        try:
            # 포즈 추정
            pose_start = time.time()
            frame_poses = self.pose_estimator.process_frame(frame, frame_idx)
            pose_time = time.time() - pose_start
            self.stage_timings['pose_estimation'].append(pose_time)
            if frame_idx < 5:  # 처음 5프레임만 로깅
                logging.info(f"After pose estimation - frame {frame_idx}: {len(frame_poses.persons)} persons")
            
            # RTMO 원본 데이터 저장 (트래킹 전)
            import copy
            rtmo_original_poses = copy.deepcopy(frame_poses)
            self.rtmo_poses_results.append(rtmo_original_poses)
            
            # 트래킹
            if self.tracker:
                track_start = time.time()
                frame_poses = self.tracker.track_frame_poses(frame_poses)
                track_time = time.time() - track_start
                self.stage_timings['tracking'].append(track_time)
                if frame_idx < 5:  # 처음 5프레임만 로깅
                    logging.info(f"After tracking - frame {frame_idx}: {len(frame_poses.persons)} persons")
            
            # 스코어링
            if self.scorer:
                score_start = time.time()
                frame_poses = self.scorer.score_frame_poses(frame_poses)
                score_time = time.time() - score_start
                self.stage_timings['scoring'].append(score_time)
                if frame_idx < 5:  # 처음 5프레임만 로깅
                    logging.info(f"After scoring - frame {frame_idx}: {len(frame_poses.persons)} persons")
            
            # 포즈 결과 저장 (시각화용)
            self.frame_poses_results.append(frame_poses)
            
            # 윈도우 처리기에 프레임 추가하고 새 윈도우들 가져오기  
            new_windows = self.window_processor.add_frame(frame_poses)
            
            # 새로 생성된 윈도우들 즉시 처리
            for window in new_windows:
                self._process_window(window, frame_idx)
            
            # 콜백 호출
            for callback in self.frame_callbacks:
                callback(frame, frame_poses.persons)
            
            # 성능 통계 업데이트
            processing_time = time.time() - frame_start_time
            self.performance_tracker.update(processing_time)
            
        except Exception as e:
            import traceback
            logging.error(f"Frame processing error: {e}")
            logging.error(f"Detailed traceback: {traceback.format_exc()}")
    
    def _process_window(self, window: 'WindowAnnotation', frame_idx: int):
        """단일 윈도우 처리"""
        try:
            # 분류 수행
            class_start = time.time()
            result = self.classifier.classify_window(window)
            class_time = time.time() - class_start
            self.stage_timings['classification'].append(class_time)
            
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
                    'probabilities': result.probabilities
                }
                
                self.classification_results.append(classification_data)
                self.performance_stats['windows_classified'] += 1
                
                # 분석 모드에서만 window_processor에 분류 결과 추가
                # 실시간 모드에서는 메인 루프에서 이미 처리함 (중복 방지)
                if self.processing_mode == 'analysis':
                    self.window_processor.add_window_result(display_id, classification_data)
                
                logging.info(f"Pipeline processed window {display_id}: {result.get_predicted_label()} (confidence: {result.confidence:.3f})")
            
        except Exception as e:
            logging.error(f"Window processing error: {e}")
    
    def _run_inference(self, frame_idx: int):
        """윈도우 기반 추론 실행 (기존 호환성 유지)"""
        if len(self.window_buffer) < self.config.window_size:
            return
        
        try:
            # 윈도우 생성
            windows = self.window_processor.process_frames(self.window_buffer)
            
            for window in windows:
                # 분류 수행
                class_start = time.time()
                result = self.classifier.classify_window(window)
                class_time = time.time() - class_start
                self.stage_timings['classification'].append(class_time)
                
                # 분류 결과 저장
                classification_data = {
                    'timestamp': time.time(),
                    'frame_idx': frame_idx,
                    'window_start': window.start_frame if hasattr(window, 'start_frame') else 0,
                    'window_end': window.end_frame if hasattr(window, 'end_frame') else 0,
                    'prediction': result.prediction,
                    'predicted_class': result.get_predicted_label(),
                    'confidence': result.confidence,
                    'probabilities': result.probabilities
                }
                self.classification_results.append(classification_data)
                
                # 알림 체크
                if result.confidence >= self.config.alert_threshold:
                    alert = {
                        'timestamp': time.time(),
                        'frame_idx': frame_idx,
                        'alert_type': result.predicted_class_name,
                        'confidence': result.confidence,
                        'details': {
                            'window_info': {
                                'start_frame': window.start_frame if hasattr(window, 'start_frame') else 0,
                                'end_frame': window.end_frame if hasattr(window, 'end_frame') else 0,
                                'person_count': self._count_unique_persons([self.window_buffer[-1]])
                            },
                            'classification': result.to_dict()
                        }
                    }
                    
                    # 알림 콜백 호출
                    for callback in self.alert_callbacks:
                        callback(alert)
                    
                    self.performance_stats['total_alerts'] += 1
                
                # 결과 큐에 추가
                if not self.result_queue.full():
                    self.result_queue.put(result)
            
            self.performance_stats['windows_classified'] += len(windows)
            
        except Exception as e:
            logging.error(f"Inference error: {e}")
    
    def _count_unique_persons(self, poses: List[FramePoses]) -> int:
        """고유 인물 수 계산"""
        person_ids = set()
        for frame_poses in poses:
            for pose in frame_poses.persons:
                if pose.person_id is not None:
                    person_ids.add(pose.person_id)
        return len(person_ids)
    
    
    def _clear_queues(self):
        """큐 초기화"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        while not self.pose_queue.empty():
            try:
                self.pose_queue.get_nowait()
            except Empty:
                break
        
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
        
        stats.update({
            'frames_processed': stats['total_processed'],
            'fps': min(pipeline_fps, target_fps * 3),  # 파이프라인 전체 FPS
            'pipeline_fps': pipeline_fps,
            'target_fps': target_fps,
            'windows_classified': self.performance_stats['windows_classified'],
            'total_alerts': self.performance_stats['total_alerts'],
            'classification_count': len(self.classification_results),
            **stage_stats  # 구간별 통계 추가
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
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, List[PersonPose]], None]):
        """프레임 콜백 추가"""
        self.frame_callbacks.append(callback)
    
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
            
            visualizer = RealtimeVisualizer(
                window_name=window_title,
                display_width=display_width,
                display_height=display_height,
                fps_limit=getattr(self.config, 'target_fps', 30),
                save_output=save_output,
                output_path=output_path,
                max_persons=max_persons,
                processing_mode=processing_mode
            )
            
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
        stats_interval = 5.0  # 5초마다 통계 출력
        
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
            
            # 프레임 처리
            process_start = time.time()
            poses = None
            classification = None
            
            try:
                # 포즈 추정
                pose_result = self.pose_estimator.process_frame(frame, frame_count)
                if pose_result:
                    poses = self._convert_pose_result(pose_result, frame_count)
                    
                    # 트래킹
                    if poses and self.tracker:
                        poses = self.tracker.track_frame_poses(poses)
                    
                    # 심플한 윈도우/스트라이드 방식
                    if poses:
                        # 1. 프레임 버퍼에 추가 (포즈 추정 결과)
                        self._add_frame_to_buffer(poses)
                        
                        # 2. 윈도우 크기(100)만큼 쌓이면 분류 처리
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
                
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")
            
            processing_time = time.time() - process_start
            
            # 추가 정보 구성
            additional_info = {
                'fps': 1.0 / processing_time if processing_time > 0 else 0,  # FPS 키 이름 통일
                'processing_fps': 1.0 / processing_time if processing_time > 0 else 0,
                'total_persons': len(poses.persons) if poses and poses.persons else 0,
                'buffer_size': len(self.frame_buffer),
                'queue_size': input_manager.get_queue_size()
            }
            
            # 오버레이 데이터 구성 (분류 결과 포함)
            overlay_data = None
            if self.classification_results:
                overlay_data = {
                    'show_keypoints': True,
                    'show_tracking_ids': True,
                    'show_classification': True,
                    'window_results': [
                        {
                            'window_id': result['display_id'],
                            'classification': {
                                'predicted_class': result['predicted_class'],
                                'confidence': result['confidence'],
                                'probabilities': result['probabilities']
                            }
                        } for result in self.classification_results
                    ]
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
        
        logging.info(f"=== Realtime Stats ===")
        logging.info(f"Processing FPS: {processing_fps:.1f}")
        logging.info(f"Display FPS: {viz_stats['current_fps']:.1f}")
        logging.info(f"Capture FPS: {input_stats['capture_fps']:.1f}")
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
        """윈도우 분류 처리 - 트래킹과 복합점수 포함"""
        try:
            logging.info(f"Processing window {window_number} with {len(frames)} frames")
            
            # 1. 복합점수 계산 (전체 윈도우 프레임에 대해 한 번에)
            if self.scorer and any(len(frame.persons) > 0 for frame in frames):
                person_scores = self.scorer.calculate_scores(frames)
                logging.info(f"Calculated composite scores for {len(person_scores)} persons")
                
                # 복합점수를 각 프레임의 person에 적용
                scored_frames = []
                for frame in frames:
                    scored_persons = []
                    for person in frame.persons:
                        if person.track_id and person.track_id in person_scores:
                            # 복합점수 정보 추가
                            person.metadata = getattr(person, 'metadata', {})
                            person.metadata['composite_score'] = person_scores[person.track_id].composite_score
                            person.metadata['movement_score'] = person_scores[person.track_id].movement_score
                            person.metadata['interaction_score'] = person_scores[person.track_id].interaction_score
                        scored_persons.append(person)
                    
                    # 프레임 복사본 생성
                    from utils.data_structure import FramePoses
                    scored_frame = FramePoses(
                        frame_idx=frame.frame_idx,
                        persons=scored_persons,
                        timestamp=frame.timestamp,
                        image_shape=frame.image_shape,
                        metadata=frame.metadata
                    )
                    scored_frames.append(scored_frame)
            else:
                scored_frames = frames
                logging.info("No persons to score, using original frames")
            
            # 2. 윈도우 어노테이션 생성 (MMAction2 표준 형식)
            try:
                # 직접 정의된 create_window_annotation 함수 사용
                from utils.data_structure import convert_poses_to_stgcn_format, WindowAnnotation
                
                # ST-GCN 형식으로 변환 (직접 구현된 함수 사용)
                def convert_to_stgcn_format(frame_poses_list, max_persons=4):
                    import numpy as np
                    T = len(frame_poses_list)
                    M = max_persons  # 더 많은 person을 고려
                    V = 17  # COCO 키포인트 수
                    C = 2   # x, y (confidence는 별도로 저장)
                    
                    keypoint = np.zeros((M, T, V, C), dtype=np.float32)  # MMAction2 표준: [M, T, V, C]
                    keypoint_score = np.zeros((M, T, V), dtype=np.float32)
                    
                    # 실제 데이터 통계 수집
                    person_count_per_frame = []
                    valid_keypoint_count = 0
                    total_keypoint_count = 0
                    
                    for t, frame_poses in enumerate(frame_poses_list):
                        person_count_per_frame.append(len(frame_poses.persons))
                        
                        for m, person in enumerate(frame_poses.persons[:max_persons]):
                            total_keypoint_count += 1
                            
                            if isinstance(person.keypoints, np.ndarray):
                                kpts = person.keypoints
                                if kpts.shape == (V, 3):  # [17, 3] 형태 (x, y, confidence)
                                    keypoint[m, t, :, :] = kpts[:, :2]  # x, y만 사용
                                    keypoint_score[m, t, :] = kpts[:, 2]  # confidence
                                    valid_keypoint_count += 1
                                elif kpts.shape == (V, 2):  # [17, 2] 형태 (x, y만)
                                    keypoint[m, t, :, :] = kpts
                                    keypoint_score[m, t, :] = 1.0  # 기본 confidence
                                    valid_keypoint_count += 1
                                elif len(kpts.flatten()) >= V * 2:
                                    # 1D 배열인 경우
                                    reshaped = kpts.flatten()[:V*3].reshape(-1, 3) if len(kpts.flatten()) >= V*3 else kpts.flatten()[:V*2].reshape(-1, 2)
                                    if reshaped.shape[1] >= 2:
                                        keypoint[m, t, :reshaped.shape[0], :] = reshaped[:, :2]
                                        if reshaped.shape[1] >= 3:
                                            keypoint_score[m, t, :reshaped.shape[0]] = reshaped[:, 2]
                                        else:
                                            keypoint_score[m, t, :reshaped.shape[0]] = 1.0
                                        valid_keypoint_count += 1
                    
                    # 통계 로그 출력
                    avg_person_count = np.mean(person_count_per_frame) if person_count_per_frame else 0
                    max_person_count = max(person_count_per_frame) if person_count_per_frame else 0
                    data_fill_ratio = valid_keypoint_count / total_keypoint_count if total_keypoint_count > 0 else 0
                    
                    logging.info(f"DEBUG: Window conversion stats - Avg persons: {avg_person_count:.1f}, Max persons: {max_person_count}")
                    logging.info(f"DEBUG: Valid keypoints: {valid_keypoint_count}/{total_keypoint_count} ({data_fill_ratio:.3f})")
                    logging.info(f"DEBUG: Final keypoint shape: {keypoint.shape}, range: [{keypoint.min():.2f}, {keypoint.max():.2f}]")
                    
                    return keypoint, keypoint_score
                
                keypoint, keypoint_score = convert_to_stgcn_format(scored_frames)
                
                # 이미지 크기 추정 (첫 번째 유효한 프레임에서)
                img_shape = (640, 640)  # 기본값
                for frame_poses in scored_frames:
                    if frame_poses.image_shape is not None:
                        img_shape = frame_poses.image_shape
                        break
                
                window_annotation = WindowAnnotation(
                    window_idx=window_number - 1,
                    start_frame=scored_frames[0].frame_idx,
                    end_frame=scored_frames[-1].frame_idx,
                    keypoint=keypoint,
                    keypoint_score=keypoint_score,
                    frame_dir=f"window_{window_number}",
                    img_shape=img_shape,
                    original_shape=img_shape,
                    total_frames=len(scored_frames),
                    label=0
                )
            except Exception as e:
                logging.error(f"Error creating window annotation: {e}")
                # 간단한 윈도우 객체로 fallback
                class SimpleWindow:
                    def __init__(self, frames, window_num):
                        self.frames = frames
                        self.start_frame = frames[0].frame_idx
                        self.end_frame = frames[-1].frame_idx
                        self.window_idx = window_num - 1
                window_annotation = SimpleWindow(scored_frames, window_number)
            
            # 3. STGCN 분류
            if self.classifier:
                classification_result = self.classifier.classify_window(window_annotation)
                if classification_result:
                    logging.info(f"Window {window_number} classified as {classification_result.get_predicted_label()} "
                               f"(confidence: {classification_result.confidence:.3f})")
                    
                    classification = {
                        'display_id': window_number,
                        'predicted_class': classification_result.get_predicted_label(),
                        'confidence': classification_result.confidence,
                        'probabilities': classification_result.probabilities,
                        'window_start': scored_frames[0].frame_idx,
                        'window_end': scored_frames[-1].frame_idx
                    }
                    
                    # 모드별 처리 분리
                    if self.mode == "analysis":
                        # 분석 모드: 결과만 저장, 시각화 없음
                        self.classification_results.append(classification)
                        logging.info(f"[ANALYSIS] Window {window_number} result stored")
                    elif self.mode == "realtime":
                        # 실시간 모드: 시각화 포함
                        if visualizer:
                            visualizer.update_classification_history(classification)
                        self.classification_results.append(classification)
                        logging.info(f"[REALTIME] Window {window_number} result processed")
                    
                    logging.info(f"Window {window_number} classified: {classification_result.get_predicted_label()} ({classification_result.confidence:.3f})")
            else:
                logging.warning(f"No classifier available for window {window_number}")
                
        except Exception as e:
            import traceback
            logging.error(f"Error processing window {window_number}: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
    
    def _trigger_alert(self, classification_result, frame_number):
        """알림 트리거"""
        alert = {
            'timestamp': time.time(),
            'frame_idx': frame_number,
            'alert_type': classification_result.get_predicted_label(),
            'confidence': classification_result.confidence,
            'details': {
                'classification': classification_result.to_dict()
            }
        }
        
        # 알림 콜백 호출
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
        
        self.performance_stats['total_alerts'] += 1
        logging.warning(f"VIOLENCE ALERT: {classification_result.get_predicted_label()} (confidence: {classification_result.confidence:.3f}) at frame {frame_number}")
    
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
            for i, window in enumerate(windows):
                try:
                    class_start = time.time()
                    result = self.classifier.classify_window(window)
                    class_time = time.time() - class_start
                    
                    # 성능 추적 (분류 시간은 별도로 기록하지 않고 전체 통계에 포함)
                    
                    if result:
                        classification_results.append(result)
                        self.window_processor.add_window_result(i, result.to_dict())
                        logging.info(f"Window {i} classified: {result.get_predicted_label()} ({result.confidence:.3f})")
                    
                except Exception as e:
                    logging.error(f"Error classifying window {i}: {e}")
            
            # 성능 통계 완료
            performance_stats = self.performance_tracker.get_stats()
            
            # 추가 성능 통계 계산
            performance_stats.update({
                'frames_processed': total_frames,
                'windows_classified': len(classification_results),
                'fps': performance_stats['total_processed'] / performance_stats['total_time'] if performance_stats['total_time'] > 0 else 0,
                'pipeline_fps': performance_stats['total_processed'] / performance_stats['total_time'] if performance_stats['total_time'] > 0 else 0,
                'target_fps': 30.0,
                'total_alerts': 0,
                'classification_count': len(classification_results)
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