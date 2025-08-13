"""
실시간 추론 파이프라인

실시간 비디오 스트림 처리를 위한 경량화된 파이프라인입니다.
RTSP 스트림, 웹캠 등의 실시간 입력을 지원합니다.
"""

import time
import threading
from queue import Queue, Empty
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
import cv2
import numpy as np
import multiprocessing as mp

from ..utils.factory import ModuleFactory
from ..utils.data_structure import (
    PersonPose, FramePoses, WindowAnnotation, ClassificationResult,
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)
from ..utils.multiprocess_manager import MultiprocessManager


@dataclass
class RealtimeConfig:
    """실시간 추론 설정"""
    # 모듈 설정
    pose_config: PoseEstimationConfig
    tracking_config: TrackingConfig
    scoring_config: ScoringConfig
    classification_config: ActionClassificationConfig
    
    # 실시간 처리 설정
    window_size: int = 100
    inference_stride: int = 25  # 추론 간격 (프레임 수)
    max_queue_size: int = 200
    target_fps: float = 30.0
    
    # 품질 관리
    min_confidence: float = 0.5
    alert_threshold: float = 0.7
    
    # 성능 최적화
    skip_frames: int = 1  # 1이면 모든 프레임, 2면 1프레임 건너뛰기
    resize_input: Optional[tuple] = None  # (width, height) 또는 None
    
    # 멀티프로세스 설정
    num_workers: int = field(default_factory=lambda: min(mp.cpu_count(), 4))  # 실시간은 적은 워커 사용
    enable_multiprocess: bool = False  # 실시간에서는 기본 비활성화 (지연 최소화)
    multiprocess_batch_size: int = 2
    multiprocess_timeout: float = 30.0  # 짧은 타임아웃


@dataclass
class RealtimeAlert:
    """실시간 알림 데이터"""
    timestamp: float
    frame_idx: int
    alert_type: str
    confidence: float
    details: Dict[str, Any]


class InferencePipeline:
    """실시간 추론 파이프라인"""
    
    def __init__(self, config: RealtimeConfig):
        """
        Args:
            config: 실시간 추론 설정
        """
        self.config = config
        self.factory = ModuleFactory()
        
        # 모듈 인스턴스들
        self.pose_estimator = None
        self.tracker = None
        self.scorer = None
        self.classifier = None
        
        # 실시간 처리용 큐들
        self.frame_queue = Queue(maxsize=config.max_queue_size)
        self.pose_queue = Queue(maxsize=config.max_queue_size)
        self.result_queue = Queue(maxsize=100)
        
        # 윈도우 버퍼 (슬라이딩 윈도우용)
        self.window_buffer: List[FramePoses] = []
        self.last_inference_frame = 0
        
        # 스레드 관리
        self.processing_thread = None
        self.is_running = False
        
        # 성능 모니터링
        self.performance_stats = {
            'frames_processed': 0,
            'poses_extracted': 0,
            'windows_classified': 0,
            'alerts_generated': 0,
            'avg_processing_time': 0.0,
            'current_fps': 0.0
        }
        
        # 콜백 함수들
        self.alert_callbacks: List[Callable[[RealtimeAlert], None]] = []
        self.frame_callbacks: List[Callable[[np.ndarray, List[PersonPose]], None]] = []
        
        self.initialize_pipeline()
    
    def initialize_pipeline(self) -> bool:
        """파이프라인 초기화"""
        try:
            logging.info("Initializing realtime inference pipeline...")
            
            # 모든 모듈 초기화
            self.pose_estimator = self.factory.create_pose_estimator(
                self.config.pose_config.model_name,
                self.config.pose_config.__dict__
            )
            if not self.pose_estimator.initialize_model():
                raise RuntimeError("Failed to initialize pose estimator")
            
            self.tracker = self.factory.create_tracker(
                self.config.tracking_config.tracker_name,
                self.config.tracking_config.__dict__
            )
            if not self.tracker.initialize_tracker():
                raise RuntimeError("Failed to initialize tracker")
            
            self.scorer = self.factory.create_scorer(
                self.config.scoring_config.scorer_name,
                self.config.scoring_config.__dict__
            )
            if not self.scorer.initialize_scorer():
                raise RuntimeError("Failed to initialize scorer")
            
            self.classifier = self.factory.create_classifier(
                self.config.classification_config.model_name,
                self.config.classification_config.__dict__
            )
            if not self.classifier.initialize_model():
                raise RuntimeError("Failed to initialize classifier")
            
            logging.info("Realtime inference pipeline initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize inference pipeline: {str(e)}")
            return False
    
    def start_realtime_processing(self, source: Union[str, int] = 0):
        """실시간 처리 시작
        
        Args:
            source: 비디오 소스 (카메라 인덱스, RTSP URL 등)
        """
        if self.is_running:
            logging.warning("Pipeline is already running")
            return
        
        self.is_running = True
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(source,),
            daemon=True
        )
        self.processing_thread.start()
        
        logging.info(f"Realtime processing started with source: {source}")
    
    def stop_realtime_processing(self):
        """실시간 처리 중지"""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # 큐 정리
        self._clear_queues()
        
        logging.info("Realtime processing stopped")
    
    def _processing_loop(self, source: Union[str, int]):
        """메인 처리 루프"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logging.error(f"Failed to open video source: {source}")
            return
        
        try:
            frame_count = 0
            last_time = time.time()
            
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 프레임 스키핑
                if frame_count % (self.config.skip_frames + 1) != 0:
                    continue
                
                # 입력 크기 조정
                if self.config.resize_input:
                    frame = cv2.resize(frame, self.config.resize_input)
                
                # 프레임 처리
                self._process_single_frame(frame, frame_count)
                
                # FPS 모니터링
                current_time = time.time()
                if current_time - last_time >= 1.0:  # 1초마다 업데이트
                    self.performance_stats['current_fps'] = frame_count / (current_time - last_time)
                    last_time = current_time
                    frame_count = 0
                
                # FPS 제어
                time.sleep(1.0 / self.config.target_fps)
                
        except Exception as e:
            logging.error(f"Error in processing loop: {str(e)}")
        finally:
            cap.release()
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int):
        """단일 프레임 처리"""
        try:
            start_time = time.time()
            
            # 1. 포즈 추정
            frame_poses = self.pose_estimator.extract_poses(frame, frame_idx)
            if not frame_poses or not frame_poses.persons:
                return
            
            # 신뢰도 필터링
            valid_persons = [
                person for person in frame_poses.persons
                if person.score >= self.config.min_confidence
            ]
            
            if not valid_persons:
                return
            
            frame_poses.persons = valid_persons
            
            # 2. 트래킹
            tracked_frame = self.tracker.track_frame_poses(frame_poses)
            
            # 3. 윈도우 버퍼 업데이트
            self.window_buffer.append(tracked_frame)
            
            # 윈도우 크기 유지
            if len(self.window_buffer) > self.config.window_size:
                self.window_buffer.pop(0)
            
            # 4. 추론 실행 (stride 간격으로)
            if (frame_idx - self.last_inference_frame) >= self.config.inference_stride:
                if len(self.window_buffer) == self.config.window_size:
                    self._run_inference(frame_idx)
                    self.last_inference_frame = frame_idx
            
            # 5. 콜백 호출
            for callback in self.frame_callbacks:
                callback(frame, valid_persons)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
        except Exception as e:
            logging.error(f"Error processing frame {frame_idx}: {str(e)}")
    
    def _run_inference(self, frame_idx: int):
        """추론 실행"""
        try:
            # 윈도우 생성
            window = WindowAnnotation(
                window_id=frame_idx,
                start_frame=frame_idx - self.config.window_size,
                end_frame=frame_idx - 1,
                poses=self.window_buffer.copy(),
                total_persons=self._count_unique_persons(self.window_buffer)
            )
            
            # 점수 계산
            scores = self.scorer.calculate_scores(self.window_buffer)
            
            # 행동 분류
            result = self.classifier.classify_single_window(window)
            
            # 알림 생성 및 콜백
            if result.confidence >= self.config.alert_threshold:
                alert = RealtimeAlert(
                    timestamp=time.time(),
                    frame_idx=frame_idx,
                    alert_type=result.predicted_class,
                    confidence=result.confidence,
                    details={
                        'class_probabilities': result.class_probabilities,
                        'window_id': result.window_id,
                        'persons_count': window.total_persons
                    }
                )
                
                # 알림 콜백 호출
                for callback in self.alert_callbacks:
                    callback(alert)
                
                self.performance_stats['alerts_generated'] += 1
            
            # 결과 큐에 추가
            if not self.result_queue.full():
                self.result_queue.put(result)
            
            self.performance_stats['windows_classified'] += 1
            
        except Exception as e:
            logging.error(f"Error in inference: {str(e)}")
    
    def _count_unique_persons(self, poses: List[FramePoses]) -> int:
        """고유한 person 수 계산"""
        unique_ids = set()
        for frame_poses in poses:
            for person in frame_poses.persons:
                if person.track_id is not None:
                    unique_ids.add(person.track_id)
        return len(unique_ids)
    
    def _update_performance_stats(self, processing_time: float):
        """성능 통계 업데이트"""
        self.performance_stats['frames_processed'] += 1
        
        # 평균 처리 시간 업데이트
        current_avg = self.performance_stats['avg_processing_time']
        frame_count = self.performance_stats['frames_processed']
        new_avg = ((current_avg * (frame_count - 1)) + processing_time) / frame_count
        self.performance_stats['avg_processing_time'] = new_avg
    
    def _clear_queues(self):
        """모든 큐 정리"""
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
    
    def add_alert_callback(self, callback: Callable[[RealtimeAlert], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, List[PersonPose]], None]):
        """프레임 처리 콜백 추가"""
        self.frame_callbacks.append(callback)
    
    def get_latest_results(self, max_count: int = 10) -> List[ClassificationResult]:
        """최신 결과 가져오기"""
        results = []
        count = 0
        
        while count < max_count and not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
                count += 1
            except Empty:
                break
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.performance_stats.copy()
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_realtime_processing()
        
        if self.pose_estimator:
            self.pose_estimator.cleanup()
        
        if self.tracker:
            self.tracker.cleanup()
        
        if self.scorer:
            self.scorer.cleanup()
        
        if self.classifier:
            self.classifier.cleanup()
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.cleanup()