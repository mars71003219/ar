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

from .config import RealtimeConfig, RealtimeAlert
from ..base import BasePipeline, ModuleInitializer, PerformanceTracker
from ...utils.data_structure import PersonPose, FramePoses, WindowAnnotation, ClassificationResult


class InferencePipeline(BasePipeline):
    """실시간 추론 파이프라인"""
    
    def __init__(self, config: RealtimeConfig):
        super().__init__(config)
        
        # 모듈 인스턴스들
        self.pose_estimator = None
        self.tracker = None
        self.scorer = None
        self.classifier = None
        self.window_processor = None
        
        # 실시간 처리용 큐들
        self.frame_queue = Queue(maxsize=config.max_queue_size)
        self.pose_queue = Queue(maxsize=config.max_queue_size)
        self.result_queue = Queue(maxsize=100)
        
        # 윈도우 버퍼
        self.window_buffer: List[FramePoses] = []
        self.last_inference_frame = 0
        
        # 스레드 관리
        self.processing_thread = None
        self.is_running = False
        
        # 성능 추적
        self.performance_tracker = PerformanceTracker()
        
        # 콜백 함수들
        self.alert_callbacks: List[Callable[[RealtimeAlert], None]] = []
        self.frame_callbacks: List[Callable[[np.ndarray, List[PersonPose]], None]] = []
    
    def initialize_pipeline(self) -> bool:
        """파이프라인 모듈 초기화"""
        try:
            logging.info("Initializing realtime inference pipeline")
            
            # ModuleInitializer 사용
            self.pose_estimator = ModuleInitializer.init_pose_estimator(
                self.factory, self.config.pose_config.__dict__
            )
            self.tracker = ModuleInitializer.init_tracker(
                self.factory, self.config.tracking_config.__dict__
            )
            self.scorer = ModuleInitializer.init_scorer(
                self.factory, self.config.scoring_config.__dict__
            )
            self.classifier = ModuleInitializer.init_classifier(
                self.factory, self.config.classification_config.__dict__
            )
            self.window_processor = ModuleInitializer.init_window_processor(
                self.factory, self.config.window_size, self.config.inference_stride
            )
            
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
                if frame_idx % self.config.skip_frames != 0:
                    frame_idx += 1
                    continue
                
                # 입력 리사이즈
                if self.config.resize_input:
                    frame = cv2.resize(frame, self.config.resize_input)
                
                # 프레임 처리
                self._process_single_frame(frame, frame_idx)
                frame_idx += 1
                
                # FPS 제어
                time.sleep(1.0 / self.config.target_fps)
                
        except Exception as e:
            logging.error(f"Processing loop error: {e}")
        finally:
            cap.release()
            self.is_running = False
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int):
        """단일 프레임 처리"""
        start_time = time.time()
        
        try:
            # 포즈 추정
            frame_poses = self.pose_estimator.process_frame(frame, frame_idx)
            
            # 트래킹
            if self.tracker:
                frame_poses = self.tracker.track_frame_poses(frame_poses)
            
            # 스코어링
            if self.scorer:
                frame_poses = self.scorer.score_frame_poses(frame_poses)
            
            # 윈도우 버퍼에 추가
            self.window_buffer.append(frame_poses)
            
            # 윈도우 크기 유지
            if len(self.window_buffer) > self.config.window_size:
                self.window_buffer.pop(0)
            
            # 추론 간격 체크
            if (frame_idx - self.last_inference_frame) >= self.config.inference_stride:
                self._run_inference(frame_idx)
                self.last_inference_frame = frame_idx
            
            # 콜백 호출
            for callback in self.frame_callbacks:
                callback(frame, frame_poses.poses)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self.performance_tracker.update(processing_time)
            
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
    
    def _run_inference(self, frame_idx: int):
        """윈도우 기반 추론 실행"""
        if len(self.window_buffer) < self.config.window_size:
            return
        
        try:
            # 윈도우 생성
            windows = self.window_processor.process_frames(self.window_buffer)
            
            for window in windows:
                # 분류 수행
                result = self.classifier.classify_window(window)
                
                # 알림 체크
                if result.confidence >= self.config.alert_threshold:
                    alert = RealtimeAlert(
                        timestamp=time.time(),
                        frame_idx=frame_idx,
                        alert_type=result.predicted_class_name,
                        confidence=result.confidence,
                        details={
                            'window_info': {
                                'start_frame': window.start_frame,
                                'end_frame': window.end_frame,
                                'person_count': self._count_unique_persons([self.window_buffer[-1]])
                            },
                            'classification': result.to_dict()
                        }
                    )
                    
                    # 알림 콜백 호출
                    for callback in self.alert_callbacks:
                        callback(alert)
                
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
            for pose in frame_poses.poses:
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
        stats.update({
            'frames_processed': stats['total_processed'],
            'fps': 1.0 / stats['avg_time'] if stats['avg_time'] > 0 else 0.0
        })
        return stats
    
    def add_alert_callback(self, callback: Callable[[RealtimeAlert], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, List[PersonPose]], None]):
        """프레임 콜백 추가"""
        self.frame_callbacks.append(callback)