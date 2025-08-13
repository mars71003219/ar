"""
통합 파이프라인 메인 클래스
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path

from .config import PipelineConfig, PipelineResult
from ..base import BasePipeline, ModuleInitializer, PerformanceTracker
from ...utils.data_structure import FramePoses, WindowAnnotation, ClassificationResult


class UnifiedPipeline(BasePipeline):
    """통합 4단계 파이프라인 매니저"""
    
    def __init__(self, config: PipelineConfig):
        if not config.validate():
            raise ValueError("Invalid pipeline configuration")
        
        super().__init__(config)
        
        # 파이프라인 모듈들
        self.pose_estimator = None
        self.tracker = None
        self.scorer = None
        self.classifier = None
        self.window_processor = None
        
        # 성능 추적
        self.performance_tracker = PerformanceTracker()
        
        # 콜백 함수들
        self.progress_callbacks: List[Callable[[float, str], None]] = []
        self.result_callbacks: List[Callable[[PipelineResult], None]] = []
    
    def initialize_pipeline(self) -> bool:
        """파이프라인 모듈 초기화"""
        try:
            logging.info("Initializing unified pipeline modules")
            
            # ModuleInitializer 사용하여 중복 제거
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
                self.factory, self.config.window_size, self.config.window_stride
            )
            
            self._initialized = True
            logging.info("Pipeline initialization completed")
            return True
            
        except Exception as e:
            logging.error(f"Pipeline initialization failed: {e}")
            return False
    
    def process_video(self, video_path: Union[str, Path]) -> PipelineResult:
        """단일 비디오 처리"""
        if not self.pose_estimator:
            if not self.initialize_pipeline():
                raise RuntimeError("Failed to initialize pipeline")
        
        video_path = Path(video_path)
        start_time = time.time()
        
        logging.info(f"Processing video: {video_path}")
        
        # Stage 1: 포즈 추정
        pose_start = time.time()
        self._notify_progress(0.1, "Pose estimation in progress")
        
        frame_poses_list = self.pose_estimator.process_video(str(video_path))
        pose_time = time.time() - pose_start
        
        # Stage 2: 트래킹
        tracking_start = time.time()
        self._notify_progress(0.3, "Tracking in progress")
        
        tracked_poses = []
        self.tracker.reset()
        
        for frame_poses in frame_poses_list:
            tracked_frame = self.tracker.track_frame_poses(frame_poses)
            tracked_poses.append(tracked_frame)
        
        tracking_time = time.time() - tracking_start
        
        # Stage 3: 스코어링
        scoring_start = time.time()
        self._notify_progress(0.5, "Scoring in progress")
        
        scored_poses = []
        for tracked_frame in tracked_poses:
            scored_frame = self.scorer.score_frame_poses(tracked_frame)
            scored_poses.append(scored_frame)
        
        scoring_time = time.time() - scoring_start
        
        # Stage 4: 윈도우 생성 및 분류
        classification_start = time.time()
        self._notify_progress(0.7, "Classification in progress")
        
        windows = self.create_sliding_windows(scored_poses)
        
        classification_results = []
        for window in windows:
            result = self.classifier.classify_window(window)
            classification_results.append(result)
        
        classification_time = time.time() - classification_start
        
        # 결과 생성
        total_time = time.time() - start_time
        avg_fps = len(frame_poses_list) / total_time if total_time > 0 else 0
        
        result = PipelineResult(
            video_path=str(video_path),
            total_frames=len(frame_poses_list),
            processed_windows=len(windows),
            classification_results=classification_results,
            processing_time=total_time,
            avg_fps=avg_fps,
            pose_extraction_time=pose_time,
            tracking_time=tracking_time,
            scoring_time=scoring_time,
            classification_time=classification_time,
            intermediate_poses=scored_poses if self.config.save_intermediate_results else None
        )
        
        # 성능 추적
        self.performance_tracker.update(total_time)
        
        # 콜백 호출
        self._notify_progress(1.0, "Processing completed")
        for callback in self.result_callbacks:
            callback(result)
        
        logging.info(f"Video processing completed: {total_time:.2f}s, {avg_fps:.1f} fps")
        
        return result
    
    def process_video_batch(self, video_paths: List[Union[str, Path]]) -> List[PipelineResult]:
        """다중 비디오 배치 처리"""
        results = []
        
        for i, video_path in enumerate(video_paths):
            logging.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            try:
                result = self.process_video(video_path)
                results.append(result)
                
                # 배치 진행률 알림
                batch_progress = (i + 1) / len(video_paths)
                self._notify_progress(batch_progress, f"Batch progress: {i+1}/{len(video_paths)}")
                
            except Exception as e:
                logging.error(f"Failed to process {video_path}: {e}")
                continue
        
        return results
    
    def create_sliding_windows(self, tracked_poses: List[FramePoses]) -> List[WindowAnnotation]:
        """슬라이딩 윈도우 생성"""
        return self.window_processor.process_frames(tracked_poses)
    
    def _count_unique_persons(self, poses: List[FramePoses]) -> int:
        """고유 인물 수 계산"""
        person_ids = set()
        for frame_poses in poses:
            for pose in frame_poses.poses:
                if pose.person_id is not None:
                    person_ids.add(pose.person_id)
        return len(person_ids)
    
    
    def _notify_progress(self, progress: float, message: str):
        """진행률 알림"""
        for callback in self.progress_callbacks:
            callback(progress, message)
    
    def add_progress_callback(self, callback: Callable[[float, str], None]):
        """진행률 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable[[PipelineResult], None]):
        """결과 콜백 추가"""
        self.result_callbacks.append(callback)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 정보 반환"""
        stats = self.performance_tracker.get_stats()
        
        return {
            'performance_stats': stats,
            'config': {
                'window_size': self.config.window_size,
                'window_stride': self.config.window_stride,
                'batch_size': self.config.batch_size
            }
        }
    
    def save_results(self, result: PipelineResult, output_path: Union[str, Path]):
        """결과 저장"""
        import pickle
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        logging.info(f"Results saved to: {output_path}")
    
