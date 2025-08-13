"""
통합 파이프라인 구현

4단계 파이프라인을 통합 관리하는 메인 클래스입니다.
포즈 추정 → 트래킹 → 복합점수 계산 → 행동 분류의 전체 과정을 관리합니다.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..utils.factory import ModuleFactory
from ..utils.data_structure import (
    PersonPose, FramePoses, WindowAnnotation, ClassificationResult,
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정"""
    # 모듈별 설정
    pose_config: PoseEstimationConfig
    tracking_config: TrackingConfig
    scoring_config: ScoringConfig
    classification_config: ActionClassificationConfig
    
    # 파이프라인 설정
    window_size: int = 100
    window_stride: int = 50
    batch_size: int = 1
    
    # 성능 설정
    enable_gpu: bool = True
    device: str = 'cuda:0'
    
    # 출력 설정
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None
    
    def validate(self) -> bool:
        """설정 유효성 검사"""
        if self.window_size <= 0 or self.window_stride <= 0:
            return False
        if self.window_stride > self.window_size:
            logging.warning("Window stride is larger than window size")
        return True


@dataclass
class PipelineResult:
    """파이프라인 처리 결과"""
    video_path: str
    total_frames: int
    processed_windows: int
    classification_results: List[ClassificationResult]
    processing_time: float
    
    # 성능 통계
    avg_fps: float
    pose_extraction_time: float
    tracking_time: float
    scoring_time: float
    classification_time: float
    
    # 중간 결과 (선택적)
    intermediate_poses: Optional[List[FramePoses]] = None
    scoring_results: Optional[Dict[int, Any]] = None


class UnifiedPipeline:
    """통합 4단계 파이프라인 매니저"""
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        if not config.validate():
            raise ValueError("Invalid pipeline configuration")
        
        # 팩토리 인스턴스
        self.factory = ModuleFactory()
        
        # 모듈 인스턴스들
        self.pose_estimator = None
        self.tracker = None
        self.scorer = None
        self.classifier = None
        
        # 성능 모니터링
        self.performance_stats = {
            'total_videos_processed': 0,
            'total_frames_processed': 0,
            'total_windows_processed': 0,
            'avg_processing_time': 0.0
        }
        
        # 콜백 함수들
        self.progress_callbacks: List[Callable] = []
        self.result_callbacks: List[Callable] = []
        
        # 초기화
        self.initialize_pipeline()
    
    def initialize_pipeline(self) -> bool:
        """파이프라인 구성 요소 초기화"""
        try:
            logging.info("Initializing unified pipeline components...")
            
            # 1. 포즈 추정 모듈 초기화
            self.pose_estimator = self.factory.create_pose_estimator(
                self.config.pose_config.model_name,
                self.config.pose_config.__dict__
            )
            
            if not self.pose_estimator.initialize_model():
                raise RuntimeError("Failed to initialize pose estimator")
            
            # 2. 트래킹 모듈 초기화  
            self.tracker = self.factory.create_tracker(
                self.config.tracking_config.tracker_name,
                self.config.tracking_config.__dict__
            )
            
            if not self.tracker.initialize_tracker():
                raise RuntimeError("Failed to initialize tracker")
            
            # 3. 점수 계산 모듈 초기화
            self.scorer = self.factory.create_scorer(
                self.config.scoring_config.scorer_name,
                self.config.scoring_config.__dict__
            )
            
            if not self.scorer.initialize_scorer():
                raise RuntimeError("Failed to initialize scorer")
            
            # 4. 행동 분류 모듈 초기화
            self.classifier = self.factory.create_classifier(
                self.config.classification_config.model_name,
                self.config.classification_config.__dict__
            )
            
            if not self.classifier.initialize_model():
                raise RuntimeError("Failed to initialize classifier")
            
            logging.info("Unified pipeline initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {str(e)}")
            return False
    
    def process_video(self, video_path: Union[str, Path]) -> PipelineResult:
        """비디오 전체 처리
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            처리 결과
        """
        video_path = str(video_path)
        start_time = time.time()
        
        try:
            # 단계별 시간 측정
            pose_time = 0.0
            tracking_time = 0.0
            scoring_time = 0.0
            classification_time = 0.0
            
            # 1단계: 포즈 추정
            logging.info(f"Step 1: Extracting poses from {video_path}")
            pose_start = time.time()
            frame_poses = self.pose_estimator.extract_video_poses(video_path)
            pose_time = time.time() - pose_start
            
            if not frame_poses:
                raise RuntimeError("No poses extracted from video")
            
            total_frames = len(frame_poses)
            logging.info(f"Extracted poses from {total_frames} frames in {pose_time:.2f}s")
            
            # 2단계: 트래킹
            logging.info("Step 2: Tracking objects")
            tracking_start = time.time()
            tracked_poses = self.tracker.track_video_poses(frame_poses)
            tracking_time = time.time() - tracking_start
            logging.info(f"Tracking completed in {tracking_time:.2f}s")
            
            # 3단계: 복합점수 계산
            logging.info("Step 3: Calculating composite scores")
            scoring_start = time.time()
            scores = self.scorer.calculate_scores(tracked_poses)
            scoring_time = time.time() - scoring_start
            logging.info(f"Scoring completed in {scoring_time:.2f}s")
            
            # 4단계: 윈도우 생성 및 행동 분류
            logging.info("Step 4: Creating windows and classifying actions")
            classification_start = time.time()
            windows = self.create_sliding_windows(tracked_poses)
            classification_results = self.classifier.classify_multiple_windows(windows)
            classification_time = time.time() - classification_start
            
            processed_windows = len(windows)
            logging.info(f"Classified {processed_windows} windows in {classification_time:.2f}s")
            
            # 결과 생성
            total_time = time.time() - start_time
            avg_fps = total_frames / total_time if total_time > 0 else 0.0
            
            result = PipelineResult(
                video_path=video_path,
                total_frames=total_frames,
                processed_windows=processed_windows,
                classification_results=classification_results,
                processing_time=total_time,
                avg_fps=avg_fps,
                pose_extraction_time=pose_time,
                tracking_time=tracking_time,
                scoring_time=scoring_time,
                classification_time=classification_time,
                intermediate_poses=tracked_poses if self.config.save_intermediate_results else None,
                scoring_results=scores if self.config.save_intermediate_results else None
            )
            
            # 통계 업데이트
            self.update_statistics(result)
            
            # 콜백 호출
            for callback in self.result_callbacks:
                callback(result)
            
            logging.info(f"Pipeline processing completed in {total_time:.2f}s (avg {avg_fps:.1f} FPS)")
            return result
            
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {str(e)}")
            raise
    
    def process_video_batch(self, video_paths: List[Union[str, Path]]) -> List[PipelineResult]:
        """다중 비디오 배치 처리
        
        Args:
            video_paths: 비디오 파일 경로 리스트
            
        Returns:
            처리 결과 리스트
        """
        results = []
        
        for i, video_path in enumerate(video_paths):
            try:
                logging.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                
                # 진행률 콜백 호출
                progress = (i / len(video_paths)) * 100
                for callback in self.progress_callbacks:
                    callback(progress, f"Processing {Path(video_path).name}")
                
                result = self.process_video(video_path)
                results.append(result)
                
            except Exception as e:
                logging.error(f"Failed to process {video_path}: {str(e)}")
                # 에러가 있어도 계속 진행
                continue
        
        # 완료 콜백
        for callback in self.progress_callbacks:
            callback(100.0, "Batch processing completed")
        
        return results
    
    def create_sliding_windows(self, tracked_poses: List[FramePoses]) -> List[WindowAnnotation]:
        """슬라이딩 윈도우 생성
        
        Args:
            tracked_poses: 트래킹된 포즈 데이터
            
        Returns:
            윈도우 어노테이션 리스트
        """
        windows = []
        
        if len(tracked_poses) < self.config.window_size:
            # 프레임이 부족하면 패딩
            logging.warning(f"Insufficient frames ({len(tracked_poses)}) for window size ({self.config.window_size})")
            return windows
        
        window_id = 0
        
        for start_idx in range(0, len(tracked_poses) - self.config.window_size + 1, self.config.window_stride):
            end_idx = start_idx + self.config.window_size
            
            window_poses = tracked_poses[start_idx:end_idx]
            
            # 윈도우 어노테이션 생성
            window = WindowAnnotation(
                window_id=window_id,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                poses=window_poses,
                total_persons=self._count_unique_persons(window_poses)
            )
            
            windows.append(window)
            window_id += 1
        
        return windows
    
    def _count_unique_persons(self, poses: List[FramePoses]) -> int:
        """윈도우 내 고유한 person 수 계산"""
        unique_ids = set()
        
        for frame_poses in poses:
            for person in frame_poses.persons:
                if person.track_id is not None:
                    unique_ids.add(person.track_id)
        
        return len(unique_ids)
    
    def update_statistics(self, result: PipelineResult):
        """성능 통계 업데이트"""
        self.performance_stats['total_videos_processed'] += 1
        self.performance_stats['total_frames_processed'] += result.total_frames
        self.performance_stats['total_windows_processed'] += result.processed_windows
        
        # 평균 처리 시간 계산
        total_videos = self.performance_stats['total_videos_processed']
        current_avg = self.performance_stats['avg_processing_time']
        new_avg = ((current_avg * (total_videos - 1)) + result.processing_time) / total_videos
        self.performance_stats['avg_processing_time'] = new_avg
    
    def add_progress_callback(self, callback: Callable[[float, str], None]):
        """진행률 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable[[PipelineResult], None]):
        """결과 콜백 추가"""
        self.result_callbacks.append(callback)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 정보 반환"""
        return {
            'config': {
                'window_size': self.config.window_size,
                'window_stride': self.config.window_stride,
                'batch_size': self.config.batch_size,
                'device': self.config.device
            },
            'modules': {
                'pose_estimator': self.pose_estimator.get_estimator_info() if self.pose_estimator else None,
                'tracker': self.tracker.get_tracker_info() if self.tracker else None,
                'scorer': self.scorer.get_scorer_info() if self.scorer else None,
                'classifier': self.classifier.get_classifier_info() if self.classifier else None
            },
            'performance_stats': self.performance_stats.copy()
        }
    
    def save_results(self, result: PipelineResult, output_path: Union[str, Path]):
        """결과를 파일로 저장"""
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과를 직렬화 가능한 형태로 변환
        result_data = {
            'video_path': result.video_path,
            'total_frames': result.total_frames,
            'processed_windows': result.processed_windows,
            'processing_time': result.processing_time,
            'avg_fps': result.avg_fps,
            'timing': {
                'pose_extraction_time': result.pose_extraction_time,
                'tracking_time': result.tracking_time,
                'scoring_time': result.scoring_time,
                'classification_time': result.classification_time
            },
            'classification_results': [
                {
                    'window_id': r.window_id,
                    'predicted_class': r.predicted_class,
                    'confidence': r.confidence,
                    'class_probabilities': r.class_probabilities
                }
                for r in result.classification_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to {output_path}")
    
    def cleanup(self):
        """리소스 정리"""
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