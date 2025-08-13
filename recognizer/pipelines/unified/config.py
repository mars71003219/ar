"""
통합 파이프라인 설정
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from ...utils.data_structure import (
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig,
    ClassificationResult
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
    intermediate_poses: Optional[List] = None
    scoring_results: Optional[Dict[int, Any]] = None