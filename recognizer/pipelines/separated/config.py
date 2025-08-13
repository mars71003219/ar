"""
분리형 파이프라인 설정
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ...utils.data_structure import (
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)


@dataclass
class SeparatedPipelineConfig:
    """통합 분리형 파이프라인 설정"""
    # 모듈 설정
    pose_config: PoseEstimationConfig
    tracking_config: TrackingConfig
    scoring_config: ScoringConfig
    classification_config: ActionClassificationConfig
    
    # 윈도우 설정
    window_size: int = 100
    window_stride: int = 50
    
    # 출력 디렉토리 설정 (시각화 지원)
    stage1_output_dir: str = "output/separated/stage1_poses"
    stage2_output_dir: str = "output/separated/stage2_tracking"
    stage3_output_dir: str = "output/separated/stage3_scoring"
    stage4_output_dir: str = "output/separated/stage4_unified"
    
    # 단계별 실행 제어
    stages_to_run: List[str] = field(default_factory=lambda: ["stage1", "stage2", "stage3", "stage4"])
    
    # 멀티프로세싱 설정
    enable_multiprocessing: bool = False
    num_workers: int = 4
    
    # Resume 기능
    enable_resume: bool = True
    
    # 데이터셋 분할
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # 품질 필터링 (annotation 모드)
    enable_quality_filter: bool = True
    min_confidence: float = 0.5
    min_keypoint_score: float = 0.3
    max_track_gap: int = 10
    
    # 출력 제어
    save_intermediate_results: bool = True
    save_visualizations: bool = True
    
    def __post_init__(self):
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Dataset split ratios must sum to 1.0")