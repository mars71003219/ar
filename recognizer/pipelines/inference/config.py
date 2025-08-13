"""
실시간 추론 파이프라인 설정
"""

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import sys
from pathlib import Path

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

from utils.data_structure import (
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)


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
    num_workers: int = field(default_factory=lambda: min(mp.cpu_count(), 4))
    enable_multiprocess: bool = False  # 실시간에서는 기본 비활성화 (지연 최소화)
    multiprocess_batch_size: int = 2
    multiprocess_timeout: float = 30.0


@dataclass
class RealtimeAlert:
    """실시간 알림 데이터"""
    timestamp: float
    frame_idx: int
    alert_type: str
    confidence: float
    details: Dict[str, Any]