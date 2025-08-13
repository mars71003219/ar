#!/usr/bin/env python3
"""
RTMO specific tracker configuration
RTMO 포즈 추정과 최적화된 설정값들
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.default_config import DefaultTrackerConfig


class RTMOTrackerConfig(DefaultTrackerConfig):
    """RTMO에 최적화된 Enhanced ByteTracker 설정"""
    
    # RTMO에 맞춤 조정된 detection score 임계값들
    obj_score_thrs = {
        'high': 0.5,  # RTMO의 일반적인 성능을 고려하여 조금 낮춤
        'low': 0.1    # 낮은 임계값 유지
    }
    
    # RTMO에 맞춘 초기화 임계값
    init_track_thr = 0.6  # 조금 더 관대하게
    
    # IoU 임계값들도 RTMO에 맞게 조정
    match_iou_thrs = {
        'high': 0.1,        # 엄격한 매칭 유지
        'low': 0.4,         # 조금 더 엄격하게 (0.5 → 0.4)
        'tentative': 0.3    # 유지
    }
    
    # 포즈 추정 특성상 더 빠른 확정
    num_tentatives = 2  # 3 → 2로 줄임
    
    # 사람 추적 특성상 더 오래 유지
    num_frames_retain = 50  # 30 → 50으로 증가
    
    # Kalman filter는 기본 설정 유지
    kalman_config = {
        'center_only': False,
        'dt': 1.0
    }
    
    # RTMO 성능 최적화 설정
    performance_config = {
        'enable_parallel': False,
        'cache_predictions': True,
        'optimize_memory': True,
        'rtmo_specific_optimizations': True  # RTMO 전용 최적화
    }
    
    # RTMO용 추가 설정들
    rtmo_specific = {
        'use_pose_features': False,      # 포즈 특징 활용 (추후 구현)
        'pose_similarity_weight': 0.0,  # 포즈 유사도 가중치
        'human_motion_model': True,      # 사람 동작 모델 사용
        'adaptive_thresholds': True      # 적응적 임계값 조정
    }
    
    @classmethod
    def get_config_dict(cls):
        """RTMO 설정을 딕셔너리로 반환"""
        base_config = super().get_config_dict()
        base_config.update({
            'rtmo_specific': cls.rtmo_specific
        })
        return base_config


# 사전 정의된 설정들
RTMO_FAST_CONFIG = {
    'obj_score_thrs': {'high': 0.4, 'low': 0.1},
    'init_track_thr': 0.5,
    'match_iou_thrs': {'high': 0.1, 'low': 0.3, 'tentative': 0.2},
    'num_tentatives': 1,
    'num_frames_retain': 30
}

RTMO_ACCURATE_CONFIG = {
    'obj_score_thrs': {'high': 0.7, 'low': 0.2},
    'init_track_thr': 0.8,
    'match_iou_thrs': {'high': 0.2, 'low': 0.5, 'tentative': 0.4},
    'num_tentatives': 3,
    'num_frames_retain': 100
}

RTMO_BALANCED_CONFIG = RTMOTrackerConfig.get_config_dict()