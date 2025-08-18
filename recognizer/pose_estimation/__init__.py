"""
포즈 추정 모듈

다양한 포즈 추정 모델을 지원하는 통합 인터페이스를 제공합니다.

지원 모델:
- RTMO: Real-Time Multi-Object pose estimation
- YOLOv8: (향후 추가 예정)
"""

try:
    from pose_estimation.base import BasePoseEstimator
    from pose_estimation.rtmo import RTMOPoseEstimator
except ImportError:
    from .base import BasePoseEstimator
    from .rtmo import RTMOPoseEstimator

__all__ = [
    'BasePoseEstimator',
    'RTMOPoseEstimator'
]