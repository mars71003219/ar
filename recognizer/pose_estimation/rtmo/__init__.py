"""
RTMO 포즈 추정 모듈

Real-Time Multi-Object pose estimation을 위한 RTMO 모델 구현
"""

try:
    from pose_estimation.rtmo.rtmo_estimator import RTMOPoseEstimator
except ImportError:
    from .rtmo_estimator import RTMOPoseEstimator

__all__ = ['RTMOPoseEstimator']