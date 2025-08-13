"""
RTMO 포즈 추정기 등록

팩토리 패턴에 RTMO 모듈을 등록합니다.
"""

from ...utils.factory import ModuleFactory
from .rtmo_estimator import RTMOPoseEstimator

# RTMO 기본 설정
DEFAULT_RTMO_CONFIG = {
    'score_threshold': 0.3,
    'nms_threshold': 0.65,
    'max_detections': 100,
    'device': 'cuda:0'
}

# RTMO 포즈 추정기를 팩토리에 등록
ModuleFactory.register_pose_estimator(
    name='rtmo',
    estimator_class=RTMOPoseEstimator,
    default_config=DEFAULT_RTMO_CONFIG
)

print("RTMO pose estimator registered successfully")