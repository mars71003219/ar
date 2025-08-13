"""
Recognizer - 모듈화된 폭력 탐지 시스템

4단계 파이프라인을 통한 효율적이고 확장 가능한 폭력 탐지:
1. 포즈 추정 (Pose Estimation)
2. 객체 추적 (Object Tracking) 
3. 복합 점수 계산 (Composite Scoring)
4. 행동 분류 (Action Classification)

팩토리 패턴을 사용하여 각 단계의 모델을 쉽게 교체할 수 있습니다.
"""

from .utils.factory import ModuleFactory
from .utils.data_structure import (
    PersonPose, FramePoses, WindowAnnotation, ClassificationResult,
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)

# 팩토리 인스턴스 생성 및 기본 모듈 등록
factory = ModuleFactory()

# 기본 모듈들 자동 등록
def initialize_factory():
    """팩토리 초기화 및 기본 모듈 등록"""
    try:
        # 포즈 추정 모듈
        from .pose_estimation.rtmo import RTMOPoseEstimator
        factory.register_pose_estimator('rtmo', RTMOPoseEstimator)
        
        # 트래킹 모듈 (mmtracking 사용)
        from .tracking.mmtracking_adapter import MMTrackingAdapter
        factory.register_tracker('bytetrack', MMTrackingAdapter)
        factory.register_tracker('deepsort', MMTrackingAdapter)
        factory.register_tracker('sort', MMTrackingAdapter)
        
        # 점수 계산 모듈
        from .scoring.region_based import RegionBasedScorer
        factory.register_scorer('region_based', RegionBasedScorer)
        
        # 행동 분류 모듈
        from .action_classification.stgcn import STGCNActionClassifier
        factory.register_classifier('stgcn', STGCNActionClassifier)
        
        print("Recognizer factory initialized with all default modules")
        
    except Exception as e:
        print(f"Warning: Some modules could not be registered: {str(e)}")

# 초기화 실행
initialize_factory()

__version__ = "1.0.0"
__all__ = [
    "ModuleFactory", "factory", "initialize_factory",
    "PersonPose", "FramePoses", "WindowAnnotation", "ClassificationResult",
    "PoseEstimationConfig", "TrackingConfig", "ScoringConfig", "ActionClassificationConfig"
]