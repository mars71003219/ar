"""
RTMO 포즈 추정 모듈

Real-Time Multi-Object pose estimation을 위한 RTMO 모델 구현
PyTorch (.pth), ONNX, TensorRT 추론 방식 지원
"""

try:
    from pose_estimation.rtmo.rtmo_estimator import RTMOPoseEstimator
except ImportError:
    from .rtmo_estimator import RTMOPoseEstimator

# ONNX 추정기 (선택적 임포트)
try:
    from pose_estimation.rtmo.rtmo_onnx_estimator import RTMOONNXEstimator
    ONNX_AVAILABLE = True
except ImportError:
    RTMOONNXEstimator = None
    ONNX_AVAILABLE = False

# TensorRT 추정기 (선택적 임포트)
try:
    from pose_estimation.rtmo.rtmo_tensorrt_estimator import RTMOTensorRTEstimator
    TENSORRT_AVAILABLE = True
except ImportError:
    RTMOTensorRTEstimator = None
    TENSORRT_AVAILABLE = False

__all__ = ['RTMOPoseEstimator']

if ONNX_AVAILABLE:
    __all__.append('RTMOONNXEstimator')

if TENSORRT_AVAILABLE:
    __all__.append('RTMOTensorRTEstimator')