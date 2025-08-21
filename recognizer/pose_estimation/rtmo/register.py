"""
RTMO 포즈 추정기 등록

팩토리 패턴에 RTMO 모듈을 등록합니다.
"""

from utils.factory import ModuleFactory
from pose_estimation.rtmo.rtmo_estimator import RTMOPoseEstimator

# ONNX와 TensorRT 추정기도 임포트 (가능한 경우에만)
try:
    from pose_estimation.rtmo.rtmo_onnx_estimator import RTMOONNXEstimator
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from pose_estimation.rtmo.rtmo_tensorrt_estimator import RTMOTensorRTEstimator
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# RTMO 기본 설정
DEFAULT_RTMO_CONFIG = {
    'score_threshold': 0.3,
    'nms_threshold': 0.65,
    'max_detections': 100,
    'device': 'cuda:0'
}

# RTMO ONNX 기본 설정
DEFAULT_RTMO_ONNX_CONFIG = {
    'score_threshold': 0.3,
    'nms_threshold': 0.45,
    'max_detections': 100,
    'device': 'cuda:0',
    'model_input_size': (640, 640),
    'mean': None,
    'std': None,
    'backend': 'onnxruntime',
    'to_openpose': False
}

# RTMO TensorRT 기본 설정
DEFAULT_RTMO_TENSORRT_CONFIG = {
    'score_threshold': 0.3,
    'nms_threshold': 0.45,
    'max_detections': 100,
    'device': 'cuda:0',
    'model_input_size': (640, 640),
    'mean': None,
    'std': None,
    'to_openpose': False,
    'fp16_mode': False
}

# RTMO 포즈 추정기들을 팩토리에 등록
ModuleFactory.register_pose_estimator(
    name='rtmo',
    estimator_class=RTMOPoseEstimator,
    default_config=DEFAULT_RTMO_CONFIG
)

if ONNX_AVAILABLE:
    ModuleFactory.register_pose_estimator(
        name='rtmo_onnx',
        estimator_class=RTMOONNXEstimator,
        default_config=DEFAULT_RTMO_ONNX_CONFIG
    )
    print("RTMO ONNX pose estimator registered successfully")

if TENSORRT_AVAILABLE:
    ModuleFactory.register_pose_estimator(
        name='rtmo_tensorrt',
        estimator_class=RTMOTensorRTEstimator,
        default_config=DEFAULT_RTMO_TENSORRT_CONFIG
    )
    print("RTMO TensorRT pose estimator registered successfully")

print("RTMO pose estimator registered successfully")