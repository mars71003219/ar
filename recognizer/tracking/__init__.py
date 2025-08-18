"""
트래킹 모듈

pose_estimation과 scoring 사이에서 정확한 입/출력을 통한 트래킹 처리를 제공합니다.
표준화된 인터페이스를 통해 다양한 트래커를 폴더 단위로 교체할 수 있습니다.

지원 트래커:
- ByteTracker: mmtracking 기반 ByteTrack 구현
- DeepSORT: (향후 추가)
- SORT: (향후 추가)
"""

try:
    from tracking.base import BaseTracker
    from tracking.bytetrack import ByteTrackerWrapper, ByteTrackerConfig
except ImportError:
    from .base import BaseTracker
    from .bytetrack import ByteTrackerWrapper, ByteTrackerConfig

__all__ = [
    'BaseTracker',
    'ByteTrackerWrapper',
    'ByteTrackerConfig'
]