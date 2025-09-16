"""
복합점수 계산 모듈

다양한 점수 계산 방법을 지원하는 통합 인터페이스를 제공합니다.

지원 점수 계산기:
- MotionBasedScorer: 움직임 기반 복합점수 계산 (이전 RegionBasedScorer)
- TemporalScorer: (향후 추가 예정)
"""

try:
    from scoring.base import BaseScorer
    from scoring.motion_based import MotionBasedScorer
except ImportError:
    from .base import BaseScorer
    from .motion_based import MotionBasedScorer

__all__ = [
    'BaseScorer',
    'MotionBasedScorer'
]