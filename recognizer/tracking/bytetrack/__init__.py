"""
ByteTracker Implementation

mmtracking의 ByteTrack 알고리즘을 recognizer 시스템에 맞게 구현했습니다.
pose_estimation과 scoring 사이에서 정확한 트래킹 처리를 제공합니다.
"""

from .byte_tracker import ByteTrackerWrapper, ByteTrackerConfig

__all__ = ['ByteTrackerWrapper', 'ByteTrackerConfig']