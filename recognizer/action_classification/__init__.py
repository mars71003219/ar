"""
행동 분류 모듈

ST-GCN++ 기반 행동 분류를 제공합니다.
기존 rtmo_gcn_pipeline의 action classification을 새로운 표준 인터페이스에 맞게 재구성했습니다.
"""

try:
    from action_classification.base import BaseActionClassifier
except ImportError:
    from .base import BaseActionClassifier

try:
    try:
        from action_classification.stgcn import STGCNActionClassifier
    except ImportError:
        from .stgcn import STGCNActionClassifier
    STGCN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ST-GCN++ classifier not available: {e}")
    STGCN_AVAILABLE = False
    STGCNActionClassifier = None

__all__ = ['BaseActionClassifier']

if STGCN_AVAILABLE:
    __all__.append('STGCNActionClassifier')