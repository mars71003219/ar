"""
통합 파이프라인 모듈
"""

# from .config import PipelineConfig, PipelineResult  # 통합 설정 시스템으로 대체
from .pipeline import UnifiedPipeline

__all__ = [
    # 'PipelineConfig',     # 통합 설정 시스템으로 대체
    # 'PipelineResult',     # 통합 설정 시스템으로 대체
    'UnifiedPipeline'
]