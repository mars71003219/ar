"""
파이프라인 모듈

모듈화된 파이프라인 구조:
- separated: 4단계 분리형 파이프라인 (포즈→트래킹→스코어링→분류)
- inference: 실시간 추론 파이프라인 (스트리밍 처리)
- unified: 통합 파이프라인 (단일 비디오 처리)
- base: 공통 베이스 클래스 및 유틸리티
"""

# 베이스 클래스
from .base import BasePipeline, ModuleInitializer, PerformanceTracker

# 분리형 파이프라인
from .separated import (
    SeparatedPipelineConfig, SeparatedPipeline, 
    StageResult, VisualizationData, STGCNData
)

# 실시간 추론 파이프라인
from .inference import (
    RealtimeConfig, RealtimeAlert, InferencePipeline
)

# 통합 파이프라인
from .unified import (
    PipelineConfig, PipelineResult, UnifiedPipeline
)

__all__ = [
    # 베이스 클래스
    'BasePipeline',
    'ModuleInitializer',
    'PerformanceTracker',
    
    # 분리형 파이프라인
    'SeparatedPipelineConfig',
    'SeparatedPipeline',
    'StageResult',
    'VisualizationData', 
    'STGCNData',
    
    # 실시간 추론
    'RealtimeConfig',
    'RealtimeAlert',
    'InferencePipeline',
    
    # 통합 파이프라인
    'PipelineConfig',
    'PipelineResult', 
    'UnifiedPipeline'
]