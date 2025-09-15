"""
파이프라인 모듈

모듈화된 파이프라인 구조:
- separated: 4단계 분리형 파이프라인 (포즈→트래킹→스코어링→분류)
- dual_service: 듀얼 서비스 파이프라인 (Fight + Falldown 동시 처리)
- analysis: 분석 전용 파이프라인 (JSON/PKL 파일 생성)
- base: 공통 베이스 클래스 및 유틸리티
"""

# 베이스 클래스
from .base import BasePipeline, ModuleInitializer, PerformanceTracker

# 분리형 파이프라인
from .separated import (
    SeparatedPipeline, 
    StageResult, VisualizationData, STGCNData
)

# 실시간 추론 파이프라인 - DualServicePipeline으로 통합됨 (inference/ 디렉토리 제거됨)

# 통합 파이프라인 - 사용되지 않아 제거됨

# 설정 클래스들은 통합 설정 시스템으로 대체
# SeparatedPipelineConfig, RealtimeConfig, RealtimeAlert, PipelineConfig, PipelineResult

__all__ = [
    # 베이스 클래스
    'BasePipeline',
    'ModuleInitializer',
    'PerformanceTracker',
    
    # 분리형 파이프라인
    # 'SeparatedPipelineConfig',  # 통합 설정 시스템으로 대체
    'SeparatedPipeline',
    'StageResult',
    'VisualizationData', 
    'STGCNData',
    
    # 실시간 추론 - DualServicePipeline으로 통합됨
    # 'RealtimeConfig',          # 통합 설정 시스템으로 대체
    # 'RealtimeAlert',           # 통합 설정 시스템으로 대체
    # 'InferencePipeline',       # DualServicePipeline으로 통합됨
    
    # 통합 파이프라인 - 사용되지 않아 제거됨
    # 'PipelineConfig',          # 통합 설정 시스템으로 대체
    # 'PipelineResult',          # 통합 설정 시스템으로 대체
    # 'UnifiedPipeline'          # 사용되지 않아 제거됨
]