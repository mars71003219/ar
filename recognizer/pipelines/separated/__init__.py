"""
분리형 파이프라인 모듈
"""

from .config import SeparatedPipelineConfig
from .data_structures import StageResult, VisualizationData, STGCNData
from .pipeline import SeparatedPipeline

# Stage 함수들
from .stage1_poses import process_stage1_pose_extraction, load_stage1_result, validate_stage1_result
from .stage2_tracking import process_stage2_tracking_scoring, load_stage2_result, validate_stage2_result
from .stage3_classification import process_stage3_classification, load_stage3_result, validate_stage3_result
from .stage4_unified import process_stage4_unified_dataset, load_stgcn_dataset, validate_stage4_result

__all__ = [
    # 메인 클래스
    'SeparatedPipelineConfig',
    'SeparatedPipeline',
    
    # 데이터 구조
    'StageResult',
    'VisualizationData', 
    'STGCNData',
    
    # Stage 1
    'process_stage1_pose_extraction',
    'load_stage1_result',
    'validate_stage1_result',
    
    # Stage 2
    'process_stage2_tracking_scoring',
    'load_stage2_result',
    'validate_stage2_result',
    
    # Stage 3
    'process_stage3_classification',
    'load_stage3_result',
    'validate_stage3_result',
    
    # Stage 4
    'process_stage4_unified_dataset',
    'load_stgcn_dataset',
    'validate_stage4_result'
]