"""
Import 유틸리티 모듈
중복되는 조건부 import 패턴을 통합 관리
"""

import logging
from typing import Any, Tuple, Optional

logger = logging.getLogger(__name__)


def safe_import_data_structure():
    """데이터 구조 모듈 안전 import"""
    try:
        from ..utils.data_structure import PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult
        return PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult
    except ImportError:
        from utils.data_structure import PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult
        return PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult


def safe_import_pose_structures():
    """포즈 구조체만 안전 import"""
    try:
        from ..utils.data_structure import PersonPose, FramePoses
        return PersonPose, FramePoses
    except ImportError:
        from utils.data_structure import PersonPose, FramePoses
        return PersonPose, FramePoses


def safe_import_classification_structures():
    """분류 구조체만 안전 import"""
    try:
        from ..utils.data_structure import ClassificationResult, AnnotationData, PipelineResult
        return ClassificationResult, AnnotationData, PipelineResult
    except ImportError:
        from utils.data_structure import ClassificationResult, AnnotationData, PipelineResult
        return ClassificationResult, AnnotationData, PipelineResult


def safe_import_module(module_name: str, fallback_module: str, *items) -> Tuple[Any, ...]:
    """
    일반적인 안전 import 함수
    
    Args:
        module_name: 상대 import 모듈 이름 (예: "..utils.data_structure")
        fallback_module: 절대 import 모듈 이름 (예: "utils.data_structure")
        *items: import할 항목들
    
    Returns:
        import된 항목들의 튜플
    """
    try:
        module = __import__(module_name, fromlist=items)
        return tuple(getattr(module, item) for item in items)
    except ImportError:
        try:
            module = __import__(fallback_module, fromlist=items)
            return tuple(getattr(module, item) for item in items)
        except ImportError as e:
            logger.error(f"Failed to import {items} from {module_name} or {fallback_module}: {e}")
            raise


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    표준화된 로거 설정
    중복되는 로거 설정 코드를 통합
    
    Args:
        name: 로거 이름
        level: 로그 레벨
    
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger