"""
파이프라인 베이스 클래스 및 공통 유틸리티
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..utils.factory import ModuleFactory


class BasePipeline(ABC):
    """파이프라인 베이스 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.factory = ModuleFactory()
        self._initialized = False
    
    @abstractmethod
    def initialize_pipeline(self) -> bool:
        """파이프라인 초기화 (구현 필요)"""
        pass
    
    def cleanup(self):
        """리소스 정리"""
        modules = ['pose_estimator', 'tracker', 'scorer', 'classifier', 'window_processor']
        
        for module_name in modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module and hasattr(module, 'cleanup'):
                    try:
                        module.cleanup()
                    except Exception as e:
                        logging.warning(f"Failed to cleanup {module_name}: {e}")
        
        self._initialized = False
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        if not self.initialize_pipeline():
            raise RuntimeError("Failed to initialize pipeline")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.cleanup()


class ModuleInitializer:
    """모듈 초기화 헬퍼 클래스"""
    
    @staticmethod
    def init_pose_estimator(factory: ModuleFactory, config_dict: Dict[str, Any]):
        """포즈 추정기 초기화"""
        return factory.create_pose_estimator(config_dict)
    
    @staticmethod
    def init_tracker(factory: ModuleFactory, config_dict: Dict[str, Any]):
        """트래커 초기화"""
        return factory.create_tracker(config_dict)
    
    @staticmethod
    def init_scorer(factory: ModuleFactory, config_dict: Dict[str, Any]):
        """스코어링 모듈 초기화"""
        return factory.create_scorer(config_dict)
    
    @staticmethod
    def init_classifier(factory: ModuleFactory, config_dict: Dict[str, Any]):
        """분류기 초기화"""
        return factory.create_classifier(config_dict)
    
    @staticmethod
    def init_window_processor(factory: ModuleFactory, window_size: int, window_stride: int):
        """윈도우 프로세서 초기화"""
        return factory.create_window_processor({
            'window_size': window_size,
            'window_stride': window_stride
        })


class PerformanceTracker:
    """성능 추적 헬퍼 클래스"""
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    def update(self, processing_time: float):
        """성능 통계 업데이트"""
        self.stats['total_processed'] += 1
        self.stats['total_time'] += processing_time
        self.stats['avg_time'] = self.stats['total_time'] / self.stats['total_processed']
    
    def get_stats(self) -> Dict[str, float]:
        """통계 반환"""
        return self.stats.copy()
    
    def reset(self):
        """통계 초기화"""
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }