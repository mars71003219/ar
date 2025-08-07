#!/usr/bin/env python3
"""
Base processor class for pipeline processing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseProcessor(ABC):
    """기본 처리기 추상 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """처리기 초기화"""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """데이터 처리"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """리소스 정리"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """입력 데이터 검증"""
        return input_data is not None
    
    def get_status(self) -> Dict[str, Any]:
        """처리기 상태 반환"""
        return {
            'initialized': self.is_initialized,
            'config': self.config
        }