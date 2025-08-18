"""
모드 매니저 - 모든 실행 모드를 통합 관리
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModeManager:
    """실행 모드 통합 매니저"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._modes = {}
        self._register_modes()
    
    def _register_modes(self):
        """모든 모드 등록"""
        # 추론 모드들
        from .inference_modes import AnalysisMode, RealtimeMode, VisualizeMode
        self._modes['inference.analysis'] = AnalysisMode
        self._modes['inference.realtime'] = RealtimeMode
        self._modes['inference.visualize'] = VisualizeMode
        
        # 어노테이션 모드들
        from .annotation_modes import Stage1Mode, Stage2Mode, Stage3Mode, AnnotationVisualizeMode
        self._modes['annotation.stage1'] = Stage1Mode
        self._modes['annotation.stage2'] = Stage2Mode
        self._modes['annotation.stage3'] = Stage3Mode
        self._modes['annotation.visualize'] = AnnotationVisualizeMode
        
        logger.info(f"Registered {len(self._modes)} modes")
    
    def execute(self, mode_name: str) -> bool:
        """모드 실행"""
        if mode_name not in self._modes:
            logger.error(f"Unknown mode: {mode_name}")
            logger.info(f"Available modes: {list(self._modes.keys())}")
            return False
        
        try:
            mode_class = self._modes[mode_name]
            mode_instance = mode_class(self.config)
            
            logger.info(f"Executing mode: {mode_name}")
            return mode_instance.execute()
            
        except Exception as e:
            logger.error(f"Failed to execute mode {mode_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_modes(self) -> Dict[str, str]:
        """사용 가능한 모드 목록"""
        return {
            # 추론 모드
            'inference.analysis': '분석 모드 - JSON/PKL 파일 생성',
            'inference.realtime': '실시간 모드 - 실시간 디스플레이',
            'inference.visualize': '시각화 모드 - PKL 기반 오버레이',
            
            # 어노테이션 모드
            'annotation.stage1': 'Stage 1 - 포즈 추정 결과 생성',
            'annotation.stage2': 'Stage 2 - 트래킹 및 정렬 결과 생성',
            'annotation.stage3': 'Stage 3 - 통합 데이터셋 생성',
            'annotation.visualize': '어노테이션 시각화 모드'
        }


class BaseMode(ABC):
    """모드 베이스 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode_config = self._get_mode_config()
    
    @abstractmethod
    def execute(self) -> bool:
        """모드 실행"""
        pass
    
    @abstractmethod
    def _get_mode_config(self) -> Dict[str, Any]:
        """모드별 설정 가져오기"""
        pass
    
    def _validate_config(self, required_keys: list) -> bool:
        """설정 검증"""
        for key in required_keys:
            if key not in self.mode_config:
                logger.error(f"Missing required config: {key}")
                return False
        return True