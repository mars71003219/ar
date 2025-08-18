"""
간단한 설정 로더
실제로 사용되는 기능만 포함
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# 설정 파일 기본 경로
CONFIG_DIR = Path(__file__).parent.parent / "configs"

logger = logging.getLogger(__name__)


class SimpleConfigLoader:
    """간단한 설정 로더"""
    
    def __init__(self, config_dir: Union[str, Path] = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    
    def load_yaml(self, config_file: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        # 절대 경로인지 확인
        if Path(config_file).is_absolute():
            config_path = Path(config_file)
        else:
            # 현재 디렉토리에서 먼저 찾기
            current_path = Path(config_file)
            if current_path.exists():
                config_path = current_path
            else:
                # configs 디렉토리에서 찾기
                config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # base_config 상속 처리
            if 'base_config' in config:
                base_config_file = config.pop('base_config')
                base_config = self.load_yaml(base_config_file)
                config = self._merge_configs(base_config, config)
            
            logger.info(f"Loaded config from: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """설정 병합 (재귀적)"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


def load_config(
    config_file: str = "config.yaml",
    config_dir: Optional[Path] = None,
    mode: Optional[str] = None,
    args_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """편의 함수: 설정 로드"""
    loader = SimpleConfigLoader(config_dir)
    config = loader.load_yaml(config_file)
    
    # mode와 args_dict는 호환성을 위해 받지만 현재는 사용하지 않음
    # 필요시 향후 확장 가능
    if mode:
        logger.info(f"Mode parameter received: {mode}")
    
    return config