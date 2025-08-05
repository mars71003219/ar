#!/usr/bin/env python3
"""
설정 파일 로더 - 다양한 형태의 설정 파일을 로드하고 파싱
"""

import os
import importlib.util
from typing import Dict, Any, Optional


class ConfigLoader:
    """설정 파일 로더 클래스"""
    
    def __init__(self):
        self.config = None
    
    def load_config_from_file(self, config_file: str) -> Any:
        """Python 설정 파일에서 설정 로드"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # 설정 파일의 절대 경로
        config_path = os.path.abspath(config_file)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        
        # 동적으로 모듈 로드
        spec = importlib.util.spec_from_file_location(config_name, config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Config 클래스 찾기 (일반적으로 *Config 형태)
        config_class = None
        for attr_name in dir(config_module):
            attr = getattr(config_module, attr_name)
            if (isinstance(attr, type) and 
                attr_name.endswith('Config') and 
                attr_name != 'Config'):
                config_class = attr
                break
        
        if config_class is None:
            # Config 클래스가 없으면 모듈 자체를 설정으로 사용
            self.config = config_module
        else:
            self.config = config_class
        
        return self.config
    
    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> type:
        """딕셔너리에서 설정 로드"""
        class DictConfig:
            pass
        
        for key, value in config_dict.items():
            setattr(DictConfig, key, value)
        
        # 가중치 메서드 추가
        if hasattr(DictConfig, 'movement_weight'):
            @classmethod
            def get_weights(cls):
                return [
                    cls.movement_weight,
                    cls.position_weight,
                    cls.interaction_weight,
                    cls.temporal_weight,
                    cls.persistence_weight
                ]
            DictConfig.get_weights = get_weights
        
        self.config = DictConfig
        return self.config
    
    def override_config(self, overrides: Dict[str, Any]):
        """설정 값 덮어쓰기"""
        if self.config is None:
            raise ValueError("No config loaded. Load config first.")
        
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"Config override: {key} = {value}")
            else:
                print(f"Warning: Unknown config key '{key}' ignored")
    
    def get_config(self) -> Any:
        """현재 로드된 설정 반환"""
        if self.config is None:
            raise ValueError("No config loaded. Load config first.")
        return self.config
    
    def validate_config(self) -> list:
        """설정 검증"""
        if self.config is None:
            return ["No config loaded"]
        
        if hasattr(self.config, 'validate_config'):
            return self.config.validate_config()
        else:
            # 기본 검증
            errors = []
            
            # 필수 속성 확인
            required_attrs = [
                'input_dir', 'output_dir', 'detector_config', 
                'detector_checkpoint', 'score_thr', 'nms_thr'
            ]
            
            for attr in required_attrs:
                if not hasattr(self.config, attr):
                    errors.append(f"Missing required config attribute: {attr}")
            
            return errors
    
    def print_config(self):
        """설정 출력"""
        if self.config is None:
            print("No config loaded")
            return
        
        if hasattr(self.config, 'print_config'):
            self.config.print_config()
        else:
            # 기본 설정 출력
            print("=" * 70)
            print(" Configuration Settings")
            print("=" * 70)
            
            for attr_name in sorted(dir(self.config)):
                if not attr_name.startswith('_') and not callable(getattr(self.config, attr_name)):
                    value = getattr(self.config, attr_name)
                    print(f"{attr_name}: {value}")
            
            print("=" * 70)


def load_config(config_file: Optional[str] = None, 
                config_dict: Optional[Dict[str, Any]] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Any:
    """
    설정 로드 헬퍼 함수
    
    Args:
        config_file: 설정 파일 경로
        config_dict: 설정 딕셔너리
        overrides: 덮어쓸 설정 값들
    
    Returns:
        설정 클래스/객체
    """
    loader = ConfigLoader()
    
    if config_file:
        config = loader.load_config_from_file(config_file)
    elif config_dict:
        config = loader.load_config_from_dict(config_dict)
    else:
        # 기본 설정 로드
        default_config_path = os.path.join(
            os.path.dirname(__file__), 'default_config.py'
        )
        config = loader.load_config_from_file(default_config_path)
    
    # 덮어쓰기 적용
    if overrides:
        loader.override_config(overrides)
    
    # 설정 검증
    errors = loader.validate_config()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Configuration validation failed")
    
    return config


# 설정 파일에서 사용 가능한 헬퍼 함수들
def get_workspace_path(relative_path: str) -> str:
    """워크스페이스 기준 경로 반환"""
    return f"/workspace/{relative_path.lstrip('/')}"

def get_data_path(relative_path: str) -> str:
    """데이터 디렉토리 기준 경로 반환"""
    return f"/aivanas/raw/surveillance/action/violence/action_recognition/data/{relative_path.lstrip('/')}"

def get_output_path(relative_path: str) -> str:
    """출력 디렉토리 기준 경로 반환"""
    return f"/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/{relative_path.lstrip('/')}"