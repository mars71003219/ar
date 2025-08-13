#!/usr/bin/env python3
"""
Base Visualizer - 기본 시각화 클래스
"""

import os
import cv2
import json
import pickle
import argparse
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

try:
    from configs.visualizer_config import config as default_config
except ImportError:
    try:
        from ..configs.visualizer_config import config as default_config
    except ImportError:
        print("Warning: Could not import visualizer config. Using default settings.")
        default_config = None


class BaseVisualizer(ABC):
    """기본 시각화 클래스"""
    
    def __init__(self, input_dir=None, output_dir=None, config=None):
        """초기화"""
        # 설정 로드
        self.config = config or default_config
        if self.config is None:
            # 폴백 설정
            self.config = type('Config', (), {
                'default_input_dir': os.getcwd(),
                'default_output_dir': os.path.join(os.getcwd(), 'visualization_output'),
                'fps': 30.0,
                'fourcc_codec': 'mp4v',
                'colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
                'get_color': lambda index: [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)][index % 6]
            })()
        
        # 경로 설정
        self.input_dir = input_dir or self.config.default_input_dir
        self.output_dir = output_dir or self.config.default_output_dir
        
        # 비디오 설정
        self.fps = self.config.fps
        self.fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc_codec)
        
        # 색상 팔레트
        self.colors = self.config.colors
    
    def ensure_output_directory(self):
        """출력 디렉토리 확보"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """트랙 ID에 따른 색상 반환"""
        if hasattr(self.config, 'get_color'):
            return self.config.get_color(track_id)
        return self.colors[track_id % len(self.colors)]
    
    def safe_load_data(self, file_path: str) -> Optional[Any]:
        """안전한 데이터 로딩"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"Unsupported file format: {file_path}")
                return None
        except Exception as e:
            print(f"Failed to load data from {file_path}: {e}")
            return None
    
    def find_video_files(self, directory: str) -> List[str]:
        """비디오 파일 검색"""
        if hasattr(self.config, 'supported_video_extensions'):
            video_extensions = self.config.supported_video_extensions
        else:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        video_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return sorted(video_files)
    
    @abstractmethod
    def run(self):
        """실행 - 서브클래스에서 구현"""
        pass
    
    @abstractmethod
    def create_parser(self) -> argparse.ArgumentParser:
        """명령행 파서 생성 - 서브클래스에서 구현"""
        pass
