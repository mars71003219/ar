#!/usr/bin/env python3
"""
Enhanced Visualizer Configuration - 개선된 시각화 설정 파일
"""

import os
from typing import Dict, List, Tuple, Any


class VisualizerConfig:
    """시각화 설정 클래스"""
    
    def __init__(self):
        # --- 기본 경로 설정 ---
        self.default_input_dir = os.getcwd()
        self.default_output_dir = os.path.join(os.getcwd(), 'visualization_output')
        
        # --- 일반 시각화 설정 ---
        self.max_persons = 4
        self.confidence_threshold = 0.3
        self.verbose = True
        self.supported_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        # --- 비디오 설정 ---
        self.fps = 30.0
        self.fourcc_codec = 'mp4v'  # 'mp4v', 'XVID', 'H264'
        
        # --- 색상 설정 (BGR 형식) ---
        self.colors = [
            (255, 0, 0),    # 빨간색
            (0, 255, 0),    # 초록색
            (0, 0, 255),    # 파란색
            (255, 255, 0),  # 노란색
            (255, 0, 255),  # 자주색
            (0, 255, 255),  # 청록색
            (128, 0, 128),  # 보라색
            (255, 165, 0),  # 주황색
            (0, 128, 128),  # 청록색 (어두운)
            (128, 128, 0),  # 올리브색
            (75, 0, 130),   # 남색
            (220, 20, 60)   # 진홍색
        ]
        
        # --- 특별 색상 ---
        self.fight_color = (0, 0, 255)      # 빨간색 (Fight)
        self.nonfight_color = (0, 255, 0)   # 초록색 (NonFight)
        self.text_color = (255, 255, 255)   # 흰색 (텍스트)
        self.bg_color = (0, 0, 0)           # 검은색 (배경)
        
        # --- 폰트 설정 ---
        self.font_scale = 0.7
        self.font_thickness = 2
        self.title_font_scale = 0.8
        self.title_font_thickness = 2
        
        # --- UI 레이아웃 설정 ---
        self.box_padding = 10
        self.line_thickness = 2
        self.keypoint_radius = 3
        
        # --- 스켈레톤 설정 ---
        # COCO 17 keypoint 연결 정보 (1-based → 0-based로 자동 변환)
        self.skeleton_connections = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 다리
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],          # 목, 어깨
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],          # 팔, 얼굴
            [2, 4], [3, 5], [4, 6], [5, 7]                     # 얼굴에서 몸으로
        ]
        
        # 스켈레톤 색상 (기본은 노란색)
        self.skeleton_colors = [(0, 255, 255)] * len(self.skeleton_connections)
        self.keypoint_color = (0, 0, 255)  # 키포인트는 빨간색
        
        # --- 모드별 설정 ---
        self.inference_mode_config = {
            'show_predictions': True,
            'show_skeleton': True,
            'show_track_info': True,
            'show_statistics': True
        }
        
        self.separated_mode_config = {
            'show_windows': True,
            'show_skeleton': True,
            'show_track_info': True,
            'window_size': 60,  # 프레임
            'show_window_info': True
        }
        
        # --- PKL 파일 검색 설정 ---
        self.pkl_search_patterns = [
            '{base_name}.pkl',
            '{base_name}_skeleton.pkl',
            '{base_name}_pose.pkl',
            '{base_name}_annotation.pkl',
            '{base_name}_window.pkl'
        ]
        
        # --- 캐시 설정 ---
        self.enable_cache = True
        self.cache_size_limit = 100  # 최대 캐시할 파일 수
        
        # --- 성능 설정 ---
        self.progress_update_interval = 30  # 프레임
        self.max_concurrent_processes = 4
        
        # --- 디버그 설정 ---
        self.debug_mode = False
        self.save_debug_frames = False
        self.debug_output_dir = 'debug_frames'
    
    def get_color(self, index: int) -> Tuple[int, int, int]:
        """인덱스에 따른 색상 반환"""
        return self.colors[index % len(self.colors)]
    
    def get_inference_config(self) -> Dict[str, Any]:
        """추론 모드 설정 반환"""
        return self.inference_mode_config.copy()
    
    def get_separated_config(self) -> Dict[str, Any]:
        """분리된 모드 설정 반환"""
        return self.separated_mode_config.copy()
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
    
    def get_pkl_patterns(self, base_name: str) -> List[str]:
        """PKL 파일 검색 패턴 생성"""
        return [pattern.format(base_name=base_name) for pattern in self.pkl_search_patterns]
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisualizerConfig':
        """딕셔너리로부터 설정 생성"""
        config = cls()
        config.update_config(**config_dict)
        return config


# 전역 설정 인스턴스
config = VisualizerConfig()


# 레거시 호환성을 위한 변수들 (기존 코드와의 호환성)
input_dir = config.default_input_dir
output_dir = config.default_output_dir
num_persons = config.max_persons
confidence_threshold = config.confidence_threshold
verbose = config.verbose
supported_video_extensions = [ext.lstrip('.') for ext in config.supported_video_extensions]
person_colors = config.colors[:5]  # 기존 5개 색상만
fight_color = config.fight_color
nonfight_color = config.nonfight_color
font_scale = config.font_scale
font_thickness = config.font_thickness
title_font_scale = config.title_font_scale
title_font_thickness = config.title_font_thickness
output_video_codec = config.fourcc_codec
skeleton_connections = config.skeleton_connections

# 새로운 설정들
SAVE_OVERLAY_VIDEO = True
OVERLAY_SUB_DIR = 'overlay'
PKL_SEARCH_SUBDIR = 'windows'

# UI 위치 설정 (레거시 호환성)
window_info_x = 10
window_info_y_start = 20
window_info_y_step = 15
frame_info_margin = 5
final_result_margin = 5
consecutive_threshold = 3