#!/usr/bin/env python3
"""
Separated Mode - 분리된 파이프라인 모드 시각화
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from ..core.base_visualizer import BaseVisualizer
from ..renderers.skeleton_renderer import SkeletonRenderer
from ..renderers.overlay_renderer import OverlayRenderer
from ..data.pkl_loader import PKLLoader

try:
    from ..configs.visualizer_config import config as default_config
except ImportError:
    try:
        from configs.visualizer_config import config as default_config
    except ImportError:
        print("Warning: Could not import visualizer config for separated mode. Using defaults.")
        default_config = None


class SeparatedVisualizerMode(BaseVisualizer):
    """분리된 파이프라인 모드 시각화자"""
    
    def __init__(self, input_dir=None, output_dir=None, config=None):
        # 설정 로드
        self.mode_config = config or default_config
        super().__init__(input_dir, output_dir, self.mode_config)
        
        # 렌더러 초기화 (설정 전달)
        self.skeleton_renderer = SkeletonRenderer(self.mode_config)
        self.overlay_renderer = OverlayRenderer(self.mode_config)
        self.pkl_loader = PKLLoader(self.mode_config)
        
        # 모드 설정
        self.mode = 'separated'
        
        # 설정 파일에서 모드별 설정 로드
        if self.mode_config:
            mode_settings = self.mode_config.get_separated_config()
            self.show_windows = mode_settings.get('show_windows', True)
            self.show_skeleton = mode_settings.get('show_skeleton', True)
            self.show_track_info = mode_settings.get('show_track_info', True)
            self.window_size = mode_settings.get('window_size', 60)
            self.show_window_info = mode_settings.get('show_window_info', True)
        else:
            # 폴백 설정
            self.show_windows = True
            self.show_skeleton = True
            self.show_track_info = True
            self.window_size = 60
            self.show_window_info = True
    
    def run(self):
        """분리된 모드 시각화 실행"""
        print(f"Starting separated pipeline visualization mode...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        self.ensure_output_directory()
        
        # 비디오 파일 찾기
        video_files = self.find_video_files(self.input_dir)
        if not video_files:
            print("No video files found in the input directory")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # 각 비디오에 대해 시각화 수행
        for video_file in video_files:
            self._visualize_video(video_file)
    
    def _visualize_video(self, video_file: str):
        """비디오 시각화 수행"""
        video_name = os.path.basename(video_file)
        base_name = os.path.splitext(video_name)[0]
        
        print(f"\nProcessing video: {video_name}")
        
        # 매칭되는 PKL 파일 찾기
        pkl_files = self.pkl_loader.find_matching_pkl_files(self.input_dir, video_name)
        
        if not pkl_files:
            print(f"No PKL files found for {video_name}. Skipping.")
            return
        
        # 윈도우 데이터 로딩
        window_data = self._load_window_data(pkl_files)
        if not window_data:
            print(f"Failed to load window data for {video_name}")
            return
        
        # 비디오 캐프처 열기
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            return
        
        # 비디오 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 비디오 설정
        output_file = os.path.join(self.output_dir, f"{base_name}_separated_overlay.mp4")
        out = cv2.VideoWriter(output_file, self.fourcc, fps, (width, height))
        
        # 프레임별 처리
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 시각화 요소 추가
            result_frame = self._add_separated_overlays(
                frame, frame_idx, window_data, base_name)
            
            # 프레임 저장
            out.write(result_frame)
            frame_idx += 1
            
            # 진행상황 표시
            if self.mode_config and hasattr(self.mode_config, 'progress_update_interval'):
                interval = self.mode_config.progress_update_interval
            else:
                interval = 30
                
            if frame_idx % interval == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        # 자원 정리
        cap.release()
        out.release()
        
        print(f"Separated visualization completed: {output_file}")
    
    def _load_window_data(self, pkl_files: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        """윈도우 데이터 로딩"""
        window_data = []
        
        try:
            # 메인 PKL 파일에서 데이터 로딩
            main_file = pkl_files.get('main') or pkl_files.get('annotation')
            if main_file:
                data = self.pkl_loader.safe_load_data(main_file)
                if data:
                    # 다양한 데이터 구조 지원
                    if isinstance(data, list):
                        # 리스트 형태의 윈도우 데이터
                        window_data = data
                    elif isinstance(data, dict):
                        # 딕셔너리 형태인 경우
                        if 'windows' in data:
                            window_data = data['windows']
                        elif 'results' in data:
                            window_data = data['results']
                        elif 'annotation' in data:
                            # 단일 어노테이션인 경우 가짜 윈도우 생성
                            window_data = [{
                                'window_idx': 0,
                                'start_frame': 0,
                                'end_frame': self.window_size,
                                'annotation': data['annotation']
                            }]
                        else:
                            # 단일 어노테이션으로 간주
                            window_data = [{
                                'window_idx': 0,
                                'start_frame': 0,
                                'end_frame': self.window_size,
                                'annotation': data
                            }]
            
            print(f"Loaded {len(window_data)} windows")
            return window_data if window_data else None
            
        except Exception as e:
            print(f"Error loading window data: {e}")
            return None
    
    def _add_separated_overlays(self, frame: np.ndarray, frame_idx: int,
                               window_data: List[Dict[str, Any]],
                               video_name: str) -> np.ndarray:
        """분리된 모드 오버레이 추가"""
        result_frame = frame.copy()
        
        try:
            # 현재 프레임에 해당하는 윈도우 찾기
            current_window = self._find_current_window(frame_idx, window_data)
            
            if current_window:
                # 윈도우 감지 정보 표시
                if self.show_windows:
                    result_frame = self.overlay_renderer.draw_window_detection_info(
                        result_frame, current_window)
                
                # 스켈레톤 그리기
                if self.show_skeleton:
                    result_frame = self._draw_window_skeleton(
                        result_frame, frame_idx, current_window)
            
            # 기본 정보 표시
            result_frame = self._draw_basic_info(result_frame, frame_idx, len(window_data))
            
        except Exception as e:
            print(f"Error adding separated overlays to frame {frame_idx}: {e}")
        
        return result_frame
    
    def _find_current_window(self, frame_idx: int, 
                            window_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """현재 프레임에 해당하는 윈도우 찾기"""
        for window in window_data:
            start_frame = window.get('start_frame', 0)
            end_frame = window.get('end_frame', self.window_size)
            
            if start_frame <= frame_idx < end_frame:
                return window
        
        return None
    
    def _draw_window_skeleton(self, frame: np.ndarray, frame_idx: int,
                             window: Dict[str, Any]) -> np.ndarray:
        """윈도우 스켈레톤 그리기"""
        result_frame = frame.copy()
        
        try:
            if 'annotation' not in window:
                return result_frame
            
            annotation = window['annotation']
            
            # 윈도우 내에서의 상대적 프레임 인덱스 계산
            start_frame = window.get('start_frame', 0)
            relative_frame_idx = max(0, frame_idx - start_frame)
            
            # 어노테이션에서 키포인트 추출
            keypoints_list = self.pkl_loader.extract_keypoints_from_annotation(
                annotation, relative_frame_idx)
            
            # 각 사람에 대해 스켈레톤 그리기
            for i, keypoints in enumerate(keypoints_list):
                color = self.get_color(i)
                result_frame = self.skeleton_renderer.draw_skeleton(
                    result_frame, keypoints, color)
                
                # 트랙 정보 표시
                if self.show_track_info and len(keypoints) > 0:
                    bbox = self._estimate_bbox_from_keypoints(keypoints)
                    if bbox:
                        # 트랙 ID 계산 (예: person_0, person_1...)
                        track_id = f"W{window.get('window_idx', 0)}_P{i}"
                        result_frame = self.overlay_renderer.draw_track_info(
                            result_frame, track_id, bbox, position_offset=(0, 0))
            
        except Exception as e:
            print(f"Error drawing window skeleton: {e}")
        
        return result_frame
    
    def _draw_basic_info(self, frame: np.ndarray, frame_idx: int, 
                        total_windows: int) -> np.ndarray:
        """기본 정보 그리기"""
        result_frame = frame.copy()
        
        try:
            # 모드 정보 표시
            mode_text = f"Mode: {self.mode.upper()}"
            result_frame = self.overlay_renderer.draw_text_with_background(
                result_frame, mode_text, (10, frame.shape[0] - 90))
            
            # 프레임 정보 표시
            frame_text = f"Frame: {frame_idx}"
            result_frame = self.overlay_renderer.draw_text_with_background(
                result_frame, frame_text, (10, frame.shape[0] - 60))
            
            # 윈도우 정보 표시
            window_text = f"Windows: {total_windows}"
            result_frame = self.overlay_renderer.draw_text_with_background(
                result_frame, window_text, (10, frame.shape[0] - 30))
            
        except Exception as e:
            print(f"Error drawing basic info: {e}")
        
        return result_frame
    
    def _estimate_bbox_from_keypoints(self, keypoints: np.ndarray) -> Optional[List[float]]:
        """키포인트로부터 바운딩 박스 추정"""
        try:
            if len(keypoints) == 0:
                return None
            
            # 유효한 키포인트만 사용
            valid_points = []
            for point in keypoints:
                if len(point) >= 2:
                    # 신룰도 임계값 (설정 파일에서 가져오기)
                    if self.mode_config and hasattr(self.mode_config, 'confidence_threshold'):
                        threshold = self.mode_config.confidence_threshold
                    else:
                        threshold = 0.3
                    
                    # 신룰도 확인 (3차원인 경우)
                    if len(point) >= 3:
                        if point[2] > threshold:
                            valid_points.append([point[0], point[1]])
                    else:
                        # 2차원인 경우 바로 사용
                        valid_points.append([point[0], point[1]])
            
            if len(valid_points) < 2:
                return None
            
            valid_points = np.array(valid_points)
            x_min, y_min = np.min(valid_points, axis=0)
            x_max, y_max = np.max(valid_points, axis=0)
            
            # 마진 추가 (설정 파일에서 가져오기)
            if self.mode_config and hasattr(self.mode_config, 'box_padding'):
                margin = self.mode_config.box_padding
            else:
                margin = 10
            return [
                max(0, x_min - margin),
                max(0, y_min - margin),
                x_max + margin,
                y_max + margin
            ]
            
        except Exception as e:
            print(f"Error estimating bbox from keypoints: {e}")
            return None
    
    def create_parser(self):
        """명령행 파서 생성"""
        import argparse
        parser = argparse.ArgumentParser(
            description='Separated Pipeline Mode Visualization Tool')
        
        parser.add_argument('--input-dir', type=str, default=None,
                          help='Input directory containing videos and PKL files')
        parser.add_argument('--output-dir', type=str, default=None,
                          help='Output directory for visualization results')
        parser.add_argument('--window-size', type=int, default=60,
                          help='Window size in frames')
        parser.add_argument('--no-windows', action='store_true',
                          help='Disable window information display')
        parser.add_argument('--no-skeleton', action='store_true',
                          help='Disable skeleton visualization')
        parser.add_argument('--no-track-info', action='store_true',
                          help='Disable track information display')
        
        return parser
    
    def configure_from_args(self, args):
        """명령행 인수로 설정 적용"""
        if args.input_dir:
            self.input_dir = args.input_dir
        if args.output_dir:
            self.output_dir = args.output_dir
        
        self.window_size = args.window_size
        self.show_windows = not args.no_windows
        self.show_skeleton = not args.no_skeleton
        self.show_track_info = not args.no_track_info
        
        # 설정 업데이트
        if self.mode_config and hasattr(self.mode_config, 'update_config'):
            self.mode_config.update_config(
                window_size=self.window_size,
                show_windows=self.show_windows,
                show_skeleton=self.show_skeleton,
                show_track_info=self.show_track_info
            )
