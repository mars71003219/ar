#!/usr/bin/env python3
"""
Inference Mode - 추론 모드 시각화
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from ..core.base_visualizer import BaseVisualizer
from ..renderers.skeleton_renderer import SkeletonRenderer
from ..renderers.overlay_renderer import OverlayRenderer
from ..data.pkl_loader import PKLLoader


class InferenceVisualizerMode(BaseVisualizer):
    """추론 모드 시각화자"""
    
    def __init__(self, input_dir=None, output_dir=None):
        super().__init__(input_dir, output_dir)
        
        # 렌더러 초기화
        self.skeleton_renderer = SkeletonRenderer()
        self.overlay_renderer = OverlayRenderer()
        self.pkl_loader = PKLLoader()
        
        # 모드 설정
        self.mode = 'inference'
        self.show_predictions = True
        self.show_skeleton = True
        self.show_track_info = True
    
    def run(self):
        """추론 모드 시각화 실행"""
        print(f"Starting inference visualization mode...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        self.ensure_output_directory()
        
        # 비디오 파일 찾기
        video_files = self.find_video_files(self.input_dir)
        if not video_files:
            print("No video files found in the input directory")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # 예측 결과 로딩
        prediction_data = self.pkl_loader.load_prediction_data(self.input_dir)
        if not prediction_data:
            print("No prediction data found. Processing without predictions.")
        
        # 각 비디오에 대해 시각화 수행
        for video_file in video_files:
            self._visualize_video(video_file, prediction_data)
    
    def _visualize_video(self, video_file: str, prediction_data: Dict[str, Any]):
        """비디오 시각화 수행"""
        video_name = os.path.basename(video_file)
        base_name = os.path.splitext(video_name)[0]
        
        print(f"\nProcessing video: {video_name}")
        
        # 매칭되는 PKL 파일 찾기
        pkl_files = self.pkl_loader.find_matching_pkl_files(self.input_dir, video_name)
        
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
        output_file = os.path.join(self.output_dir, f"{base_name}_inference_overlay.mp4")
        out = cv2.VideoWriter(output_file, self.fourcc, fps, (width, height))
        
        # 스켈레톤 데이터 로딩
        skeleton_data = None
        if 'skeleton' in pkl_files:
            skeleton_data = self.pkl_loader.load_skeleton_data(pkl_files['skeleton'])
        elif 'main' in pkl_files:
            skeleton_data = self.pkl_loader.load_skeleton_data(pkl_files['main'])
        
        # 프레임별 처리
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 시각화 요소 추가
            result_frame = self._add_inference_overlays(
                frame, frame_idx, skeleton_data, prediction_data, base_name)
            
            # 프레임 저장
            out.write(result_frame)
            frame_idx += 1
            
            # 진행상황 표시
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        # 자원 정리
        cap.release()
        out.release()
        
        print(f"Inference visualization completed: {output_file}")
    
    def _add_inference_overlays(self, frame: np.ndarray, frame_idx: int,
                               skeleton_data: Optional[Dict[str, Any]],
                               prediction_data: Dict[str, Any],
                               video_name: str) -> np.ndarray:
        """추론 모드 오버레이 추가"""
        result_frame = frame.copy()
        
        try:
            # 스켈레톤 그리기
            if self.show_skeleton and skeleton_data:
                result_frame = self._draw_skeleton_data(
                    result_frame, frame_idx, skeleton_data)
            
            # 예측 결과 표시
            if self.show_predictions and prediction_data:
                result_frame = self._draw_prediction_info(
                    result_frame, frame_idx, prediction_data, video_name)
            
            # 기본 정보 표시
            result_frame = self._draw_basic_info(result_frame, frame_idx)
            
        except Exception as e:
            print(f"Error adding overlays to frame {frame_idx}: {e}")
        
        return result_frame
    
    def _draw_skeleton_data(self, frame: np.ndarray, frame_idx: int,
                           skeleton_data: Dict[str, Any]) -> np.ndarray:
        """스켈레톤 데이터 그리기"""
        result_frame = frame.copy()
        
        try:
            # 어노테이션 데이터에서 키포인트 추출
            keypoints_list = self.pkl_loader.extract_keypoints_from_annotation(
                skeleton_data, frame_idx)
            
            # 각 사람에 대해 스켈레톤 그리기
            for i, keypoints in enumerate(keypoints_list):
                color = self.get_color(i)
                result_frame = self.skeleton_renderer.draw_skeleton(
                    result_frame, keypoints, color)
                
                # 트랙 정보 표시 (가능한 경우)
                if self.show_track_info and len(keypoints) > 0:
                    # 바운딩 박스 추정 (키포인트로부터)
                    bbox = self._estimate_bbox_from_keypoints(keypoints)
                    if bbox:
                        result_frame = self.overlay_renderer.draw_track_info(
                            result_frame, i, bbox, position_offset=(0, 0))
            
        except Exception as e:
            print(f"Error drawing skeleton data: {e}")
        
        return result_frame
    
    def _draw_prediction_info(self, frame: np.ndarray, frame_idx: int,
                             prediction_data: Dict[str, Any],
                             video_name: str) -> np.ndarray:
        """예측 정보 그리기"""
        result_frame = frame.copy()
        
        try:
            if 'inference_results' in prediction_data:
                inference_results = prediction_data['inference_results']
                
                # 해당 비디오의 결과 찾기
                for result in inference_results:
                    if (isinstance(result, dict) and 
                        'video_name' in result and 
                        video_name in result['video_name']):
                        
                        # 예측 결과 표시
                        prediction = result.get('prediction', 'Unknown')
                        confidence = result.get('confidence', 0.0)
                        
                        result_frame = self.overlay_renderer.draw_classification_result(
                            result_frame, prediction, confidence)
                        
                        # 추가 통계 정보
                        stats = {
                            'Frame': f"{frame_idx}",
                            'Prediction': prediction
                        }
                        if 'persons_count' in result:
                            stats['Persons'] = result['persons_count']
                        
                        result_frame = self.overlay_renderer.draw_statistics(
                            result_frame, stats)
                        break
            
        except Exception as e:
            print(f"Error drawing prediction info: {e}")
        
        return result_frame
    
    def _draw_basic_info(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """기본 정보 그리기"""
        result_frame = frame.copy()
        
        try:
            # 모드 정보 표시
            mode_text = f"Mode: {self.mode.upper()}"
            result_frame = self.overlay_renderer.draw_text_with_background(
                result_frame, mode_text, (10, frame.shape[0] - 60))
            
            # 프레임 정보 표시
            frame_text = f"Frame: {frame_idx}"
            result_frame = self.overlay_renderer.draw_text_with_background(
                result_frame, frame_text, (10, frame.shape[0] - 90))
            
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
                    # 신룰도 확인 (3차원인 경우)
                    if len(point) >= 3:
                        if point[2] > 0.3:  # 신룰도 임계값
                            valid_points.append([point[0], point[1]])
                    else:
                        # 2차원인 경우 바로 사용
                        valid_points.append([point[0], point[1]])
            
            if len(valid_points) < 2:
                return None
            
            valid_points = np.array(valid_points)
            x_min, y_min = np.min(valid_points, axis=0)
            x_max, y_max = np.max(valid_points, axis=0)
            
            # 마진 추가
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
            description='Inference Mode Visualization Tool')
        
        parser.add_argument('--input-dir', type=str, default=None,
                          help='Input directory containing videos and PKL files')
        parser.add_argument('--output-dir', type=str, default=None,
                          help='Output directory for visualization results')
        parser.add_argument('--no-skeleton', action='store_true',
                          help='Disable skeleton visualization')
        parser.add_argument('--no-predictions', action='store_true',
                          help='Disable prediction information display')
        parser.add_argument('--no-track-info', action='store_true',
                          help='Disable track information display')
        parser.add_argument('--no-statistics', action='store_true',
                          help='Disable statistics display')
        
        return parser
    
    def configure_from_args(self, args):
        """명령행 인수로 설정 적용"""
        if args.input_dir:
            self.input_dir = args.input_dir
        if args.output_dir:
            self.output_dir = args.output_dir
        
        self.show_skeleton = not args.no_skeleton
        self.show_predictions = not args.no_predictions
        self.show_track_info = not args.no_track_info
        self.show_statistics = not getattr(args, 'no_statistics', False)
        
        # 설정 업데이트
        if self.mode_config and hasattr(self.mode_config, 'update_config'):
            self.mode_config.update_config(
                show_skeleton=self.show_skeleton,
                show_predictions=self.show_predictions,
                show_track_info=self.show_track_info,
                show_statistics=self.show_statistics
            )
