#!/usr/bin/env python3
"""
Video processing utilities for RTMO tracking pipeline
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from .rtmo_tracking_pipeline import RTMOTrackingPipeline
from .visualization import TrackingVisualizer


class VideoProcessor:
    """
    비디오 파일 처리를 위한 유틸리티 클래스
    """
    
    def __init__(self, 
                 pipeline: RTMOTrackingPipeline,
                 visualizer: Optional[TrackingVisualizer] = None):
        """
        Args:
            pipeline: RTMO 트래킹 파이프라인
            visualizer: 시각화기 (None이면 기본 설정으로 생성)
        """
        self.pipeline = pipeline
        self.visualizer = visualizer or TrackingVisualizer()
        
        # 성능 측정용
        self.process_times = []
        
    def process_video(self,
                     input_path: str,
                     output_path: Optional[str] = None,
                     show_video: bool = True,
                     save_video: bool = True,
                     max_frames: Optional[int] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        비디오 파일 처리
        
        Args:
            input_path: 입력 비디오 경로
            output_path: 출력 비디오 경로 (None이면 자동 생성)
            show_video: 실시간 화면 표시 여부
            save_video: 비디오 저장 여부
            max_frames: 최대 처리 프레임 수 (None이면 전체)
            progress_callback: 진행률 콜백 함수
            
        Returns:
            처리 결과 정보
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # 출력 경로 설정
        if output_path is None and save_video:
            output_path = input_path.parent / f"tracked_{input_path.name}"
        
        # 비디오 캡처 초기화
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video: {input_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_video:
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"  Output: {output_path}")
        
        # 처리 시작
        frame_idx = 0
        start_time = time.time()
        
        try:
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # 프레임 처리
                tracks, pose_result = self.pipeline.process_frame(frame)
                
                # 트랙 ID를 포즈 결과에 할당
                if pose_result is not None:
                    pose_result = self.pipeline.assign_track_ids_to_pose_result(
                        pose_result, tracks)
                
                # 시각화
                vis_frame = self.visualizer.draw_tracks(frame, tracks, pose_result)
                
                # 정보 패널 추가
                current_fps = 1.0 / (time.time() - frame_start)
                vis_frame = self.visualizer.draw_info_panel(
                    vis_frame, frame_idx, len(tracks), current_fps)
                
                # 처리 시간 기록
                frame_time = time.time() - frame_start
                self.process_times.append(frame_time)
                
                # 비디오 저장
                if save_video and out is not None:
                    out.write(vis_frame)
                
                # 화면 표시
                if show_video:
                    cv2.imshow('RTMO Tracking', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User interrupted")
                        break
                
                # 진행률 콜백
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)
                
                frame_idx += 1
                
                # 진행 상황 출력
                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_idx / elapsed
                    print(f"Processed {frame_idx}/{total_frames} frames, "
                          f"avg FPS: {avg_fps:.1f}")
        
        finally:
            # 리소스 정리
            cap.release()
            if out is not None:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
        
        # 결과 정보
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        result = {
            'input_path': str(input_path),
            'output_path': str(output_path) if output_path else None,
            'processed_frames': frame_idx,
            'total_time': total_time,
            'average_fps': avg_fps,
            'pipeline_stats': self.pipeline.get_stats()
        }
        
        if self.process_times:
            result['frame_times'] = {
                'mean': np.mean(self.process_times),
                'std': np.std(self.process_times),
                'min': np.min(self.process_times),
                'max': np.max(self.process_times)
            }
        
        print(f"Processing completed!")
        print(f"  Processed frames: {frame_idx}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        return result
    
    def process_image_sequence(self,
                              input_dir: str,
                              output_dir: str,
                              image_pattern: str = "*.jpg") -> Dict[str, Any]:
        """
        이미지 시퀀스 처리
        
        Args:
            input_dir: 입력 이미지 디렉토리
            output_dir: 출력 이미지 디렉토리  
            image_pattern: 이미지 파일 패턴
            
        Returns:
            처리 결과 정보
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일들 찾기
        image_files = sorted(list(input_dir.glob(image_pattern)))
        
        if not image_files:
            raise ValueError(f"No images found in {input_dir} with pattern {image_pattern}")
        
        print(f"Processing {len(image_files)} images from {input_dir}")
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            # 이미지 로드
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            # 프레임 처리
            tracks, pose_result = self.pipeline.process_frame(image)
            
            # 트랙 ID 할당
            if pose_result is not None:
                pose_result = self.pipeline.assign_track_ids_to_pose_result(
                    pose_result, tracks)
            
            # 시각화
            vis_image = self.visualizer.draw_tracks(image, tracks, pose_result)
            vis_image = self.visualizer.draw_info_panel(vis_image, i, len(tracks))
            
            # 저장
            output_path = output_dir / f"tracked_{image_path.name}"
            cv2.imwrite(str(output_path), vis_image)
            
            if i % 50 == 0:
                print(f"Processed {i+1}/{len(image_files)} images")
        
        total_time = time.time() - start_time
        avg_fps = len(image_files) / total_time if total_time > 0 else 0
        
        result = {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'processed_images': len(image_files),
            'total_time': total_time,
            'average_fps': avg_fps,
            'pipeline_stats': self.pipeline.get_stats()
        }
        
        print(f"Image sequence processing completed!")
        print(f"  Processed images: {len(image_files)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        return result