#!/usr/bin/env python3
"""
포즈 추정 결과 시각화 프로그램
원본 비디오와 PKL 파일을 입력받아 트래킹 결과를 오버레이하여 표시
"""

import os
import argparse
import pickle
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class PoseVisualizer:
    """포즈 추정 결과 시각화 클래스"""
    
    def __init__(self, video_path: str, pkl_path: str):
        """
        Args:
            video_path: 원본 비디오 파일 경로
            pkl_path: 포즈 추정 결과 PKL 파일 경로
        """
        self.video_path = video_path
        self.pkl_path = pkl_path
        
        # 결과 데이터 로드
        self.video_data = None
        self.pose_data = None
        self._load_data()
        
        # 색상 팔레트 (트랙 ID별 색상)
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203), # Pink
            (0, 128, 0),    # Dark Green
        ]
        
        # COCO 스켈레톤 연결 정보
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 몸통
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # 어깨-팔꿈치
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],         # 팔꿈치-손목, 머리
            [2, 4], [3, 5], [4, 6], [5, 7]                    # 목-어깨
        ]
    
    def _load_data(self):
        """비디오와 PKL 데이터 로드"""
        try:
            # 비디오 데이터 로드
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video info: {self.width}x{self.height}, {self.fps:.2f} fps, {self.total_frames} frames")
            
            # PKL 데이터 로드
            with open(self.pkl_path, 'rb') as f:
                self.pose_data = pickle.load(f)
            
            print(f"Loaded pose data with {len(self.pose_data.get('windows', []))} windows")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def visualize_interactive(self):
        """대화형 시각화 (키보드 컨트롤)"""
        print("\\n=== Interactive Pose Visualization ===")
        print("Controls:")
        print("  SPACE: Play/Pause")
        print("  LEFT/RIGHT: Previous/Next frame")
        print("  UP/DOWN: Previous/Next window")
        print("  'q': Quit")
        print("  'r': Reset to start")
        print("  's': Save current frame")
        print()
        
        current_window = 0
        current_frame_in_window = 0
        is_playing = False
        
        while True:
            # 현재 윈도우 정보
            if current_window >= len(self.pose_data.get('windows', [])):
                print("No more windows available")
                break
            
            window_data = self.pose_data['windows'][current_window]
            start_frame = window_data.get('start_frame', 0)
            end_frame = window_data.get('end_frame', start_frame + 100)
            
            # 현재 프레임 계산
            current_abs_frame = start_frame + current_frame_in_window
            
            if current_abs_frame >= end_frame:
                current_frame_in_window = 0
                current_abs_frame = start_frame
            
            # 비디오 프레임 읽기
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_abs_frame)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Cannot read frame {current_abs_frame}")
                break
            
            # 포즈 오버레이
            frame_with_poses = self._draw_poses_on_frame(
                frame, window_data, current_frame_in_window
            )
            
            # 정보 텍스트 추가
            info_text = [
                f"Window: {current_window + 1}/{len(self.pose_data['windows'])} (idx: {window_data.get('window_idx', 'N/A')})",
                f"Frame: {current_abs_frame} ({current_frame_in_window + 1}/{end_frame - start_frame})",
                f"Range: {start_frame}-{end_frame}",
            ]
            
            if 'annotation' in window_data and 'persons' in window_data['annotation']:
                persons = window_data['annotation']['persons']
                info_text.append(f"Tracked persons: {len(persons)}")
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame_with_poses, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            # 화면 표시
            cv2.imshow('Pose Visualization', frame_with_poses)
            
            # 키 입력 처리
            key = cv2.waitKey(30 if is_playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # SPACE: Play/Pause
                is_playing = not is_playing
                print(f"{'Playing' if is_playing else 'Paused'}")
            elif key == 83 or key == ord('d'):  # RIGHT arrow or 'd': Next frame
                current_frame_in_window += 1
                if current_frame_in_window >= (end_frame - start_frame):
                    current_window = min(current_window + 1, len(self.pose_data['windows']) - 1)
                    current_frame_in_window = 0
            elif key == 81 or key == ord('a'):  # LEFT arrow or 'a': Previous frame
                current_frame_in_window -= 1
                if current_frame_in_window < 0:
                    current_window = max(current_window - 1, 0)
                    if current_window < len(self.pose_data['windows']):
                        prev_window = self.pose_data['windows'][current_window]
                        prev_start = prev_window.get('start_frame', 0)
                        prev_end = prev_window.get('end_frame', prev_start + 100)
                        current_frame_in_window = max(0, prev_end - prev_start - 1)
                    else:
                        current_frame_in_window = 0
            elif key == 82 or key == ord('w'):  # UP arrow or 'w': Previous window
                current_window = max(current_window - 1, 0)
                current_frame_in_window = 0
            elif key == 84 or key == ord('s'):  # DOWN arrow or 's': Next window
                current_window = min(current_window + 1, len(self.pose_data['windows']) - 1)
                current_frame_in_window = 0
            elif key == ord('r'):  # Reset
                current_window = 0
                current_frame_in_window = 0
                is_playing = False
            elif key == ord('s'):  # Save frame
                save_path = f"frame_{current_window}_{current_abs_frame}.jpg"
                cv2.imwrite(save_path, frame_with_poses)
                print(f"Saved frame to {save_path}")
            
            # 자동 재생 시 프레임 진행
            if is_playing:
                current_frame_in_window += 1
                if current_frame_in_window >= (end_frame - start_frame):
                    current_window = (current_window + 1) % len(self.pose_data['windows'])
                    current_frame_in_window = 0
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def _draw_poses_on_frame(self, frame: np.ndarray, window_data: dict, frame_idx: int) -> np.ndarray:
        """프레임에 포즈 정보 그리기"""
        frame_copy = frame.copy()
        
        try:
            annotation = window_data.get('annotation', {})
            if 'persons' not in annotation:
                return frame_copy
            
            persons = annotation['persons']
            
            for person_id, person_data in persons.items():
                color_idx = int(person_id) % len(self.colors)
                color = self.colors[color_idx]
                
                # 해당 프레임의 키포인트 가져오기
                if 'keypoints' in person_data and frame_idx < len(person_data['keypoints']):
                    keypoints = person_data['keypoints'][frame_idx]
                    
                    if keypoints and len(keypoints) > 0:
                        # 키포인트 그리기
                        self._draw_keypoints(frame_copy, keypoints, color)
                        
                        # 스켈레톤 그리기
                        self._draw_skeleton(frame_copy, keypoints, color)
                
                # 바운딩 박스 그리기 (있는 경우)
                if 'bboxes' in person_data and frame_idx < len(person_data['bboxes']):
                    bbox = person_data['bboxes'][frame_idx]
                    if bbox and len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                        
                        # 트랙 ID 표시
                        cv2.putText(frame_copy, f"ID: {person_id}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 복합점수 표시 (있는 경우)
                if 'composite_scores' in person_data:
                    scores = person_data['composite_scores']
                    if isinstance(scores, dict):
                        score_text = f"Score: {scores.get('total', 0.0):.2f}"
                        text_pos = (50, 50 + int(person_id) * 25)
                        cv2.putText(frame_copy, score_text, text_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        except Exception as e:
            print(f"Error drawing poses: {str(e)}")
        
        return frame_copy
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: list, color: tuple):
        """키포인트 그리기"""
        if not keypoints or len(keypoints) == 0:
            return
        
        # keypoints 형식: [[x1, y1, conf1], [x2, y2, conf2], ...]
        for kpt in keypoints:
            if len(kpt) >= 3 and kpt[2] > 0.3:  # confidence threshold
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 3, color, -1)
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: list, color: tuple):
        """스켈레톤 연결선 그리기"""
        if not keypoints or len(keypoints) < 17:
            return
        
        for connection in self.skeleton:
            if len(connection) >= 2:
                idx1, idx2 = connection[0] - 1, connection[1] - 1  # 1-based to 0-based
                
                if (0 <= idx1 < len(keypoints) and 0 <= idx2 < len(keypoints) and
                    len(keypoints[idx1]) >= 3 and len(keypoints[idx2]) >= 3 and
                    keypoints[idx1][2] > 0.3 and keypoints[idx2][2] > 0.3):
                    
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    cv2.line(frame, pt1, pt2, color, 2)
    
    def save_video_with_overlay(self, output_path: str, window_indices: List[int] = None):
        """오버레이된 비디오 저장"""
        print(f"Saving overlayed video to: {output_path}")
        
        if window_indices is None:
            window_indices = list(range(len(self.pose_data.get('windows', []))))
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        try:
            frames_written = 0
            
            for window_idx in window_indices:
                if window_idx >= len(self.pose_data['windows']):
                    continue
                
                window_data = self.pose_data['windows'][window_idx]
                start_frame = window_data.get('start_frame', 0)
                end_frame = window_data.get('end_frame', start_frame + 100)
                
                print(f"Processing window {window_idx}: frames {start_frame}-{end_frame}")
                
                for frame_idx in range(end_frame - start_frame):
                    abs_frame = start_frame + frame_idx
                    
                    # 프레임 읽기
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        continue
                    
                    # 포즈 오버레이
                    frame_with_poses = self._draw_poses_on_frame(frame, window_data, frame_idx)
                    
                    # 윈도우 정보 텍스트 추가
                    info_text = f"Window {window_idx}, Frame {abs_frame}"
                    cv2.putText(frame_with_poses, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # 비디오에 쓰기
                    out.write(frame_with_poses)
                    frames_written += 1
            
            print(f"Video saved successfully! {frames_written} frames written.")
            
        except Exception as e:
            print(f"Error saving video: {str(e)}")
        finally:
            out.release()
    
    def print_summary(self):
        """포즈 데이터 요약 정보 출력"""
        print("\\n=== Pose Data Summary ===")
        print(f"Video: {self.video_path}")
        print(f"PKL: {self.pkl_path}")
        print(f"Video info: {self.width}x{self.height}, {self.fps:.2f} fps, {self.total_frames} frames")
        
        if self.pose_data:
            windows = self.pose_data.get('windows', [])
            print(f"Windows: {len(windows)}")
            
            if 'tracking_settings' in self.pose_data:
                settings = self.pose_data['tracking_settings']
                print(f"Tracking settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
            
            # 각 윈도우 정보
            for i, window in enumerate(windows[:5]):  # 처음 5개만 표시
                annotation = window.get('annotation', {})
                persons = annotation.get('persons', {})
                start_frame = window.get('start_frame', 0)
                end_frame = window.get('end_frame', 0)
                
                print(f"  Window {i}: frames {start_frame}-{end_frame}, {len(persons)} persons")
            
            if len(windows) > 5:
                print(f"  ... and {len(windows) - 5} more windows")


def main():
    parser = argparse.ArgumentParser(description='Pose Estimation Result Visualizer')
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to the original video file')
    parser.add_argument('--pkl', type=str, required=True,
                       help='Path to the pose estimation result PKL file')
    parser.add_argument('--mode', type=str, choices=['interactive', 'save', 'summary'], 
                       default='interactive',
                       help='Visualization mode')
    parser.add_argument('--output', type=str, default='output_with_poses.mp4',
                       help='Output video path (for save mode)')
    parser.add_argument('--windows', type=str, default=None,
                       help='Comma-separated window indices to process (e.g., "0,1,2")')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.pkl):
        print(f"Error: PKL file not found: {args.pkl}")
        return
    
    # 시각화기 초기화
    try:
        visualizer = PoseVisualizer(args.video, args.pkl)
    except Exception as e:
        print(f"Error initializing visualizer: {str(e)}")
        return
    
    # 윈도우 인덱스 파싱
    window_indices = None
    if args.windows:
        try:
            window_indices = [int(x.strip()) for x in args.windows.split(',')]
        except ValueError:
            print(f"Error: Invalid window indices format: {args.windows}")
            return
    
    # 모드별 실행
    try:
        if args.mode == 'interactive':
            visualizer.visualize_interactive()
        elif args.mode == 'save':
            visualizer.save_video_with_overlay(args.output, window_indices)
        elif args.mode == 'summary':
            visualizer.print_summary()
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
    
    print("Visualization completed.")


if __name__ == "__main__":
    main()