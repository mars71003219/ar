#!/usr/bin/env python3
"""
Visualization utilities for tracking results
트래킹 결과 시각화 유틸리티
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import colorsys
import random

try:
    from mmpose.apis import inference_bottomup
    from mmpose.structures import PoseDataSample
    from mmpose.visualization import PoseLocalVisualizer
except ImportError as e:
    print(f"MMPose import error: {e}")


class TrackingVisualizer:
    """
    트래킹 결과 시각화 클래스
    """
    
    def __init__(self, 
                 max_colors: int = 50,
                 line_thickness: int = 2,
                 font_scale: float = 0.6,
                 show_pose: bool = True,
                 show_bbox: bool = True,
                 show_track_id: bool = True):
        """
        Args:
            max_colors: 최대 색상 수 (트랙 ID용)
            line_thickness: 라인 두께
            font_scale: 폰트 크기
            show_pose: 포즈 스켈레톤 표시 여부
            show_bbox: 바운딩 박스 표시 여부
            show_track_id: 트랙 ID 표시 여부
        """
        self.max_colors = max_colors
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_pose = show_pose
        self.show_bbox = show_bbox
        self.show_track_id = show_track_id
        
        # 색상 팔레트 생성
        self.colors = self._generate_colors(max_colors)
        
        # 포즈 시각화기 초기화 (MMPose 사용)
        if self.show_pose:
            try:
                self.pose_visualizer = PoseLocalVisualizer()
            except:
                self.pose_visualizer = None
                print("Warning: Could not initialize PoseLocalVisualizer")
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        고유한 색상들 생성
        
        Args:
            num_colors: 생성할 색상 수
            
        Returns:
            BGR 색상 리스트
        """
        colors = []
        for i in range(num_colors):
            # HSV 색공간에서 균등하게 분포된 색상 생성
            hue = i / num_colors
            saturation = 0.8 + (i % 3) * 0.1  # 0.8, 0.9, 1.0
            value = 0.8 + (i % 2) * 0.2       # 0.8, 1.0
            
            # HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # RGB to BGR (OpenCV 형식)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        트랙 ID에 따른 고유 색상 반환
        
        Args:
            track_id: 트랙 ID
            
        Returns:
            BGR 색상
        """
        if track_id < 0:
            return (128, 128, 128)  # 회색 (매칭 안된 경우)
        return self.colors[track_id % len(self.colors)]
    
    def draw_tracks(self, 
                   image: np.ndarray, 
                   tracks: List,
                   pose_result: Optional[PoseDataSample] = None) -> np.ndarray:
        """
        이미지에 트랙들을 그리기
        
        Args:
            image: 입력 이미지
            tracks: 트랙 리스트
            pose_result: 포즈 결과 (선택적)
            
        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()
        
        # 포즈 스켈레톤 그리기 (트랙 ID와 함께)
        if self.show_pose and pose_result is not None:
            vis_image = self._draw_poses_with_track_ids(vis_image, pose_result, tracks)
        
        # 트랙 바운딩 박스와 ID 그리기
        for track in tracks:
            color = self.get_track_color(track.track_id)
            
            if self.show_bbox:
                self._draw_track_bbox(vis_image, track, color)
            
            if self.show_track_id:
                self._draw_track_id(vis_image, track, color)
        
        return vis_image
    
    def _draw_poses_with_track_ids(self, 
                                  image: np.ndarray,
                                  pose_result: PoseDataSample,
                                  tracks: List) -> np.ndarray:
        """
        포즈 스켈레톤을 트랙 ID 색상으로 그리기
        
        Args:
            image: 입력 이미지
            pose_result: 포즈 결과
            tracks: 트랙 리스트
            
        Returns:
            포즈가 그려진 이미지
        """
        if not hasattr(pose_result, 'pred_instances'):
            return image
        
        pred_instances = pose_result.pred_instances
        
        if (not hasattr(pred_instances, 'keypoints') or 
            len(pred_instances.keypoints) == 0):
            return image
        
        # 키포인트 추출
        if hasattr(pred_instances.keypoints, 'cpu'):
            keypoints = pred_instances.keypoints.cpu().numpy()
        else:
            keypoints = np.array(pred_instances.keypoints)
        
        # 키포인트 점수 추출
        if hasattr(pred_instances, 'keypoint_scores'):
            if hasattr(pred_instances.keypoint_scores, 'cpu'):
                keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()
            else:
                keypoint_scores = np.array(pred_instances.keypoint_scores)
        else:
            keypoint_scores = np.ones(keypoints.shape[:2])
        
        # 트랙 ID 추출
        if hasattr(pred_instances, 'track_ids'):
            if hasattr(pred_instances.track_ids, 'cpu'):
                track_ids = pred_instances.track_ids.cpu().numpy()
            else:
                track_ids = np.array(pred_instances.track_ids)
        else:
            track_ids = np.full(len(keypoints), -1)
        
        # 각 사람별로 포즈 그리기
        for i, (kpts, kpt_scores, track_id) in enumerate(zip(keypoints, keypoint_scores, track_ids)):
            color = self.get_track_color(track_id)
            self._draw_skeleton(image, kpts, kpt_scores, color)
        
        return image
    
    def _draw_skeleton(self, 
                      image: np.ndarray,
                      keypoints: np.ndarray,
                      keypoint_scores: np.ndarray,
                      color: Tuple[int, int, int],
                      score_threshold: float = 0.3):
        """
        스켈레톤 그리기
        
        Args:
            image: 이미지
            keypoints: 키포인트 좌표 (17, 2)
            keypoint_scores: 키포인트 점수 (17,)
            color: 색상
            score_threshold: 점수 임계값
        """
        # COCO-17 스켈레톤 연결 정보
        skeleton_connections = [
            (0, 1), (0, 2),    # nose to eyes
            (1, 3), (2, 4),    # eyes to ears
            (5, 6),            # shoulders
            (5, 7), (7, 9),    # left arm
            (6, 8), (8, 10),   # right arm
            (5, 11), (6, 12),  # shoulders to hips
            (11, 12),          # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16)  # right leg
        ]
        
        # 키포인트 그리기
        for i, (kpt, score) in enumerate(zip(keypoints, keypoint_scores)):
            if score > score_threshold:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(image, (x, y), 3, color, -1)
        
        # 스켈레톤 라인 그리기
        for start_idx, end_idx in skeleton_connections:
            if (start_idx < len(keypoint_scores) and end_idx < len(keypoint_scores) and
                keypoint_scores[start_idx] > score_threshold and
                keypoint_scores[end_idx] > score_threshold):
                
                start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                
                cv2.line(image, start_pt, end_pt, color, self.line_thickness)
    
    def _draw_track_bbox(self, 
                        image: np.ndarray, 
                        track,
                        color: Tuple[int, int, int]):
        """
        트랙 바운딩 박스 그리기
        
        Args:
            image: 이미지
            track: 트랙 객체
            color: 색상
        """
        bbox = track.to_xyxy()
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        
        # 바운딩 박스
        cv2.rectangle(image, pt1, pt2, color, self.line_thickness)
        
        # 점수 표시 (옵션)
        score_text = f"{track.score:.2f}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                   self.font_scale * 0.7, 1)[0]
        cv2.rectangle(image, 
                     (pt1[0], pt1[1] - text_size[1] - 5),
                     (pt1[0] + text_size[0], pt1[1]),
                     color, -1)
        cv2.putText(image, score_text,
                   (pt1[0], pt1[1] - 2),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   self.font_scale * 0.7,
                   (255, 255, 255), 1)
    
    def _draw_track_id(self, 
                      image: np.ndarray,
                      track,
                      color: Tuple[int, int, int]):
        """
        트랙 ID 그리기
        
        Args:
            image: 이미지
            track: 트랙 객체
            color: 색상
        """
        bbox = track.to_xyxy()
        pt1 = (int(bbox[0]), int(bbox[1]))
        
        # 트랙 ID 텍스트
        id_text = f"ID:{track.track_id}"
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX,
                                   self.font_scale, 2)[0]
        
        # 배경 사각형
        bg_pt1 = (pt1[0], pt1[1] - text_size[1] - 10)
        bg_pt2 = (pt1[0] + text_size[0] + 5, pt1[1])
        cv2.rectangle(image, bg_pt1, bg_pt2, color, -1)
        
        # 텍스트
        cv2.putText(image, id_text,
                   (pt1[0] + 2, pt1[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   self.font_scale,
                   (255, 255, 255), 2)
    
    def draw_info_panel(self, 
                       image: np.ndarray,
                       frame_id: int,
                       num_tracks: int,
                       fps: Optional[float] = None) -> np.ndarray:
        """
        정보 패널 그리기
        
        Args:
            image: 이미지
            frame_id: 프레임 번호
            num_tracks: 트랙 수
            fps: FPS (선택적)
            
        Returns:
            정보가 추가된 이미지
        """
        info_text = [
            f"Frame: {frame_id}",
            f"Tracks: {num_tracks}"
        ]
        
        if fps is not None:
            info_text.append(f"FPS: {fps:.1f}")
        
        # 배경 패널
        panel_height = len(info_text) * 30 + 10
        cv2.rectangle(image, (10, 10), (250, 10 + panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (250, 10 + panel_height),
                     (255, 255, 255), 2)
        
        # 정보 텍스트
        for i, text in enumerate(info_text):
            cv2.putText(image, text,
                       (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       self.font_scale,
                       (255, 255, 255), 2)
        
        return image