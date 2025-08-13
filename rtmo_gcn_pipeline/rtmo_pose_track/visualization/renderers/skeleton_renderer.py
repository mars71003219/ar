#!/usr/bin/env python3
"""
Skeleton Renderer - 스켈레톤 렌더링 모듈
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    from configs.visualizer_config import config as default_config
except ImportError:
    try:
        from ..configs.visualizer_config import config as default_config
    except ImportError:
        print("Warning: Could not import visualizer config for skeleton renderer. Using defaults.")
        default_config = None


class SkeletonRenderer:
    """스켈레톤 렌더링 클래스"""
    
    def __init__(self, config=None):
        # 설정 로드
        self.config = config or default_config
        
        if self.config:
            # 설정 파일에서 값 로드
            self.skeleton = self.config.skeleton_connections
            self.skeleton_colors = self.config.skeleton_colors
            self.keypoint_color = self.config.keypoint_color
            self.line_thickness = self.config.line_thickness
            self.keypoint_radius = self.config.keypoint_radius
            self.confidence_threshold = self.config.confidence_threshold
        else:
            # 폴백 설정
            self.skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 다리
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],          # 목, 어깨
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],          # 팔, 얼굴
                [2, 4], [3, 5], [4, 6], [5, 7]                     # 얼굴에서 몸으로
            ]
            self.skeleton_colors = [(0, 255, 255)] * len(self.skeleton)
            self.keypoint_color = (0, 0, 255)
            self.line_thickness = 2
            self.keypoint_radius = 3
            self.confidence_threshold = 0.3
    
    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                     color: Optional[Tuple[int, int, int]] = None, 
                     stage: str = None) -> np.ndarray:
        """
        스켈레톤을 이미지에 그리기
        
        Args:
            image: 대상 이미지
            keypoints: 키포인트 배열 (17, 3) - [x, y, confidence]
            color: 기본 색상 (옵션)
        
        Returns:
            스켈레톤이 그려진 이미지
        """
        if keypoints is None:
            return image
            
        result_image = image.copy()
        
        try:
            # keypoints 형태 정규화
            if isinstance(keypoints, list):
                keypoints = np.array(keypoints)
            
            # 차원 확인
            if keypoints.ndim == 1 and len(keypoints) >= 51:
                # 평면화된 경우 (51,) -> (17, 3)
                keypoints = keypoints.reshape(17, 3)
            elif keypoints.ndim == 2:
                # 이미 올바른 형태
                if keypoints.shape[0] != 17:
                    print(f"Warning: Expected 17 keypoints, got {keypoints.shape[0]}")
                    return result_image
            else:
                print(f"Warning: Unsupported keypoints shape: {keypoints.shape}")
                return result_image
            
            # 스켈레톤 연결선 그리기
            for i, (start_idx, end_idx) in enumerate(self.skeleton):
                # 1-based 인덱스를 0-based로 변환
                start_idx -= 1
                end_idx -= 1
                
                if (0 <= start_idx < len(keypoints) and 
                    0 <= end_idx < len(keypoints)):
                    
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]
                    
                    # 신뢰도 임계값 조정 (stage2는 더 관대하게)
                    threshold = self.confidence_threshold
                    if stage and stage in ['stage2', 'step2']:
                        threshold = 0.1  # tracking 데이터는 더 낮은 임계값 사용
                    
                    # 신뢰도 확인 (3차원인 경우)
                    if (len(start_point) >= 3 and len(end_point) >= 3 and
                        start_point[2] > threshold and 
                        end_point[2] > threshold and
                        start_point[0] > 0 and start_point[1] > 0 and
                        end_point[0] > 0 and end_point[1] > 0):
                        
                        start_pos = (int(start_point[0]), int(start_point[1]))
                        end_pos = (int(end_point[0]), int(end_point[1]))
                        
                        line_color = color if color else self.skeleton_colors[i]
                        
                        cv2.line(result_image, start_pos, end_pos, 
                                line_color, self.line_thickness)
                    elif (len(start_point) >= 2 and len(end_point) >= 2 and
                          start_point[0] > 0 and start_point[1] > 0 and
                          end_point[0] > 0 and end_point[1] > 0):
                        # 2차원인 경우 좌표값만 확인
                        start_pos = (int(start_point[0]), int(start_point[1]))
                        end_pos = (int(end_point[0]), int(end_point[1]))
                        
                        line_color = color if color else self.skeleton_colors[i]
                        
                        cv2.line(result_image, start_pos, end_pos, 
                                line_color, self.line_thickness)
            
            # 키포인트 그리기
            for i, point in enumerate(keypoints):
                if len(point) >= 2:
                    # 신뢰도 임계값 조정 (stage2는 더 관대하게)
                    threshold = self.confidence_threshold
                    if stage and stage in ['stage2', 'step2']:
                        threshold = 0.1
                    
                    # 신뢰도 확인 (3차원인 경우) 및 좌표값 확인
                    if ((len(point) >= 3 and point[2] > threshold) or len(point) == 2) and point[0] > 0 and point[1] > 0:
                        pos = (int(point[0]), int(point[1]))
                        point_color = color if color else self.keypoint_color
                        cv2.circle(result_image, pos, self.keypoint_radius, 
                                  point_color, -1)
            
            return result_image
            
        except Exception as e:
            print(f"Error drawing skeleton: {e}")
            return result_image
    
    def draw_multiple_skeletons(self, image: np.ndarray, 
                               persons_data: List[dict],
                               colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        여러 사람의 스켈레톤 그리기
        
        Args:
            image: 대상 이미지
            persons_data: 사람별 데이터 리스트
            colors: 사람별 색상 리스트 (옵션)
        
        Returns:
            모든 스켈레톤이 그려진 이미지
        """
        result_image = image.copy()
        
        for i, person_data in enumerate(persons_data):
            color = None
            if colors and i < len(colors):
                color = colors[i]
            
            if 'keypoints' in person_data:
                result_image = self.draw_skeleton(result_image, 
                                                person_data['keypoints'], color)
        
        return result_image
