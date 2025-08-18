"""
베이스 시각화 클래스
추론 모드별 시각화의 공통 기능을 제공
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseInferenceVisualizer(ABC):
    """추론 시각화 베이스 클래스 - 공통 기능 제공"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_persons = config.get('models', {}).get('action_classification', {}).get('max_persons', 4)
        
        # COCO 17 키포인트 연결 구조 (0-based index)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
        
        # 색상 설정
        self.colors = {
            'fight': (0, 0, 255),           # 빨강
            'normal': (0, 255, 0),          # 초록
            'keypoint_face': (255, 255, 0), # 노란색 (얼굴)
            'keypoint_arm': (0, 255, 0),    # 초록색 (팔)
            'keypoint_leg': (255, 0, 0),    # 파란색 (다리)
            'top_person': (0, 0, 255),      # 빨간색 (max_persons 이내)
            'other_person': (255, 0, 0),    # 파란색 (나머지)
            'text': (255, 255, 255)         # 흰색
        }
        
        # 폰트 크기 기본값
        self.base_font_scale = 1.0
        self.base_thickness = 2
    
    def get_dynamic_font_params(self, frame_height: int, frame_width: int) -> Tuple[float, int]:
        """비디오 해상도에 따른 동적 폰트 크기 계산"""
        # 기준 해상도: 720p (1280x720)
        base_height = 720
        scale_factor = frame_height / base_height
        
        font_scale = max(0.5, self.base_font_scale * scale_factor)
        thickness = max(1, int(self.base_thickness * scale_factor))
        
        return font_scale, thickness
    
    def draw_classification_result(self, frame: np.ndarray, classification_data: Dict, 
                                 frame_idx: int, mode_specific_info: Dict = None) -> np.ndarray:
        """분류 결과 오버레이 (공통 기능)"""
        height, width = frame.shape[:2]
        font_scale, thickness = self.get_dynamic_font_params(height, width)
        
        if classification_data:
            # 분류 결과 표시
            predicted_class = classification_data.get('predicted_label', 
                             classification_data.get('predicted_class', 'Unknown'))
            confidence = classification_data.get('confidence', 0.0)
            
            color = self.colors['fight'] if predicted_class == 'Fight' else self.colors['normal']
            text = f"{predicted_class} ({confidence:.2f})"
            
            # 동적 폰트 크기 적용
            cv2.putText(frame, text, (int(10 * width/1280), int(30 * height/720)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # 모드별 추가 정보 표시
            if mode_specific_info:
                self._draw_mode_specific_info(frame, mode_specific_info, height, width, font_scale, thickness)
        
        return frame
    
    def draw_poses(self, frame: np.ndarray, poses_data, height: int, width: int) -> np.ndarray:
        """포즈 데이터 오버레이 (공통 기능)"""
        if not poses_data:
            return frame
        
        persons_list = self._extract_persons_list(poses_data)
        
        if persons_list:
            # confidence 기준으로 정렬 (내림차순)
            sorted_persons = sorted(persons_list, 
                                  key=lambda p: p.score if hasattr(p, 'score') and p.score else 0.0, 
                                  reverse=True)
            
            for idx, person_pose in enumerate(sorted_persons):
                # max_persons 이내: 빨간색, 나머지: 파란색
                is_top_ranked = idx < self.max_persons
                self._draw_person_pose(frame, person_pose, is_top_ranked, height, width)
        
        return frame
    
    def draw_frame_info(self, frame: np.ndarray, frame_idx: int, additional_info: Dict = None) -> np.ndarray:
        """프레임 정보 표시 (공통 기능)"""
        height, width = frame.shape[:2]
        font_scale, thickness = self.get_dynamic_font_params(height, width)
        
        # 프레임 번호 표시
        cv2.putText(frame, f"Frame: {frame_idx}", 
                   (int(10 * width/1280), height - int(10 * height/720)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, self.colors['text'], thickness)
        
        # 추가 정보 표시
        if additional_info:
            y_offset = height - int(40 * height/720)
            for key, value in additional_info.items():
                text = f"{key}: {value}"
                cv2.putText(frame, text, (int(10 * width/1280), y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, self.colors['text'], thickness)
                y_offset -= int(25 * height/720)
        
        return frame
    
    def _extract_persons_list(self, poses_data):
        """포즈 데이터에서 persons 리스트 추출"""
        persons_list = []
        
        # FramePoses 객체인지 확인
        if hasattr(poses_data, 'persons'):
            persons_list = poses_data.persons
        # person_poses 속성이 있는 경우 (이전 구조)
        elif hasattr(poses_data, 'person_poses'):
            persons_list = poses_data.person_poses
        # 딕셔너리 형태인 경우
        elif isinstance(poses_data, dict) and 'persons' in poses_data:
            persons_list = poses_data['persons']
        
        return persons_list
    
    def _draw_person_pose(self, frame: np.ndarray, person_pose, is_top_ranked: bool, height: int, width: int):
        """개별 사람의 포즈 그리기"""
        if not hasattr(person_pose, 'keypoints'):
            return
        
        # 동적 크기 계산
        font_scale, thickness = self.get_dynamic_font_params(height, width)
        keypoint_radius = max(2, int(3 * height / 720))
        bbox_thickness = max(1, int(2 * height / 720))
        skeleton_thickness = max(1, int(2 * height / 720))
        
        # max_persons 이내: 빨간색, 나머지: 파란색
        person_color = self.colors['top_person'] if is_top_ranked else self.colors['other_person']
        
        # 바운딩 박스와 트래킹 ID
        if hasattr(person_pose, 'bbox') and person_pose.bbox:
            x1, y1, x2, y2 = map(int, person_pose.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, bbox_thickness)
            
            # 텍스트 정보
            text_lines = []
            if hasattr(person_pose, 'track_id') and person_pose.track_id is not None:
                text_lines.append(f"ID: {person_pose.track_id}")
            if hasattr(person_pose, 'score') and person_pose.score is not None:
                text_lines.append(f"Conf: {person_pose.score:.2f}")
            
            if text_lines:
                text = " | ".join(text_lines)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, thickness)[0]
                
                # 텍스트 배경
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), 
                             (x1 + text_size[0] + 5, y1), person_color, -1)
                
                # 텍스트
                cv2.putText(frame, text, (x1 + 2, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), thickness)
        
        # 키포인트와 스켈레톤
        self._draw_keypoints_and_skeleton(frame, person_pose.keypoints, person_color, 
                                        keypoint_radius, skeleton_thickness)
    
    def _draw_keypoints_and_skeleton(self, frame: np.ndarray, keypoints, person_color: Tuple[int, int, int],
                                   keypoint_radius: int, skeleton_thickness: int):
        """키포인트와 스켈레톤 그리기"""
        if keypoints is None:
            return
        
        # 키포인트가 numpy array가 아닌 경우 변환
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # 키포인트 그리기 (부위별 색상 구분)
        for i, kpt in enumerate(keypoints):
            if len(kpt) >= 2 and kpt[0] > 0 and kpt[1] > 0:  # 유효한 키포인트만
                # 키포인트별 색상 구분
                if i < 5:  # 얼굴 (코, 눈, 귀)
                    kpt_color = self.colors['keypoint_face']  # 노란색
                elif i < 11:  # 팔 (어깨, 팔꿈치, 손목)
                    kpt_color = self.colors['keypoint_arm']   # 초록색
                else:  # 다리 (엉덩이, 무릎, 발목)
                    kpt_color = self.colors['keypoint_leg']   # 파란색
                
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), keypoint_radius, kpt_color, -1)
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), keypoint_radius + 1, person_color, 1)
        
        # 스켈레톤 연결선 그리기
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx][:2]
                pt2 = keypoints[pt2_idx][:2]
                
                # 두 점이 모두 유효한 경우만 선 그리기
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), person_color, skeleton_thickness)
    
    @abstractmethod
    def _draw_mode_specific_info(self, frame: np.ndarray, mode_info: Dict, 
                               height: int, width: int, font_scale: float, thickness: int):
        """모드별 특화 정보 표시 (추상 메서드)"""
        pass
    
    @abstractmethod
    def visualize_frame(self, frame: np.ndarray, frame_data: Dict, frame_idx: int) -> np.ndarray:
        """프레임 시각화 (추상 메서드)"""
        pass