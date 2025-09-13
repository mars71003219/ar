"""
쓰러짐 탐지 전용 복합점수 계산기 구현

쓰러짐 현상에 최적화된 점수 계산:
1. 높이 변화 점수 (height_change_score): 사람의 높이가 급격히 낮아지는 정도
2. 자세 각도 점수 (posture_angle_score): 몸이 수평에 가까워지는 정도  
3. 움직임 강도 점수 (movement_intensity_score): 갑작스러운 움직임 정도
4. 지속성 점수 (persistence_score): 쓰러진 자세가 지속되는 정도
5. 위치 점수 (position_score): 화면 중앙 영역에서의 발생 여부
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import math

try:
    from scoring.base import BaseScorer, PersonScores
    from utils.data_structure import FramePoses, ScoringConfig, PersonPose
except ImportError:
    from ..base import BaseScorer, PersonScores
    from ...utils.data_structure import FramePoses, ScoringConfig, PersonPose


class FalldownScorer(BaseScorer):
    """쓰러짐 탐지 전용 복합점수 계산기"""
    
    def __init__(self, config: ScoringConfig, img_width: int = 640, img_height: int = 640):
        """
        Args:
            config: 점수 계산 설정
            img_width: 이미지 너비
            img_height: 이미지 높이
        """
        super().__init__(config)
        
        self.img_width = img_width
        self.img_height = img_height
        
        # 쓰러짐 탐지 전용 가중치 설정
        self.falldown_weights = {
            'height_change': 0.35,        # 높이 변화 (가장 중요)
            'posture_angle': 0.25,        # 자세 각도  
            'movement_intensity': 0.20,   # 움직임 강도
            'persistence': 0.15,          # 지속성
            'position': 0.05              # 위치 (상대적으로 낮음)
        }
        
        # 쓰러짐 탐지 임계값들
        self.height_drop_threshold = 0.4     # 높이 감소 40% 이상
        self.angle_threshold = 45.0          # 45도 이하로 기울어지면 쓰러짐 의심
        self.movement_threshold = 50.0       # 급격한 움직임 임계값 (픽셀)
        self.min_keypoints_for_angle = 5     # 각도 계산에 필요한 최소 키포인트 수
        
        # 자동 초기화
        self.initialize_scorer()
    
    def initialize_scorer(self) -> bool:
        """점수 계산기 초기화"""
        try:
            if self.is_initialized:
                return True
            
            self.is_initialized = True
            logging.info("Falldown scorer initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize falldown scorer: {str(e)}")
            self.is_initialized = False
            return False
    
    def calculate_scores(self, tracked_poses: List[FramePoses]) -> Dict[int, PersonScores]:
        """쓰러짐 중심의 복합점수 계산"""
        if not self.is_initialized:
            raise RuntimeError("Scorer not initialized. Call initialize_scorer() first.")
        
        # 유효한 트랙 데이터 필터링
        valid_tracks = self.filter_valid_tracks(tracked_poses)
        valid_tracks = self.validate_track_data(valid_tracks)
        
        if not valid_tracks:
            return {}
        
        # 각 트랙별 쓰러짐 점수 계산
        track_scores = {}
        
        for track_id, track_data in valid_tracks.items():
            try:
                person_scores = PersonScores(track_id)
                person_scores.frame_count = len(track_data)
                
                # 바운딩 박스와 키포인트 히스토리 추출
                bbox_history = [frame['bbox'] for frame in track_data]
                keypoint_history = [frame['keypoints'] for frame in track_data]
                quality_scores = [frame['score'] for frame in track_data]
                
                person_scores.bbox_history = bbox_history
                person_scores.keypoint_history = keypoint_history
                person_scores.quality_scores = quality_scores
                
                # 쓰러짐 전용 점수 요소 계산
                person_scores.height_change_score = self.calculate_height_change_score(bbox_history)
                person_scores.posture_angle_score = self.calculate_posture_angle_score(keypoint_history)
                person_scores.movement_intensity_score = self.calculate_movement_intensity_score(bbox_history)
                person_scores.persistence_score = self.calculate_falldown_persistence_score(
                    bbox_history, keypoint_history
                )
                person_scores.position_score = self.calculate_position_score(bbox_history)
                
                # 기존 인터페이스 호환성을 위한 매핑
                person_scores.movement_score = person_scores.movement_intensity_score
                person_scores.interaction_score = 0.0  # 쓰러짐 탐지에서는 상호작용 점수 사용 안함
                person_scores.temporal_consistency_score = person_scores.persistence_score
                
                # 쓰러짐 전용 복합점수 계산
                person_scores.composite_score = self.calculate_falldown_composite_score(person_scores)
                
                track_scores[track_id] = person_scores
                
            except Exception as e:
                logging.warning(f"Error calculating falldown scores for track {track_id}: {str(e)}")
                continue
        
        # 통계 업데이트
        self.update_statistics(track_scores)
        
        return track_scores
    
    def calculate_height_change_score(self, bbox_history: List[List[float]]) -> float:
        """높이 변화 점수 계산 - 쓰러짐의 핵심 지표"""
        if len(bbox_history) < 2:
            return 0.0
        
        height_changes = []
        max_height = 0.0
        
        # 각 bbox의 높이 계산
        heights = []
        for bbox in bbox_history:
            height = bbox[3] - bbox[1]  # y2 - y1
            heights.append(height)
            max_height = max(max_height, height)
        
        if max_height == 0:
            return 0.0
        
        # 높이 감소율 계산
        for i in range(1, len(heights)):
            prev_height = heights[i-1]
            curr_height = heights[i]
            
            if prev_height > 0:
                height_change_ratio = (prev_height - curr_height) / prev_height
                height_changes.append(max(0.0, height_change_ratio))  # 감소만 고려
        
        if not height_changes:
            return 0.0
        
        # 최대 높이 감소율을 점수로 변환
        max_height_drop = max(height_changes)
        
        # 임계값 기반 점수 계산
        if max_height_drop >= self.height_drop_threshold:
            # 임계값 이상이면 높은 점수
            score = min(1.0, max_height_drop * 2.0)
        else:
            # 임계값 미만이면 낮은 점수
            score = max_height_drop / self.height_drop_threshold * 0.5
        
        return score
    
    def calculate_posture_angle_score(self, keypoint_history: List[List[List[float]]]) -> float:
        """자세 각도 점수 계산 - 몸이 수평에 가까워지는 정도"""
        if not keypoint_history:
            return 0.0
        
        angle_scores = []
        
        for keypoints in keypoint_history:
            if len(keypoints) < 17:  # COCO 17 키포인트가 없으면 건너뛰기
                continue
            
            # 유효한 키포인트만 필터링
            valid_keypoints = []
            for kpt in keypoints:
                if len(kpt) >= 3 and kpt[2] > 0.3:  # 신뢰도 임계값
                    valid_keypoints.append(kpt)
            
            if len(valid_keypoints) < self.min_keypoints_for_angle:
                continue
            
            # 몸의 기울기 각도 계산
            body_angle = self.calculate_body_inclination_angle(keypoints)
            if body_angle is not None:
                # 각도를 점수로 변환 (수평에 가까울수록 높은 점수)
                angle_score = self.angle_to_falldown_score(body_angle)
                angle_scores.append(angle_score)
        
        return np.mean(angle_scores) if angle_scores else 0.0
    
    def calculate_body_inclination_angle(self, keypoints: List[List[float]]) -> Optional[float]:
        """몸의 기울기 각도 계산"""
        try:
            # COCO 키포인트 인덱스
            # 5: left_shoulder, 6: right_shoulder
            # 11: left_hip, 12: right_hip
            
            if len(keypoints) < 17:
                return None
            
            # 어깨 중점과 골반 중점 계산
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            # 유효한 키포인트 확인
            valid_shoulders = [kpt for kpt in [left_shoulder, right_shoulder] 
                             if len(kpt) >= 3 and kpt[2] > 0.3]
            valid_hips = [kpt for kpt in [left_hip, right_hip] 
                         if len(kpt) >= 3 and kpt[2] > 0.3]
            
            if len(valid_shoulders) == 0 or len(valid_hips) == 0:
                return None
            
            # 어깨 중점
            shoulder_x = np.mean([kpt[0] for kpt in valid_shoulders])
            shoulder_y = np.mean([kpt[1] for kpt in valid_shoulders])
            
            # 골반 중점
            hip_x = np.mean([kpt[0] for kpt in valid_hips])
            hip_y = np.mean([kpt[1] for kpt in valid_hips])
            
            # 수직선과의 각도 계산
            if shoulder_x == hip_x:
                return 0.0  # 완전히 수직
            
            # 몸의 기울기 각도 (수직선 기준)
            angle_rad = math.atan2(abs(shoulder_x - hip_x), abs(shoulder_y - hip_y))
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logging.debug(f"Error calculating body angle: {e}")
            return None
    
    def angle_to_falldown_score(self, angle_deg: float) -> float:
        """각도를 쓰러짐 점수로 변환"""
        # 90도에 가까우면 (누워있음) 높은 점수
        # 0도에 가까우면 (서있음) 낮은 점수
        
        if angle_deg >= 90 - self.angle_threshold:  # 45도 이상 기울어짐
            # 수평에 가까울수록 높은 점수
            score = min(1.0, (angle_deg - (90 - self.angle_threshold)) / self.angle_threshold)
        else:
            # 서있는 자세는 낮은 점수
            score = 0.0
        
        return score
    
    def calculate_movement_intensity_score(self, bbox_history: List[List[float]]) -> float:
        """움직임 강도 점수 계산 - 갑작스러운 움직임 탐지"""
        if len(bbox_history) < 2:
            return 0.0
        
        movement_intensities = []
        
        for i in range(1, len(bbox_history)):
            prev_bbox = bbox_history[i-1]
            curr_bbox = bbox_history[i]
            
            # 바운딩박스 중심점 이동거리
            prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
            prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
            curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
            curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
            
            distance = np.sqrt((curr_center_x - prev_center_x)**2 + 
                             (curr_center_y - prev_center_y)**2)
            
            movement_intensities.append(distance)
        
        if not movement_intensities:
            return 0.0
        
        # 급격한 움직임 점수 계산
        max_movement = max(movement_intensities)
        avg_movement = np.mean(movement_intensities)
        
        # 평균보다 크게 움직이거나 임계값 이상이면 높은 점수
        if max_movement > self.movement_threshold:
            score = min(1.0, max_movement / (self.movement_threshold * 2))
        else:
            score = max_movement / self.movement_threshold * 0.5
        
        return score
    
    def calculate_falldown_persistence_score(self, bbox_history: List[List[float]], 
                                           keypoint_history: List[List[List[float]]]) -> float:
        """쓰러진 자세 지속성 점수 계산"""
        if len(bbox_history) < 3:
            return 0.0
        
        persistence_indicators = []
        
        # 각 프레임에서 쓰러짐 지표 확인
        for i in range(len(bbox_history)):
            bbox = bbox_history[i]
            keypoints = keypoint_history[i] if i < len(keypoint_history) else []
            
            falldown_indicators = 0
            total_indicators = 0
            
            # 1. 높이 기반 지표
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            if width > 0:
                aspect_ratio = height / width
                if aspect_ratio < 1.0:  # 가로가 세로보다 긴 경우
                    falldown_indicators += 1
                total_indicators += 1
            
            # 2. 각도 기반 지표
            if keypoints and len(keypoints) >= 17:
                body_angle = self.calculate_body_inclination_angle(keypoints)
                if body_angle is not None:
                    if body_angle >= 90 - self.angle_threshold:
                        falldown_indicators += 1
                    total_indicators += 1
            
            # 3. 위치 기반 지표 (화면 하단부에 위치)
            center_y = (bbox[1] + bbox[3]) / 2
            if center_y > self.img_height * 0.6:  # 화면 하단 40% 영역
                falldown_indicators += 1
            total_indicators += 1
            
            if total_indicators > 0:
                frame_persistence = falldown_indicators / total_indicators
                persistence_indicators.append(frame_persistence)
        
        if not persistence_indicators:
            return 0.0
        
        # 지속성 점수: 평균 지표 + 연속성 보너스
        avg_persistence = np.mean(persistence_indicators)
        
        # 연속성 보너스 계산
        consecutive_count = 0
        max_consecutive = 0
        for indicator in persistence_indicators:
            if indicator >= 0.5:  # 절반 이상의 지표가 쓰러짐을 나타내면
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        # 연속성 보너스 (최대 0.3)
        continuity_bonus = min(0.3, max_consecutive / len(persistence_indicators))
        
        final_score = min(1.0, avg_persistence + continuity_bonus)
        return final_score
    
    def calculate_position_score(self, bbox_history: List[List[float]]) -> float:
        """위치 점수 계산 - 화면 중앙에서의 쓰러짐 가중"""
        if not bbox_history:
            return 0.0
        
        center_x = self.img_width / 2
        center_y = self.img_height / 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        position_scores = []
        
        for bbox in bbox_history:
            # 바운딩 박스 중심점
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            # 화면 중앙으로부터의 거리
            distance = np.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
            
            # 거리를 점수로 변환 (가까울수록 높은 점수)
            score = max(0.0, 1.0 - (distance / max_distance))
            position_scores.append(score)
        
        return np.mean(position_scores) if position_scores else 0.0
    
    def calculate_falldown_composite_score(self, person_scores: PersonScores) -> float:
        """쓰러짐 전용 복합점수 계산"""
        score = 0.0
        
        # 각 점수 요소에 가중치 적용
        score += person_scores.height_change_score * self.falldown_weights['height_change']
        score += person_scores.posture_angle_score * self.falldown_weights['posture_angle']
        score += person_scores.movement_intensity_score * self.falldown_weights['movement_intensity']
        score += person_scores.persistence_score * self.falldown_weights['persistence']
        score += person_scores.position_score * self.falldown_weights['position']
        
        return min(1.0, score)
    
    def score_frame_poses(self, frame_poses: FramePoses) -> FramePoses:
        """단일 프레임의 포즈에 대해 쓰러짐 기본 점수 계산"""
        if not self.is_initialized:
            if not self.initialize_scorer():
                logging.warning("Falldown scorer not initialized, returning original frame poses")
                return frame_poses
        
        if not frame_poses.persons:
            return frame_poses
        
        scored_persons = []
        
        for person in frame_poses.persons:
            if person.bbox and len(person.bbox) == 4:
                # 단일 프레임에서 계산 가능한 기본 점수들
                position_score = self.calculate_position_score([person.bbox])
                
                # 키포인트가 있으면 각도 점수도 계산
                angle_score = 0.0
                if person.keypoints and len(person.keypoints) >= 17:
                    body_angle = self.calculate_body_inclination_angle(person.keypoints)
                    if body_angle is not None:
                        angle_score = self.angle_to_falldown_score(body_angle)
                
                # 메타데이터에 쓰러짐 점수 정보 추가
                metadata = person.metadata if hasattr(person, 'metadata') and person.metadata else {}
                metadata.update({
                    'position_score': position_score,
                    'angle_score': angle_score,
                    'falldown_indicator': angle_score > 0.5,  # 쓰러짐 의심 여부
                    'scored_by': 'falldown_scorer',
                    'frame_scoring': True
                })
                
                # PersonPose 객체 복사 및 메타데이터 업데이트
                scored_person = PersonPose(
                    person_id=person.person_id,
                    bbox=person.bbox,
                    keypoints=person.keypoints,
                    score=person.score
                )
                scored_person.track_id = getattr(person, 'track_id', None)
                scored_person.metadata = metadata
                
                # 기본 복합점수 (단일 프레임 기준)
                basic_composite = (position_score * 0.3 + angle_score * 0.7)
                scored_person.composite_score = basic_composite
                
                scored_persons.append(scored_person)
            else:
                scored_persons.append(person)
        
        # 새로운 FramePoses 생성
        scored_frame_poses = FramePoses(
            frame_idx=frame_poses.frame_idx,
            persons=scored_persons,
            timestamp=frame_poses.timestamp,
            image_shape=frame_poses.image_shape,
            metadata={
                **frame_poses.metadata,
                'scoring_info': {
                    'scorer_type': 'falldown_scorer',
                    'scored_persons': len(scored_persons),
                    'frame_level_scoring': True,
                    'falldown_focused': True
                }
            }
        )
        
        return scored_frame_poses
    
    def get_scorer_info(self) -> Dict[str, Any]:
        """점수 계산기 정보 반환"""
        base_info = super().get_scorer_info()
        
        base_info.update({
            'scorer_type': 'falldown_scorer',
            'image_size': (self.img_width, self.img_height),
            'weights': self.falldown_weights,
            'thresholds': {
                'height_drop': self.height_drop_threshold,
                'angle': self.angle_threshold,
                'movement': self.movement_threshold,
                'min_keypoints': self.min_keypoints_for_angle
            },
            'falldown_optimized': True
        })
        
        return base_info
    
    def set_image_size(self, width: int, height: int):
        """이미지 크기 설정"""
        if width > 0 and height > 0:
            self.img_width = width
            self.img_height = height
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()