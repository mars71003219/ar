"""
영역 기반 복합점수 계산기 구현

기존 rtmo_gcn_pipeline의 RegionBasedPositionScorer와 
EnhancedFightInvolvementScorer를 새로운 표준 인터페이스에 맞게 재구성한 버전입니다.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

# 이전한 scoring_system 모듈 사용
try:
    from .enhanced_scoring_system import (
        RegionBasedPositionScorer, 
        AdaptiveRegionImportance, 
        EnhancedFightInvolvementScorer
    )
    SCORING_AVAILABLE = True
except ImportError as e:
    SCORING_AVAILABLE = False
    logging.warning(f"Enhanced scoring system not available: {e}")

try:
    from scoring.base import BaseScorer, PersonScores
    from utils.data_structure import FramePoses, ScoringConfig, PersonPose
except ImportError:
    from ..base import BaseScorer, PersonScores
    from ...utils.data_structure import FramePoses, ScoringConfig, PersonPose


class RegionBasedScorer(BaseScorer):
    """영역 기반 복합점수 계산기"""
    
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
        
        # 원본 점수 계산기들
        self.position_scorer = None
        self.fight_scorer = None
        
        # 자동 초기화
        self.initialize_scorer()
    
    def initialize_scorer(self) -> bool:
        """점수 계산기 초기화"""
        try:
            if self.is_initialized:
                return True
            
            if not SCORING_AVAILABLE:
                # 원본 모듈이 없으면 기본 구현 사용
                logging.info("Using basic scoring implementation")
                self.is_initialized = True
                return True
            
            # 원본 점수 계산기 초기화
            self.position_scorer = RegionBasedPositionScorer(
                self.img_width, self.img_height
            )
            
            self.fight_scorer = EnhancedFightInvolvementScorer(img_shape=(self.img_height, self.img_width))
            
            self.is_initialized = True
            logging.info("Region-based scorer initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize region-based scorer: {str(e)}")
            self.is_initialized = False
            return False
    
    def calculate_scores(self, tracked_poses: List[FramePoses]) -> Dict[int, PersonScores]:
        """복합점수 계산"""
        if not self.is_initialized:
            raise RuntimeError("Scorer not initialized. Call initialize_scorer() first.")
        
        # 유효한 트랙 데이터 필터링
        valid_tracks = self.filter_valid_tracks(tracked_poses)
        valid_tracks = self.validate_track_data(valid_tracks)
        
        if not valid_tracks:
            return {}
        
        # 각 트랙별 점수 계산
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
                
                # 각 점수 요소 계산
                person_scores.movement_score = self.calculate_movement_score(bbox_history)
                person_scores.position_score = self.calculate_position_score(bbox_history)
                person_scores.interaction_score = self.calculate_interaction_score(
                    track_id, track_data, valid_tracks
                )
                person_scores.temporal_consistency_score = self.calculate_temporal_consistency_score(track_data)
                person_scores.persistence_score = self.calculate_persistence_score(
                    track_data, len(tracked_poses)
                )
                
                # 복합점수 계산
                person_scores.calculate_composite_score(self.weights)
                
                track_scores[track_id] = person_scores
                
            except Exception as e:
                logging.warning(f"Error calculating scores for track {track_id}: {str(e)}")
                continue
        
        # 통계 업데이트
        self.update_statistics(track_scores)
        
        return track_scores
    
    def calculate_position_score(self, bbox_history: List[List[float]]) -> float:
        """위치 점수 계산"""
        if not bbox_history:
            return 0.0
        
        if SCORING_AVAILABLE and self.position_scorer:
            try:
                # 원본 구현 사용
                position_score, region_scores = self.position_scorer.calculate_position_score(bbox_history)
                return position_score
            except Exception as e:
                logging.warning(f"Error in original position scorer: {str(e)}")
        
        # 기본 구현: 화면 중앙에 가까울수록 높은 점수
        return self._basic_position_score(bbox_history)
    
    def _basic_position_score(self, bbox_history: List[List[float]]) -> float:
        """기본 위치 점수 계산 구현"""
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
    
    def calculate_interaction_score(self, track_id: int, track_data: List[Dict[str, Any]], 
                                  all_tracks: Dict[int, List[Dict[str, Any]]]) -> float:
        """상호작용 점수 계산"""
        if len(all_tracks) <= 1:  # 다른 트랙이 없으면 상호작용 없음
            return 0.0
        
        interaction_scores = []
        
        # 시간별로 다른 트랙들과의 상호작용 계산
        for frame_data in track_data:
            frame_idx = frame_data['frame_idx']
            curr_bbox = frame_data['bbox']
            
            frame_interactions = []
            
            # 같은 프레임의 다른 트랙들과 비교
            for other_track_id, other_track_data in all_tracks.items():
                if other_track_id == track_id:
                    continue
                
                # 같은 프레임의 데이터 찾기
                other_frame_data = None
                for other_frame in other_track_data:
                    if other_frame['frame_idx'] == frame_idx:
                        other_frame_data = other_frame
                        break
                
                if other_frame_data:
                    other_bbox = other_frame_data['bbox']
                    
                    # IoU 계산
                    iou = self._calculate_iou(curr_bbox, other_bbox)
                    
                    # 거리 기반 상호작용 점수
                    distance_score = self._calculate_distance_interaction(curr_bbox, other_bbox)
                    
                    # 상호작용 점수 (IoU + 거리)
                    interaction = iou * 0.7 + distance_score * 0.3
                    frame_interactions.append(interaction)
            
            if frame_interactions:
                interaction_scores.append(max(frame_interactions))
        
        return np.mean(interaction_scores) if interaction_scores else 0.0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance_interaction(self, bbox1: List[float], bbox2: List[float]) -> float:
        """거리 기반 상호작용 점수"""
        # 바운딩 박스 중심점들 간의 거리
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # 상호작용 임계 거리 (바운딩 박스 크기 기반)
        bbox1_size = np.sqrt((bbox1[2] - bbox1[0])**2 + (bbox1[3] - bbox1[1])**2)
        bbox2_size = np.sqrt((bbox2[2] - bbox2[0])**2 + (bbox2[3] - bbox2[1])**2)
        avg_size = (bbox1_size + bbox2_size) / 2
        
        interaction_threshold = avg_size * 2.0  # 평균 크기의 2배를 임계값으로
        
        if distance > interaction_threshold:
            return 0.0
        
        # 가까울수록 높은 점수
        interaction_score = max(0.0, 1.0 - (distance / interaction_threshold))
        
        return interaction_score
    
    def get_scorer_info(self) -> Dict[str, Any]:
        """점수 계산기 정보 반환"""
        base_info = super().get_scorer_info()
        
        base_info.update({
            'scorer_type': 'region_based',
            'image_size': (self.img_width, self.img_height),
            'uses_original_implementation': SCORING_AVAILABLE,
            'position_scorer_available': self.position_scorer is not None,
            'fight_scorer_available': self.fight_scorer is not None
        })
        
        return base_info
    
    def set_image_size(self, width: int, height: int):
        """이미지 크기 설정
        
        Args:
            width: 이미지 너비
            height: 이미지 높이
        """
        if width > 0 and height > 0:
            self.img_width = width
            self.img_height = height
            
            # 위치 점수 계산기 재초기화
            if SCORING_AVAILABLE:
                try:
                    self.position_scorer = RegionBasedPositionScorer(width, height)
                except Exception as e:
                    logging.warning(f"Failed to reinitialize position scorer: {str(e)}")
    
    def score_frame_poses(self, frame_poses: FramePoses) -> FramePoses:
        """단일 프레임의 포즈에 대해 점수를 계산하여 반환
        
        파이프라인 인터페이스용 메소드
        
        Args:
            frame_poses: 점수 계산할 프레임 포즈 데이터
            
        Returns:
            점수가 계산된 프레임 포즈 데이터
        """
        if not self.is_initialized:
            if not self.initialize_scorer():
                logging.warning("Scorer not initialized, returning original frame poses")
                return frame_poses
        
        if not frame_poses.persons:
            return frame_poses
        
        # 현재 프레임에 대한 기본적인 점수 계산
        # (완전한 복합점수 계산은 multiple frames가 필요하므로 여기서는 기본 점수만 적용)
        scored_persons = []
        
        for person in frame_poses.persons:
            # 기본적인 위치 점수만 계산 (단일 프레임)
            if person.bbox and len(person.bbox) == 4:
                position_score = self._basic_position_score([person.bbox])
                
                # 메타데이터에 점수 정보 추가
                metadata = person.metadata if hasattr(person, 'metadata') and person.metadata else {}
                metadata.update({
                    'position_score': position_score,
                    'scored_by': 'region_based_scorer',
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
                scored_persons.append(scored_person)
            else:
                # bbox가 유효하지 않은 경우 원본 그대로
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
                    'scorer_type': 'region_based',
                    'scored_persons': len(scored_persons),
                    'frame_level_scoring': True
                }
            }
        )
        
        return scored_frame_poses
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        if self.position_scorer:
            self.position_scorer = None
        
        if self.fight_scorer:
            self.fight_scorer = None