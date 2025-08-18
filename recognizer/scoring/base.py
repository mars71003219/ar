"""
복합점수 계산 모듈 기본 클래스

모든 점수 계산 모델이 구현해야 하는 표준 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
try:
    from utils.data_structure import FramePoses, ScoringConfig
except ImportError:
    from ..utils.data_structure import FramePoses, ScoringConfig


class PersonScores:
    """개별 person의 점수 정보"""
    
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.movement_score = 0.0
        self.position_score = 0.0
        self.interaction_score = 0.0
        self.temporal_consistency_score = 0.0
        self.persistence_score = 0.0
        self.composite_score = 0.0
        
        # 상세 정보
        self.frame_count = 0
        self.bbox_history = []
        self.keypoint_history = []
        self.quality_scores = []
    
    def calculate_composite_score(self, weights: List[float]) -> float:
        """가중치를 적용하여 복합점수 계산"""
        if len(weights) != 5:
            weights = [0.3, 0.2, 0.1, 0.2, 0.2]  # 기본 가중치
        
        self.composite_score = (
            self.movement_score * weights[0] +
            self.position_score * weights[1] +
            self.interaction_score * weights[2] +
            self.temporal_consistency_score * weights[3] +
            self.persistence_score * weights[4]
        )
        
        return self.composite_score
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'track_id': self.track_id,
            'movement_score': self.movement_score,
            'position_score': self.position_score,
            'interaction_score': self.interaction_score,
            'temporal_consistency_score': self.temporal_consistency_score,
            'persistence_score': self.persistence_score,
            'composite_score': self.composite_score,
            'frame_count': self.frame_count
        }


class BaseScorer(ABC):
    """복합점수 계산 모듈 기본 클래스"""
    
    def __init__(self, config: ScoringConfig):
        """
        Args:
            config: 점수 계산 설정
        """
        self.config = config
        self.weights = config.get_weights_as_list()
        self.quality_threshold = config.quality_threshold
        self.min_track_length = config.min_track_length
        
        # 초기화 상태
        self.is_initialized = False
        
        # 통계
        self.stats = {
            'total_tracks_processed': 0,
            'valid_tracks': 0,
            'avg_composite_score': 0.0,
            'score_distribution': {}
        }
        
    @abstractmethod
    def initialize_scorer(self) -> bool:
        """점수 계산기 초기화
        
        Returns:
            초기화 성공 여부
        """
        pass
        
    @abstractmethod
    def calculate_scores(self, tracked_poses: List[FramePoses]) -> Dict[int, PersonScores]:
        """복합점수 계산
        
        Args:
            tracked_poses: 시간 순서별 트래킹된 포즈 데이터
            
        Returns:
            Dict[track_id, PersonScores]: 각 track_id별 점수 정보
        """
        pass
    
    def rank_persons(self, scores: Dict[int, PersonScores]) -> List[Tuple[int, float]]:
        """person 랭킹 생성
        
        Args:
            scores: track_id별 점수 딕셔너리
            
        Returns:
            (track_id, composite_score) 튜플 리스트 (높은 점수부터 정렬)
        """
        rankings = [(track_id, score_obj.composite_score) 
                   for track_id, score_obj in scores.items()]
        
        # 복합점수로 내림차순 정렬
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def filter_valid_tracks(self, tracked_poses: List[FramePoses]) -> Dict[int, List[Dict[str, Any]]]:
        """유효한 트랙들을 필터링하고 정리
        
        Args:
            tracked_poses: 트래킹된 포즈 데이터 리스트
            
        Returns:
            Dict[track_id, track_data]: 유효한 트랙별 데이터
        """
        track_data = {}
        
        # 트랙별로 데이터 수집
        for frame_poses in tracked_poses:
            for person in frame_poses.persons:
                if person.track_id is not None:
                    track_id = person.track_id
                    
                    if track_id not in track_data:
                        track_data[track_id] = []
                    
                    frame_data = {
                        'frame_idx': frame_poses.frame_idx,
                        'bbox': person.bbox,
                        'keypoints': person.keypoints,
                        'score': person.score,
                        'timestamp': frame_poses.timestamp
                    }
                    
                    track_data[track_id].append(frame_data)
        
        # 길이 기준으로 필터링
        valid_tracks = {}
        for track_id, data in track_data.items():
            if len(data) >= self.min_track_length:
                valid_tracks[track_id] = data
        
        return valid_tracks
    
    def calculate_movement_score(self, bbox_history: List[List[float]]) -> float:
        """움직임 점수 계산 (기본 구현)
        
        Args:
            bbox_history: 바운딩 박스 히스토리
            
        Returns:
            움직임 점수 (0.0 ~ 1.0)
        """
        if len(bbox_history) < 2:
            return 0.0
        
        movements = []
        for i in range(1, len(bbox_history)):
            prev_bbox = bbox_history[i-1]
            curr_bbox = bbox_history[i]
            
            # 중심점 이동 거리 계산
            prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, 
                          (prev_bbox[1] + prev_bbox[3]) / 2]
            curr_center = [(curr_bbox[0] + curr_bbox[2]) / 2,
                          (curr_bbox[1] + curr_bbox[3]) / 2]
            
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            movements.append(distance)
        
        # 평균 이동 거리를 정규화 (0~1 범위)
        avg_movement = np.mean(movements)
        
        # 이동 거리를 0~1 범위로 정규화 (최대 100픽셀 이동을 1.0으로 가정)
        normalized_movement = min(avg_movement / 100.0, 1.0)
        
        return normalized_movement
    
    def calculate_temporal_consistency_score(self, track_data: List[Dict[str, Any]]) -> float:
        """시간적 일관성 점수 계산 (기본 구현)
        
        Args:
            track_data: 트랙 데이터
            
        Returns:
            시간적 일관성 점수 (0.0 ~ 1.0)
        """
        if len(track_data) < 3:
            return 0.5  # 기본값
        
        # 프레임 간격의 일관성 확인
        frame_intervals = []
        for i in range(1, len(track_data)):
            interval = track_data[i]['frame_idx'] - track_data[i-1]['frame_idx']
            frame_intervals.append(interval)
        
        if not frame_intervals:
            return 0.5
        
        # 간격의 표준편차가 작을수록 높은 점수
        std_interval = np.std(frame_intervals)
        max_std = 5.0  # 최대 표준편차를 5로 가정
        
        consistency_score = max(0.0, 1.0 - (std_interval / max_std))
        
        return consistency_score
    
    def calculate_persistence_score(self, track_data: List[Dict[str, Any]], total_frames: int) -> float:
        """지속성 점수 계산 (기본 구현)
        
        Args:
            track_data: 트랙 데이터
            total_frames: 전체 프레임 수
            
        Returns:
            지속성 점수 (0.0 ~ 1.0)
        """
        if total_frames <= 0:
            return 0.0
        
        # 트랙이 존재하는 프레임 비율
        persistence_ratio = len(track_data) / total_frames
        
        return min(persistence_ratio, 1.0)
    
    def update_statistics(self, scores: Dict[int, PersonScores]):
        """통계 정보 업데이트"""
        if not scores:
            return
        
        self.stats['total_tracks_processed'] = len(scores)
        self.stats['valid_tracks'] = len([s for s in scores.values() 
                                         if s.frame_count >= self.min_track_length])
        
        composite_scores = [s.composite_score for s in scores.values()]
        self.stats['avg_composite_score'] = np.mean(composite_scores) if composite_scores else 0.0
        
        # 점수 분포 계산
        score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        distribution = {f"{low:.1f}-{high:.1f}": 0 for low, high in score_ranges}
        
        for score in composite_scores:
            for low, high in score_ranges:
                if low <= score < high:
                    distribution[f"{low:.1f}-{high:.1f}"] += 1
                    break
        
        self.stats['score_distribution'] = distribution
    
    def get_scorer_info(self) -> Dict[str, Any]:
        """점수 계산기 정보 반환"""
        return {
            'scorer_name': self.config.scorer_name,
            'weights': self.weights,
            'quality_threshold': self.quality_threshold,
            'min_track_length': self.min_track_length,
            'is_initialized': self.is_initialized,
            'statistics': self.stats.copy()
        }
    
    def set_weights(self, weights: List[float]):
        """가중치 설정
        
        Args:
            weights: 5개 요소의 가중치 리스트
                    [movement, position, interaction, temporal, persistence]
        """
        if len(weights) == 5 and all(isinstance(w, (int, float)) for w in weights):
            self.weights = list(weights)
            self.config.movement_weight = weights[0]
            self.config.position_weight = weights[1]
            self.config.interaction_weight = weights[2]
            self.config.temporal_consistency_weight = weights[3]
            self.config.persistence_weight = weights[4]
    
    def validate_track_data(self, track_data: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        """트랙 데이터 유효성 검사"""
        valid_tracks = {}
        
        for track_id, data in track_data.items():
            valid_data = []
            
            for frame_data in data:
                # 바운딩 박스 유효성 확인
                bbox = frame_data.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                        # 점수 유효성 확인
                        score = frame_data.get('score', 0.0)
                        if score >= self.quality_threshold:
                            valid_data.append(frame_data)
            
            if len(valid_data) >= self.min_track_length:
                valid_tracks[track_id] = valid_data
        
        return valid_tracks
    
    def __enter__(self):
        """Context manager 진입"""
        if not self.is_initialized:
            self.initialize_scorer()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        self.is_initialized = False