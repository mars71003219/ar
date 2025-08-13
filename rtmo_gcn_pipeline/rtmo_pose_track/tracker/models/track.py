#!/usr/bin/env python3
"""
Enhanced Track class for multi-object tracking
MMTracking 기반으로 향상된 Track 구현
"""

import numpy as np
from enum import Enum
from typing import Optional, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kalman_filter import EnhancedKalmanFilter
from utils.bbox_utils import bbox_xyxy_to_cxcyah


class TrackState(Enum):
    """트랙 상태"""
    NEW = 1           # 새로운 트랙 (아직 확정되지 않음)
    TRACKED = 2       # 활성 추적 중
    LOST = 3          # 일시적으로 잃어버림
    REMOVED = 4       # 완전히 제거됨


class Track:
    """
    Enhanced Track class for ByteTracker
    MMTracking의 BaseTrack을 참고하여 구현
    """
    
    count = 0
    
    def __init__(self, bbox: np.ndarray, score: float, label: int = 0):
        """
        Args:
            bbox: [x1, y1, x2, y2] 형태의 바운딩 박스
            score: 검출 신뢰도 점수
            label: 클래스 레이블 (기본값 0)
        """
        # 기본 정보
        self.track_id = Track.count
        Track.count += 1
        
        self.bbox = np.array(bbox)
        self.score = score
        self.label = label
        
        # 상태 관리
        self.state = TrackState.NEW
        self.is_activated = False
        
        # 프레임 및 시간 정보
        self.frame_id = -1
        self.start_frame = -1
        self.tracklet_len = 0
        
        # ByteTracker 호환성
        self.time_since_update = 0
        self.hit_streak = 1
        self.hits = 1
        self.age = 1
        
        # 칼만 필터 초기화
        self.kalman_filter = EnhancedKalmanFilter()
        
        # [cx, cy, aspect_ratio, height] 형태로 변환
        measurement = bbox_xyxy_to_cxcyah(bbox.reshape(1, -1))[0]
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)
        
        # 히스토리 저장용
        self.history = {
            'bboxes': [bbox.copy()],
            'scores': [score],
            'frame_ids': [],
            'states': [self.state]
        }
    
    @staticmethod
    def next_id():
        """다음 ID 반환"""
        Track.count += 1
        return Track.count
    
    @staticmethod
    def reset_count():
        """ID 카운터 리셋"""
        Track.count = 0
    
    def predict(self):
        """칼만 필터 예측 단계"""
        # lost 상태에서는 속도를 0으로 설정
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0  # height velocity를 0으로
        
        self.mean, self.covariance = self.kalman_filter.predict(
            self.mean, self.covariance)
        
        # 예측된 위치를 bbox로 업데이트
        from ..utils.bbox_utils import bbox_cxcyah_to_xyxy
        predicted_bbox = bbox_cxcyah_to_xyxy(self.mean[:4].reshape(1, -1))[0]
        self.bbox = predicted_bbox
        
        # 시간 관련 카운터 업데이트
        self.age += 1
        self.time_since_update += 1
        
        # hit_streak 리셋 (매칭이 안된 상태)
        if self.time_since_update > 0:
            self.hit_streak = 0
    
    def update(self, bbox: np.ndarray, score: float):
        """트랙 업데이트 (매칭되었을 때)"""
        self.bbox = np.array(bbox)
        self.score = score
        
        # 칼만 필터 업데이트
        measurement = bbox_xyxy_to_cxcyah(bbox.reshape(1, -1))[0]
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement)
        
        # 상태 업데이트
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.tracklet_len += 1
        
        # 상태 전환 로직
        if self.state == TrackState.NEW and self.hits >= 3:  # tentative에서 확정
            self.activate(self.frame_id)
        elif self.state == TrackState.LOST:
            self.re_activate(bbox, score, self.frame_id)
        
        # 히스토리 업데이트
        self.history['bboxes'].append(bbox.copy())
        self.history['scores'].append(score)
        self.history['frame_ids'].append(self.frame_id)
        self.history['states'].append(self.state)
    
    def activate(self, frame_id: int):
        """트랙 활성화"""
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        if self.start_frame == -1:
            self.start_frame = frame_id
        self.tracklet_len = 0
    
    def re_activate(self, bbox: np.ndarray, score: float, frame_id: int):
        """잃어버린 트랙 재활성화"""
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        
        # 칼만 필터 재초기화
        measurement = bbox_xyxy_to_cxcyah(bbox.reshape(1, -1))[0]
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)
    
    def mark_lost(self):
        """트랙을 잃어버림 상태로 표시"""
        self.state = TrackState.LOST
        self.is_activated = False
    
    def mark_removed(self):
        """트랙을 제거됨 상태로 표시"""
        self.state = TrackState.REMOVED
        self.is_activated = False
    
    def to_xyxy(self) -> np.ndarray:
        """현재 바운딩 박스를 [x1, y1, x2, y2] 형태로 반환"""
        return self.bbox.copy()
    
    def to_cxcyah(self) -> np.ndarray:
        """현재 상태를 [cx, cy, aspect_ratio, height] 형태로 반환"""
        return self.mean[:4].copy()
    
    def get_state_dict(self) -> dict:
        """트랙 상태를 딕셔너리로 반환"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox.tolist(),
            'score': self.score,
            'label': self.label,
            'state': self.state.name,
            'is_activated': self.is_activated,
            'frame_id': self.frame_id,
            'start_frame': self.start_frame,
            'tracklet_len': self.tracklet_len,
            'time_since_update': self.time_since_update,
            'hit_streak': self.hit_streak,
            'hits': self.hits,
            'age': self.age
        }
    
    def __len__(self):
        """트랙 길이 (히스토리 개수)"""
        return len(self.history['bboxes'])
    
    def __str__(self):
        return f"Track(id={self.track_id}, state={self.state.name}, bbox={self.bbox}, score={self.score:.3f})"
    
    def __repr__(self):
        return self.__str__()