#!/usr/bin/env python3
"""
Fight-Prioritized Tracking System with ByteTrack
Fight-우선 트래킹 시스템 - ByteTrack 기반 안정적 트래킹 ID 관리
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class KalmanFilter:
    """2D 바운딩 박스 트래킹을 위한 간단한 칼만 필터"""
    
    def __init__(self):
        # 상태: [center_x, center_y, width, height, dx, dy, dw, dh]
        self.state = np.zeros(8)
        
        # 상태 전이 행렬
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
        ])
        
        # 관측 행렬
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # 공분산 행렬
        self.P = np.eye(8) * 1000
        
        # 노이즈 행렬
        self.Q = np.eye(8)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 1.0
        
        self.R = np.eye(4) * 10.0
        self.initialized = False
    
    def init_state(self, bbox):
        """바운딩 박스로 상태 초기화"""
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        self.state[:4] = [center_x, center_y, width, height]
        self.state[4:] = 0.0
        self.initialized = True
    
    def predict(self):
        """예측 단계"""
        if not self.initialized:
            return self.get_bbox()
            
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.get_bbox()
    
    def update(self, bbox):
        """업데이트 단계"""
        if not self.initialized:
            self.init_state(bbox)
            return self.get_bbox()
        
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        z = np.array([center_x, center_y, width, height])
        
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self):
        """현재 상태에서 바운딩 박스 반환"""
        if not self.initialized:
            return np.array([0, 0, 0, 0])
            
        center_x, center_y, width, height = self.state[:4]
        
        width = max(width, 1.0)
        height = max(height, 1.0)
        
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        
        return np.array([x1, y1, x2, y2])


@dataclass
class Track:
    """트랙 객체 (칼만 필터 포함)"""
    track_id: int
    bbox: np.ndarray
    score: float
    keypoints: Optional[np.ndarray] = None
    keypoint_scores: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    composite_score: float = 0.0
    
    def __post_init__(self):
        """칼만 필터 초기화"""
        self.kalman = KalmanFilter()
        self.kalman.init_state(self.bbox)
    
    def update(self, bbox: np.ndarray, score: float, keypoints: Optional[np.ndarray] = None, 
              keypoint_scores: Optional[np.ndarray] = None):
        """트랙 업데이트"""
        self.bbox = self.kalman.update(bbox)
        self.score = score
        if keypoints is not None:
            self.keypoints = keypoints
        if keypoint_scores is not None:
            self.keypoint_scores = keypoint_scores
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
    
    def predict(self):
        """예측 단계"""
        self.bbox = self.kalman.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1


class ByteTracker:
    """ByteTrack 알고리즘 구현"""
    
    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.1, 
                 max_disappeared: int = 30, min_hits: int = 3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        
        self.tracks: List[Track] = []
        self.next_id = 0
        
    def update(self, detections: np.ndarray, keypoints_list: Optional[List[np.ndarray]] = None, 
              scores_list: Optional[List[np.ndarray]] = None) -> List[Track]:
        """트래킹 업데이트"""
        for track in self.tracks:
            track.predict()
        
        high_dets = detections[detections[:, 4] >= self.high_thresh]
        low_dets = detections[(detections[:, 4] >= self.low_thresh) & 
                             (detections[:, 4] < self.high_thresh)]
        
        # 키포인트 데이터 분리
        high_keypoints = None
        high_scores = None
        if keypoints_list is not None and scores_list is not None:
            high_indices = np.where(detections[:, 4] >= self.high_thresh)[0]
            high_keypoints = [keypoints_list[i] for i in high_indices] if len(high_indices) > 0 else []
            high_scores = [scores_list[i] for i in high_indices] if len(high_indices) > 0 else []
        
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(
            self.tracks, high_dets, iou_threshold=0.5)
        
        for track_idx, det_idx in matched_tracks:
            kpts = high_keypoints[det_idx] if high_keypoints else None
            kpt_scores = high_scores[det_idx] if high_scores else None
            self.tracks[track_idx].update(high_dets[det_idx, :4], high_dets[det_idx, 4], kpts, kpt_scores)
        
        unmatched_tracks_for_low = [self.tracks[i] for i in unmatched_tracks 
                                   if self.tracks[i].time_since_update == 1]
        
        if len(unmatched_tracks_for_low) > 0 and len(low_dets) > 0:
            matched_tracks_low, unmatched_dets_low, unmatched_tracks_low = self._associate(
                unmatched_tracks_for_low, low_dets, iou_threshold=0.5)
            
            for track_idx, det_idx in matched_tracks_low:
                track = unmatched_tracks_for_low[track_idx]
                track.update(low_dets[det_idx, :4], low_dets[det_idx, 4])
        
        for det_idx in unmatched_dets:
            kpts = high_keypoints[det_idx] if high_keypoints else None
            kpt_scores = high_scores[det_idx] if high_scores else None
            new_track = Track(
                track_id=self.next_id,
                bbox=high_dets[det_idx, :4],
                score=high_dets[det_idx, 4],
                keypoints=kpts,
                keypoint_scores=kpt_scores
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update < self.max_disappeared]
        
        active_tracks = [track for track in self.tracks 
                        if track.hits >= self.min_hits or track.time_since_update < 1]
        
        return active_tracks

    def _associate(self, tracks: List[Track], detections: np.ndarray, 
                  iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """IoU 기반 연관"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(track.bbox, det[:4])
        
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= iou_threshold:
                matched_tracks.append((row, col))
                unmatched_dets.remove(col)
                unmatched_tracks.remove(row)
        
        return matched_tracks, unmatched_dets, unmatched_tracks

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0


class FightPrioritizedTracker:
    """
    Fight-우선 인물 트래킹 시스템
    enhanced_rtmo_bytetrack_pose_extraction.py의 5영역 분할 기반 복합 점수 시스템
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480, 
                 region_weights: Optional[Dict[str, float]] = None,
                 composite_weights: Optional[Dict[str, float]] = None,
                 bytetrack_config: Optional[Dict[str, float]] = None):
        """
        초기화 - ByteTrack 기반 안정적 트래킹 ID 관리
        
        Args:
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            region_weights: 영역별 가중치
            composite_weights: 복합 점수 가중치
            bytetrack_config: ByteTrack 설정
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # 기본 영역 가중치 (enhanced_rtmo_bytetrack_pose_extraction.py 기반)
        self.region_weights = region_weights or {
            'center': 1.0,         # 중앙 영역 (가장 중요)
            'top_left': 0.7,       # 좌상단
            'top_right': 0.7,      # 우상단
            'bottom_left': 0.6,    # 좌하단
            'bottom_right': 0.6    # 우하단
        }
        
        # 복합 점수 가중치
        self.composite_weights = composite_weights or {
            'position': 0.3,       # 위치 점수 (30%)
            'movement': 0.25,      # 움직임 점수 (25%)
            'interaction': 0.25,   # 상호작용 점수 (25%)
            'detection': 0.1,      # 검출 신뢰도 (10%)
            'consistency': 0.1     # 시간적 일관성 (10%)
        }
        
        # ByteTracker 초기화
        bytetrack_config = bytetrack_config or {}
        self.byte_tracker = ByteTracker(
            high_thresh=bytetrack_config.get('high_thresh', 0.6),
            low_thresh=bytetrack_config.get('low_thresh', 0.1),
            max_disappeared=bytetrack_config.get('max_disappeared', 30),
            min_hits=bytetrack_config.get('min_hits', 3)
        )
        
        # 5영역 정의
        self.regions = self._define_regions()
        
        # 트래킹 히스토리 (track_id 기반)
        self.track_history = defaultdict(lambda: {
            'composite_scores': deque(maxlen=10),
            'positions': deque(maxlen=10),
            'keypoints_history': deque(maxlen=5),
            'last_seen': 0,
            'consistency_score': 0.0,
            'total_frames': 0
        })
        
        # 현재 활성 트랙들
        self.active_tracks: List[Track] = []
        self.frame_count = 0
        
        logger.info(f"Fight-우선 트래커 초기화 (ByteTrack 통합): {frame_width}x{frame_height}")
    
    def _define_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        5영역 정의 - 전체 4분할 + 중앙 집중
        enhanced_rtmo_bytetrack_pose_extraction.py 방식
        """
        w, h = self.frame_width, self.frame_height
        
        return {
            # 전체 4분할 (완전한 공간 커버리지)
            'top_left': (0, 0, w//2, h//2),              # 좌상단 영역
            'top_right': (w//2, 0, w, h//2),             # 우상단 영역  
            'bottom_left': (0, h//2, w//2, h),           # 좌하단 영역
            'bottom_right': (w//2, h//2, w, h),          # 우하단 영역
            
            # 중앙 집중 영역 (가장 중요 - 싸움이 주로 발생하는 구역)
            'center': (w//4, h//4, 3*w//4, 3*h//4)       # 중앙 50% 영역
        }
    
    def _calculate_position_score(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        인물의 영역별 위치 점수 계산
        
        Args:
            keypoints: 키포인트 좌표 (17, 2), (1, 17, 2) 또는 (N, 2) 단일 점
            
        Returns:
            영역별 위치 점수
        """
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]  # (1, 17, 2) -> (17, 2)
        
        # 유효한 키포인트의 중심점 계산
        if keypoints.shape[1] == 2:  # 일반적인 키포인트 데이터
            valid_points = keypoints[keypoints[:, 0] > 0]
        else:  # 단일 점 데이터 (bbox center 등)
            valid_points = keypoints
        
        if len(valid_points) == 0:
            return {region: 0.0 for region in self.regions}
        
        center_x = np.mean(valid_points[:, 0])
        center_y = np.mean(valid_points[:, 1])
        
        region_scores = {}
        for region_name, (x1, y1, x2, y2) in self.regions.items():
            # 중심점이 영역 내에 있는지 확인
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                # 영역 중앙에서의 거리에 따른 점수 (0.5 ~ 1.0)
                region_center_x = (x1 + x2) / 2
                region_center_y = (y1 + y2) / 2
                distance = np.sqrt((center_x - region_center_x)**2 + (center_y - region_center_y)**2)
                max_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
                score = max(0.5, 1.0 - (distance / max_distance) * 0.5)
                region_scores[region_name] = score * self.region_weights[region_name]
            else:
                region_scores[region_name] = 0.0
        
        return region_scores
    
    def _calculate_movement_score(self, track: Track) -> float:
        """
        동작의 격렬함 점수 계산 (ByteTrack 기반)
        
        Args:
            track: ByteTrack Track 객체
            
        Returns:
            움직임 점수 (0.0 ~ 1.0)
        """
        track_id = track.track_id
        
        if track_id not in self.track_history:
            return 0.5
        
        history = self.track_history[track_id]
        if len(history['positions']) < 2:
            return 0.5
        
        # 최근 위치 변화량 계산
        if track.keypoints is not None:
            keypoints = track.keypoints
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]
            
            valid_points = keypoints[keypoints[:, 0] > 0]
            if len(valid_points) > 0:
                current_pos = np.mean(valid_points, axis=0)
            else:
                # 키포인트가 없으면 바운딩 박스 중심 사용
                bbox = track.bbox
                current_pos = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        else:
            # 키포인트가 없으면 바운딩 박스 중심 사용
            bbox = track.bbox
            current_pos = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        
        prev_pos = history['positions'][-1]
        movement = np.linalg.norm(current_pos - prev_pos)
        
        # 정규화 (0.0 ~ 1.0) - 50픽셀 이동을 최대값으로 설정
        movement_score = min(1.0, movement / 50.0)
        
        return movement_score
    
    def _calculate_interaction_score(self, tracks: List[Track]) -> Dict[int, float]:
        """
        인물 간 상호작용 점수 계산 (ByteTrack 기반)
        
        Args:
            tracks: ByteTrack Track 객체 리스트
            
        Returns:
            Track ID별 상호작용 점수 딕셔너리
        """
        if len(tracks) < 2:
            return {track.track_id: 0.5 for track in tracks}
        
        interaction_scores = {}
        
        for i, track_i in enumerate(tracks):
            max_interaction = 0.0
            
            # track_i의 중심점 계산
            if track_i.keypoints is not None:
                keypoints_i = track_i.keypoints
                if len(keypoints_i.shape) == 3:
                    keypoints_i = keypoints_i[0]
                valid_i = keypoints_i[keypoints_i[:, 0] > 0]
                if len(valid_i) > 0:
                    center_i = np.mean(valid_i, axis=0)
                else:
                    bbox_i = track_i.bbox
                    center_i = np.array([(bbox_i[0] + bbox_i[2])/2, (bbox_i[1] + bbox_i[3])/2])
            else:
                bbox_i = track_i.bbox
                center_i = np.array([(bbox_i[0] + bbox_i[2])/2, (bbox_i[1] + bbox_i[3])/2])
            
            for j, track_j in enumerate(tracks):
                if i == j:
                    continue
                
                # track_j의 중심점 계산
                if track_j.keypoints is not None:
                    keypoints_j = track_j.keypoints
                    if len(keypoints_j.shape) == 3:
                        keypoints_j = keypoints_j[0]
                    valid_j = keypoints_j[keypoints_j[:, 0] > 0]
                    if len(valid_j) > 0:
                        center_j = np.mean(valid_j, axis=0)
                    else:
                        bbox_j = track_j.bbox
                        center_j = np.array([(bbox_j[0] + bbox_j[2])/2, (bbox_j[1] + bbox_j[3])/2])
                else:
                    bbox_j = track_j.bbox
                    center_j = np.array([(bbox_j[0] + bbox_j[2])/2, (bbox_j[1] + bbox_j[3])/2])                
                
                distance = np.linalg.norm(center_i - center_j)
                
                # 가까울수록 높은 상호작용 점수 (100픽셀 이내에서 최대값)
                if distance > 0:
                    interaction = max(0.0, 1.0 - (distance / 100.0))
                    max_interaction = max(max_interaction, interaction)
            
            interaction_scores[track_i.track_id] = max_interaction
        
        return interaction_scores
    
    def update_with_detections(self, detections: np.ndarray, keypoints_list: List[np.ndarray], 
                              scores_list: List[np.ndarray]) -> List[Track]:
        """
        검출 결과로 ByteTracker 업데이트 및 복합 점수 계산
        
        Args:
            detections: Detection 결과 (N, 5) [x1, y1, x2, y2, score]
            keypoints_list: 키포인트 리스트
            scores_list: 키포인트 신뢰도 리스트
            
        Returns:
            복합 점수가 계산된 활성 트랙 리스트
        """
        self.frame_count += 1
        
        # ByteTracker 업데이트
        self.active_tracks = self.byte_tracker.update(detections, keypoints_list, scores_list)
        
        # 복합 점수 계산
        if self.active_tracks:
            self._calculate_composite_scores(self.active_tracks)
        
        return self.active_tracks
    
    def _calculate_composite_scores(self, tracks: List[Track]) -> None:
        """
        복합 점수 계산 - 싸움 관련 Track들을 최상위로 정렬 (ByteTrack 기반)
        
        Args:
            tracks: ByteTrack Track 객체 리스트
        """
        if not tracks:
            return
        
        # 상호작용 점수 계산
        interaction_scores = self._calculate_interaction_score(tracks)
        
        for track in tracks:
            track_id = track.track_id
            
            # 1. 위치 점수 (영역별 가중치 적용)
            if track.keypoints is not None:
                position_scores = self._calculate_position_score(track.keypoints)
                position_score = max(position_scores.values())
            else:
                # 키포인트가 없으면 바운딩 박스 중심으로 계산
                bbox = track.bbox
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # 중심점을 키포인트 형태로 변환하여 계산
                fake_keypoints = np.array([[center_x, center_y]])
                position_scores = self._calculate_position_score(fake_keypoints)
                position_score = max(position_scores.values())
            
            # 2. 움직임 점수
            movement_score = self._calculate_movement_score(track)
            
            # 3. 상호작용 점수
            interaction_score = interaction_scores[track_id]
            
            # 4. 검출 신뢰도 점수
            if track.keypoint_scores is not None:
                scores = track.keypoint_scores
                detection_score = np.mean(scores[scores > 0]) if len(scores[scores > 0]) > 0 else 0.0
            else:
                detection_score = track.score  # 바운딩 박스 신뢰도 사용
            
            # 5. 시간적 일관성 점수
            consistency_score = self.track_history[track_id]['consistency_score']
            
            # 복합 점수 계산 (가중 평균)
            composite_score = (
                position_score * self.composite_weights['position'] +
                movement_score * self.composite_weights['movement'] +
                interaction_score * self.composite_weights['interaction'] +
                detection_score * self.composite_weights['detection'] +
                consistency_score * self.composite_weights['consistency']
            )
            
            # Track 객체에 복합 점수 저장
            track.composite_score = composite_score
            
            # 히스토리 업데이트
            self.track_history[track_id]['composite_scores'].append(composite_score)
            self.track_history[track_id]['last_seen'] = self.frame_count
            self.track_history[track_id]['total_frames'] += 1
            
            # 현재 위치 저장
            if track.keypoints is not None:
                keypoints = track.keypoints
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]
                valid_points = keypoints[keypoints[:, 0] > 0]
                if len(valid_points) > 0:
                    current_pos = np.mean(valid_points, axis=0)
                else:
                    bbox = track.bbox
                    current_pos = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            else:
                bbox = track.bbox
                current_pos = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            
            self.track_history[track_id]['positions'].append(current_pos)
            
            # 키포인트 히스토리 저장
            if track.keypoints is not None:
                self.track_history[track_id]['keypoints_history'].append(track.keypoints.copy())
            
            # 일관성 점수 업데이트 (최근 점수들의 표준편차 역수)
            if len(self.track_history[track_id]['composite_scores']) >= 3:
                recent_scores = list(self.track_history[track_id]['composite_scores'])[-3:]
                std_dev = np.std(recent_scores)
                self.track_history[track_id]['consistency_score'] = 1.0 / (1.0 + std_dev)
    
    def get_fight_prioritized_order(self) -> List[int]:
        """
        싸움 우선 정렬된 Track ID 리스트 반환
        
        Returns:
            복합 점수 기준 내림차순 정렬된 Track ID 리스트
        """
        if not self.active_tracks:
            return []
        
        # 복합 점수 기준 내림차순 정렬
        sorted_tracks = sorted(self.active_tracks, key=lambda track: track.composite_score, reverse=True)
        
        return [track.track_id for track in sorted_tracks]
    
    def select_fight_prioritized_people(self, num_person: int = 2) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Fight-우선 내림차순 정렬된 상위 N명 반환 (ByteTrack 기반)
        
        Args:
            num_person: 선택할 인물 수
            
        Returns:
            (정렬된 키포인트 리스트, 정렬된 점수 리스트, 정렬된 Track ID 리스트)
        """
        if not self.active_tracks:
            return [], [], []
        
        # 복합 점수 기준 내림차순 정렬
        sorted_tracks = sorted(self.active_tracks, key=lambda track: track.composite_score, reverse=True)
        
        # 상위 num_person명 선택
        selected_tracks = sorted_tracks[:num_person]
        
        sorted_keypoints = []
        sorted_scores = []
        sorted_track_ids = []
        
        for track in selected_tracks:
            if track.keypoints is not None:
                sorted_keypoints.append(track.keypoints)
            else:
                # 키포인트가 없으면 빈 데이터
                sorted_keypoints.append(np.zeros((17, 2)))
            
            if track.keypoint_scores is not None:
                sorted_scores.append(track.keypoint_scores)
            else:
                # 키포인트 신뢰도가 없으면 바운딩 박스 신뢰도를 모든 키포인트에 적용
                sorted_scores.append(np.full(17, track.score))
            
            sorted_track_ids.append(track.track_id)
        
        return sorted_keypoints, sorted_scores, sorted_track_ids
    
    def select_top_person(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        가장 높은 복합 점수를 가진 인물 선택 (하위 호환성 유지)
        
        Returns:
            (선택된 키포인트, 선택된 점수) 또는 (None, None)
        """
        sorted_keypoints, sorted_scores, _ = self.select_fight_prioritized_people(num_person=1)
        
        if sorted_keypoints:
            return sorted_keypoints[0], sorted_scores[0]
        
        return None, None
    
    def process_windowed_sequence_with_detections(self, detections_list: List[np.ndarray], 
                                                 keypoints_list: List[List[np.ndarray]], 
                                                 scores_list: List[List[np.ndarray]], 
                                                 window_size: int = 30, stride: int = 15, 
                                                 num_person: int = 2) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        윈도우 단위로 Fight-우선 인물 선택 (ByteTrack 기반, STGCN++ 윈도우와 일치)
        
        Args:
            detections_list: 프레임별 detection 결과 리스트
            keypoints_list: 프레임별 키포인트 리스트
            scores_list: 프레임별 신뢰도 리스트
            window_size: 윈도우 크기 (30프레임)
            stride: 윈도우 간격 (15프레임)
            num_person: 선택할 인물 수
            
        Returns:
            (selected_keypoints, selected_scores, track_ids_per_frame) - (T, num_person, 17, 2), (T, num_person, 17), [[track_id, ...]]
        """
        if not detections_list:
            return np.array([]), np.array([]), []
        
        final_keypoints = []
        final_scores = []
        final_track_ids = []
        
        # 각 프레임에 대해 ByteTracker 업데이트 및 복합 점수 계산
        for frame_idx, (detections, keypoints, scores) in enumerate(zip(detections_list, keypoints_list, scores_list)):
            if len(detections) > 0:
                # ByteTracker 업데이트
                active_tracks = self.update_with_detections(detections, keypoints, scores)
                
                # 상위 num_person명 선택
                selected_keypoints, selected_scores, selected_track_ids = self.select_fight_prioritized_people(num_person)
                
                # 부족한 인물은 제로 패딩
                while len(selected_keypoints) < num_person:
                    selected_keypoints.append(np.zeros((17, 2)))
                    selected_scores.append(np.zeros(17))
                    selected_track_ids.append(-1)  # 유효하지 않은 track_id
                
                frame_keypoints = np.array(selected_keypoints[:num_person])
                frame_scores = np.array(selected_scores[:num_person])
                frame_track_ids = selected_track_ids[:num_person]
            else:
                # 빈 프레임
                frame_keypoints = np.zeros((num_person, 17, 2))
                frame_scores = np.zeros((num_person, 17))
                frame_track_ids = [-1] * num_person
            
            final_keypoints.append(frame_keypoints)
            final_scores.append(frame_scores)
            final_track_ids.append(frame_track_ids)
        
        return np.array(final_keypoints), np.array(final_scores), final_track_ids
    
    def process_video_sequence_with_detections(self, detections_list: List[np.ndarray], 
                                              keypoints_list: List[List[np.ndarray]], 
                                              scores_list: List[List[np.ndarray]], 
                                              sequence_length: int = 30, num_person: int = 1) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        비디오 시퀀스에서 Fight-우선 인물 선택 (ByteTrack 기반, 설정 기반 윈도우 처리)
        
        Args:
            detections_list: 프레임별 detection 결과 리스트
            keypoints_list: 프레임별 키포인트 리스트
            scores_list: 프레임별 신뢰도 리스트
            sequence_length: 윈도우 크기 (None=기본30프레임, >=10=해당크기 윈도우 사용)
            num_person: 선택할 인물 수
            
        Returns:
            (selected_keypoints, selected_scores, track_ids_per_frame) - (T, num_person, 17, 2), (T, num_person, 17), [[track_id, ...]]
        """
        # sequence_length가 None이면 윈도우 방식 사용 (기본값 30프레임)
        if sequence_length is None:
            return self.process_windowed_sequence_with_detections(
                detections_list, keypoints_list, scores_list,
                window_size=30, 
                stride=15, 
                num_person=num_person
            )
        
        # sequence_length가 설정되어 있으면 해당 크기로 윈도우 방식 사용
        elif sequence_length >= 10:  # 최소 윈도우 크기 체크
            return self.process_windowed_sequence_with_detections(
                detections_list, keypoints_list, scores_list,
                window_size=sequence_length,
                stride=sequence_length // 2,  # 50% 오버랩
                num_person=num_person
            )
        
        # 기존 방식 (프레임별 처리) - 하위 호환성
        selected_keypoints = []
        selected_scores = []
        track_ids_per_frame = []
        
        for detections, keypoints, scores in zip(detections_list, keypoints_list, scores_list):
            frame_keypoints = []
            frame_scores = []
            frame_track_ids = []
            
            if len(detections) > 0:
                # ByteTracker 업데이트
                active_tracks = self.update_with_detections(detections, keypoints, scores)
                
                # Fight-우선 내림차순 정렬로 모든 인물 가져오기
                sorted_keypoints, sorted_scores, sorted_track_ids = self.select_fight_prioritized_people(num_person)
                
                # 상위 num_person명 선택
                for i in range(num_person):
                    if i < len(sorted_keypoints):
                        person_keypoints = sorted_keypoints[i]
                        person_scores = sorted_scores[i]
                        person_track_id = sorted_track_ids[i]
                        
                        # 차원 정규화
                        if len(person_keypoints.shape) == 3:
                            person_keypoints = person_keypoints[0]  # (1, 17, 2) -> (17, 2)
                        if len(person_scores.shape) == 2:
                            person_scores = person_scores[0]  # (1, 17) -> (17,)
                        
                        frame_keypoints.append(person_keypoints)
                        frame_scores.append(person_scores)
                        frame_track_ids.append(person_track_id)
                    else:
                        # 부족한 인물은 빈 데이터로 패딩
                        frame_keypoints.append(np.zeros((17, 2)))
                        frame_scores.append(np.zeros(17))
                        frame_track_ids.append(-1)
            else:
                # 전체 프레임이 비어있는 경우
                for i in range(num_person):
                    frame_keypoints.append(np.zeros((17, 2)))
                    frame_scores.append(np.zeros(17))
                    frame_track_ids.append(-1)
            
            selected_keypoints.append(np.array(frame_keypoints))  # (num_person, 17, 2)
            selected_scores.append(np.array(frame_scores))        # (num_person, 17)
            track_ids_per_frame.append(frame_track_ids)
        
        # 시퀀스 길이 조정
        if len(selected_keypoints) < sequence_length:
            # 패딩 (마지막 프레임 반복)
            needed = sequence_length - len(selected_keypoints)
            if selected_keypoints:
                last_kpts = selected_keypoints[-1]
                last_scores = selected_scores[-1]
                last_track_ids = track_ids_per_frame[-1]
                for _ in range(needed):
                    selected_keypoints.append(last_kpts.copy())
                    selected_scores.append(last_scores.copy())
                    track_ids_per_frame.append(last_track_ids.copy())
            else:
                # 완전히 빈 시퀀스
                for _ in range(sequence_length):
                    empty_frame_kpts = np.zeros((num_person, 17, 2))
                    empty_frame_scores = np.zeros((num_person, 17))
                    empty_track_ids = [-1] * num_person
                    selected_keypoints.append(empty_frame_kpts)
                    selected_scores.append(empty_frame_scores)
                    track_ids_per_frame.append(empty_track_ids)
        else:
            # 최신 프레임들만 사용
            selected_keypoints = selected_keypoints[-sequence_length:]
            selected_scores = selected_scores[-sequence_length:]
            track_ids_per_frame = track_ids_per_frame[-sequence_length:]
        
        return np.array(selected_keypoints), np.array(selected_scores), track_ids_per_frame  # (T, num_person, 17, 2), (T, num_person, 17), [[track_id, ...]]
    
    def get_fight_prioritized_ranking(self) -> List[Dict]:
        """
        Fight-우선 순위와 복합 점수를 포함한 상세 랭킹 반환 (ByteTrack 기반)
        
        Returns:
            랭킹 정보 리스트 (순위, Track ID, 복합 점수 포함)
        """
        if not self.active_tracks:
            return []
        
        # 복합 점수 기준 내림차순 정렬
        sorted_tracks = sorted(self.active_tracks, key=lambda track: track.composite_score, reverse=True)
        
        ranking = []
        for rank, track in enumerate(sorted_tracks):
            # 키포인트 신뢰도 계산
            if track.keypoint_scores is not None:
                valid_scores = track.keypoint_scores[track.keypoint_scores > 0]
                avg_confidence = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
            else:
                avg_confidence = track.score
            
            ranking.append({
                'rank': rank + 1,
                'track_id': track.track_id,
                'composite_score': track.composite_score,
                'bbox': track.bbox.tolist(),
                'avg_confidence': avg_confidence,
                'hits': track.hits,
                'age': track.age,
                'keypoints_available': track.keypoints is not None
            })
        
        return ranking
    
    def create_detections_from_pose_results(self, keypoints_list: List[np.ndarray], 
                                           scores_list: List[np.ndarray]) -> np.ndarray:
        """
        포즈 결과에서 detection 형태로 변환 (하위 호환성)
        
        Args:
            keypoints_list: 키포인트 리스트
            scores_list: 신뢰도 리스트
            
        Returns:
            Detection 배열 (N, 5) [x1, y1, x2, y2, score]
        """
        if not keypoints_list or not scores_list:
            return np.empty((0, 5))
        
        detections = []
        
        for keypoints, scores in zip(keypoints_list, scores_list):
            if len(keypoints.shape) == 3:
                keypoints = keypoints[0]
            if len(scores.shape) == 2:
                scores = scores[0]
            
            # 유효한 키포인트로 바운딩 박스 계산
            valid_points = keypoints[scores > 0.3]  # 신뢰도 임계값
            if len(valid_points) > 0:
                x_min, y_min = np.min(valid_points, axis=0)
                x_max, y_max = np.max(valid_points, axis=0)
                
                # 바운딩 박스 확장 (10% 마진)
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - width * 0.1)
                y_min = max(0, y_min - height * 0.1)
                x_max = x_max + width * 0.1
                y_max = y_max + height * 0.1
                
                # 키포인트 평균 신뢰도를 detection 점수로 사용
                detection_score = np.mean(scores[scores > 0])
                
                detections.append([x_min, y_min, x_max, y_max, detection_score])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def reset(self):
        """트래커 상태 초기화"""
        self.track_history.clear()
        self.active_tracks.clear()
        self.byte_tracker = ByteTracker(
            high_thresh=0.6,
            low_thresh=0.1,
            max_disappeared=30,
            min_hits=3
        )
        self.frame_count = 0
        logger.info("Fight-우선 트래커 초기화 완료 (ByteTrack 리셋)")