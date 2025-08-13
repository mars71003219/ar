#!/usr/bin/env python3
"""
ByteTracker implementation and related tracking functions
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment


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
        
        obs = np.array([
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]
        ])
        
        # Kalman 필터 업데이트
        y = obs - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self):
        """바운딩 박스 형태로 변환"""
        if not self.initialized:
            return np.array([0, 0, 0, 0])
            
        center_x, center_y, width, height = self.state[:4]
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        return np.array([x1, y1, x2, y2])


@dataclass
class Track:
    """ByteTracker용 트랙 정보"""
    track_id: int
    bbox: np.ndarray
    score: float
    age: int = 0
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    state: str = "new"  # "new", "tracked", "lost", "removed"
    
    def __post_init__(self):
        """칼만 필터 초기화"""
        self.kalman = KalmanFilter()
        self.kalman.init_state(self.bbox)
    
    def update(self, bbox: np.ndarray, score: float):
        """트랙 업데이트 - 매칭될 때 호출"""
        self.bbox = self.kalman.update(bbox)
        self.score = score
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # 상태 업데이트
        if self.state == "new" and self.hits >= 3:  # min_hits 조건
            self.state = "tracked"
        elif self.state == "lost":
            self.state = "tracked"  # 재발견
    
    def predict(self):
        """예측 단계 - 매 프레임마다 호출"""
        self.bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        
        # 상태 업데이트
        if self.time_since_update == 1:
            if self.state == "tracked":
                self.state = "lost"
        
        # hit_streak 리셋 (매칭 안됨)
        if self.time_since_update > 0:
            self.hit_streak = 0
    
    def to_bbox(self):
        """바운딩 박스 좌표 반환"""
        return self.bbox[:4] if len(self.bbox) >= 4 else self.bbox
    
    def is_activated(self) -> bool:
        """활성 상태 확인 (출력 가능한 트랙)"""
        return self.state == "tracked" or (self.state == "new" and self.hits >= 3)


class ByteTracker:
    """정확한 ByteTrack 알고리즘 구현
    
    원본 논문: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
    핵심 아이디어: High/Low confidence detection 2단계 연관
    """
    
    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.1, 
                 max_disappeared: int = 30, min_hits: int = 3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh  
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        
        # 트랙 관리
        self.tracked_tracks: List[Track] = []  # 활성 트랙들
        self.lost_tracks: List[Track] = []     # 잃어버린 트랙들  
        self.removed_tracks: List[Track] = []  # 제거된 트랙들
        
        self.next_id = 0
        self.frame_id = 0
        
    def update(self, detections: np.ndarray) -> List[Track]:
        """ByteTracker 메인 업데이트 로직"""
        self.frame_id += 1
        
        # 입력 검증
        if len(detections) == 0:
            # 검출이 없을 때도 기존 트랙들의 predict 수행
            for track in self.tracked_tracks + self.lost_tracks:
                track.predict()
            
            # 트랙 상태 관리
            self._manage_tracks()
            return self._get_active_tracks()
        
        # 1단계: 모든 트랙에 대해 예측 수행
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()
        
        # 2단계: Detection을 high/low로 분리
        high_dets = detections[detections[:, 4] >= self.high_thresh]
        low_dets = detections[(detections[:, 4] >= self.low_thresh) & 
                             (detections[:, 4] < self.high_thresh)]
        
        # 3단계: High confidence detection과 tracked tracks 연관
        matched_pairs, unmatched_dets, unmatched_tracks = self._associate(
            self.tracked_tracks, high_dets, iou_threshold=0.5)
        
        # 4단계: 매칭된 트랙들 업데이트
        for track_idx, det_idx in matched_pairs:
            track = self.tracked_tracks[track_idx]
            track.update(high_dets[det_idx, :4], high_dets[det_idx, 4])
        
        # 5단계: 매칭되지 않은 tracked tracks를 lost_tracks로 이동
        unmatched_tracked_tracks = [self.tracked_tracks[i] for i in unmatched_tracks]
        
        # 6단계: Low confidence detection과 lost tracks 연관
        # 중요: 최근에 lost된 트랙들만 대상 (time_since_update == 1)
        recent_lost_tracks = [track for track in unmatched_tracked_tracks 
                             if track.time_since_update == 1]
        
        matched_pairs_low = []
        unmatched_dets_low = list(range(len(low_dets)))
        unmatched_lost_tracks = recent_lost_tracks.copy()
        
        if len(recent_lost_tracks) > 0 and len(low_dets) > 0:
            matched_pairs_low, unmatched_dets_low, _ = self._associate(
                recent_lost_tracks, low_dets, iou_threshold=0.5)
            
            # Low detection으로 매칭된 트랙들 업데이트
            for track_idx, det_idx in matched_pairs_low:
                track = recent_lost_tracks[track_idx]
                track.update(low_dets[det_idx, :4], low_dets[det_idx, 4])
                unmatched_lost_tracks.remove(track)
        
        # 7단계: 매칭되지 않은 high confidence detection으로 새 트랙 생성
        for det_idx in unmatched_dets:
            if high_dets[det_idx, 4] >= self.high_thresh:  # 추가 검증
                new_track = Track(
                    track_id=self.next_id,
                    bbox=high_dets[det_idx, :4],
                    score=high_dets[det_idx, 4],
                    state="new"
                )
                self.tracked_tracks.append(new_track)
                self.next_id += 1
        
        # 8단계: 트랙 상태 재구성
        # 매칭된 트랙들은 tracked_tracks에 유지
        # 매칭되지 않은 트랙들은 lost_tracks로 이동
        self.lost_tracks.extend(unmatched_lost_tracks)
        
        # tracked_tracks에서 매칭되지 않은 트랙들 제거
        self.tracked_tracks = [track for i, track in enumerate(self.tracked_tracks) 
                              if i not in unmatched_tracks or track in 
                              [recent_lost_tracks[j] for j, _ in matched_pairs_low]]
        
        # 9단계: 트랙 정리 (제거 조건 확인)
        self._manage_tracks()
        
        # 10단계: 활성 트랙 반환
        return self._get_active_tracks()
        
    def _associate(self, tracks: List[Track], detections: np.ndarray, 
                  iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """헝가리안 알고리즘 기반 트랙-검출 연관
        
        Returns:
            matched_pairs: [(track_idx, det_idx), ...]
            unmatched_detections: [det_idx, ...]
            unmatched_tracks: [track_idx, ...]
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # IoU 매트릭스 계산
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(track.bbox, det[:4])
        
        # 헝가리안 알고리즘으로 최적 매칭 찾기 (비용 최소화)
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        # 매칭 결과 정리
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # IoU 임계값을 만족하는 매칭만 유효하다고 판단
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= iou_threshold:
                matched_pairs.append((row, col))
                unmatched_dets.remove(col)
                unmatched_tracks.remove(row)
        
        return matched_pairs, unmatched_dets, unmatched_tracks
    
    def _manage_tracks(self):
        """트랙 상태 관리 및 정리"""
        # lost_tracks에서 제거할 트랙들 찾기
        tracks_to_remove = []
        
        for track in self.lost_tracks:
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track)
        
        # 제거 대상 트랙들을 removed_tracks로 이동
        for track in tracks_to_remove:
            self.lost_tracks.remove(track)
            track.state = "removed"
            self.removed_tracks.append(track)
        
        # tracked_tracks에서도 너무 오래된 트랙 제거 (안전장치)
        tracks_to_remove = []
        for track in self.tracked_tracks:
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.tracked_tracks.remove(track)
            track.state = "removed"
            self.removed_tracks.append(track)
    
    def _get_active_tracks(self) -> List[Track]:
        """출력할 활성 트랙들 반환"""
        active_tracks = []
        
        for track in self.tracked_tracks:
            # 충분한 hit을 가지거나 최근에 매칭된 트랙만 출력
            if track.hits >= self.min_hits or track.time_since_update < 1:
                active_tracks.append(track)
        
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


def create_detection_results(pose_result):
    """포즈 추정 결과에서 detection 형태로 변환"""
    # MMPose 버전에 따른 호환성 처리
    if hasattr(pose_result, '_pred_instances'):
        pred_instances = pose_result._pred_instances
    elif hasattr(pose_result, 'pred_instances'):
        pred_instances = pose_result.pred_instances
    else:
        return np.empty((0, 5))
    
    if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
        return np.empty((0, 5))
    
    bboxes = pred_instances.bboxes.cpu().numpy() if hasattr(pred_instances.bboxes, 'cpu') else pred_instances.bboxes
    
    # 점수 속성 확인 (bbox_scores 또는 scores)
    if hasattr(pred_instances, 'bbox_scores'):
        scores = pred_instances.bbox_scores.cpu().numpy() if hasattr(pred_instances.bbox_scores, 'cpu') else pred_instances.bbox_scores
        scores = scores.reshape(-1, 1)
    elif hasattr(pred_instances, 'scores'):
        scores = pred_instances.scores.cpu().numpy() if hasattr(pred_instances.scores, 'cpu') else pred_instances.scores
        scores = scores.reshape(-1, 1)
    else:
        scores = np.ones((len(bboxes), 1))
    
    detections = np.concatenate([bboxes, scores], axis=1)
    return detections


def assign_track_ids_from_bytetrack(pose_result, active_tracks, iou_threshold=0.5):
    """ByteTrack 결과를 기반으로 pose_result에 track_id 할당"""
    # MMPose 버전에 따른 호환성 처리
    if hasattr(pose_result, '_pred_instances'):
        frame_result = pose_result._pred_instances
    elif hasattr(pose_result, 'pred_instances'):
        frame_result = pose_result.pred_instances
    else:
        return pose_result
        
    if not hasattr(frame_result, 'bboxes') or len(frame_result.bboxes) == 0:
        frame_result.track_ids = np.array([])
        return pose_result
    
    # tensor와 numpy 배열 구분 처리
    if hasattr(frame_result.bboxes, 'cpu'):
        frame_bboxes = frame_result.bboxes.cpu().numpy()
    else:
        frame_bboxes = frame_result.bboxes
    
    track_ids = np.full(len(frame_bboxes), -1, dtype=int)
    
    for i, frame_bbox in enumerate(frame_bboxes):
        best_iou = 0
        best_track_id = -1
        
        for track in active_tracks:
            track_bbox = track.bbox
            iou = _compute_iou_simple(frame_bbox, track_bbox)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_track_id = track.track_id
        
        track_ids[i] = best_track_id
    
    frame_result.track_ids = track_ids
    return pose_result


def _compute_iou_simple(box1, box2):
    """간단한 IoU 계산 함수"""
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