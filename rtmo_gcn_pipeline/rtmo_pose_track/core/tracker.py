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
    """트랙 정보"""
    track_id: int
    bbox: np.ndarray
    score: float
    age: int = 0
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    
    def __post_init__(self):
        """칼만 필터 초기화"""
        self.kalman = KalmanFilter()
        self.kalman.init_state(self.bbox)
    
    def update(self, bbox: np.ndarray, score: float):
        """트랙 업데이트"""
        self.bbox = self.kalman.update(bbox)
        self.score = score
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
    
    def to_bbox(self):
        """바운딩 박스 좌표 반환"""
        return self.bbox[:4] if len(self.bbox) >= 4 else self.bbox


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
        
    def update(self, detections: np.ndarray) -> List[Track]:
        """트래킹 업데이트"""
        for track in self.tracks:
            track.predict()
        
        high_dets = detections[detections[:, 4] >= self.high_thresh]
        low_dets = detections[(detections[:, 4] >= self.low_thresh) & 
                             (detections[:, 4] < self.high_thresh)]
        
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(
            self.tracks, high_dets, iou_threshold=0.5)
        
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(high_dets[det_idx, :4], high_dets[det_idx, 4])
        
        unmatched_tracks_for_low = [self.tracks[i] for i in unmatched_tracks 
                                   if self.tracks[i].time_since_update == 1]
        
        if len(unmatched_tracks_for_low) > 0 and len(low_dets) > 0:
            matched_tracks_low, unmatched_dets_low, unmatched_tracks_low = self._associate(
                unmatched_tracks_for_low, low_dets, iou_threshold=0.5)
            
            for track_idx, det_idx in matched_tracks_low:
                track = unmatched_tracks_for_low[track_idx]
                track.update(low_dets[det_idx, :4], low_dets[det_idx, 4])
        
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                bbox=high_dets[det_idx, :4],
                score=high_dets[det_idx, 4]
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