# Copyright (c) OpenMMLab. All rights reserved.
"""
Enhanced STGCN++ Dataset Annotation Generator
개선된 싸움 분류기를 위한 데이터셋 어노테이션 생성 시스템

주요 개선사항:
1. 5영역 분할 기반 위치 점수 시스템
2. 복합 점수 계산 (움직임 + 위치 + 상호작용 + 시간적 일관성 + 지속성)
3. 적응적 영역 가중치 학습
4. 모든 객체 랭킹 및 저장
5. 실패 케이스 로깅
6. 성능 최적화 및 병렬 처리
"""

import os
import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS



# =============================================================================
# Core Tracking Classes
# =============================================================================

class Detection:
    """바운딩 박스 검출 결과를 담는 클래스"""
    
    def __init__(self, bbox, confidence=None):
        self.bbox = bbox  # [x1, y1, x2, y2, score] 또는 [x1, y1, x2, y2]
        self.confidence = confidence or (bbox[4] if len(bbox) > 4 else 1.0)
    
    def to_bbox(self):
        """바운딩 박스 좌표 반환"""
        return self.bbox[:4] if len(self.bbox) >= 4 else self.bbox

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


# =============================================================================
# Enhanced Scoring System
# =============================================================================

class RegionBasedPositionScorer:
    """5영역 분할 기반 위치 점수 계산기"""
    
    def __init__(self, img_width, img_height):
        self.width = img_width
        self.height = img_height
        self.regions = self._define_regions()
        
        # 각 영역별 기본 활동성 가중치
        self.region_weights = {
            'top_left': 0.7,
            'top_right': 0.7,
            'bottom_left': 0.8,
            'bottom_right': 0.8,
            'center_overlap': 1.0
        }
    
    def _define_regions(self):
        """화면을 5개 영역으로 분할"""
        w, h = self.width, self.height
        
        # 중앙 겹침 영역 크기 (전체의 50%)
        center_w = int(w * 0.5)
        center_h = int(h * 0.5)
        center_x = (w - center_w) // 2
        center_y = (h - center_h) // 2
        
        regions = {
            'top_left': (0, 0, w//2, h//2),
            'top_right': (w//2, 0, w, h//2),
            'bottom_left': (0, h//2, w//2, h),
            'bottom_right': (w//2, h//2, w, h),
            'center_overlap': (center_x, center_y, center_x + center_w, center_y + center_h)
        }
        
        return regions
    
    def calculate_position_score(self, bbox_history):
        """5영역 기반 위치 점수 계산"""
        region_activities = {region: 0.0 for region in self.regions.keys()}
        total_frames = len(bbox_history)
        
        for bbox in bbox_history:
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            for region_name, (x1, y1, x2, y2) in self.regions.items():
                if x1 <= bbox_center_x <= x2 and y1 <= bbox_center_y <= y2:
                    relative_score = self._calculate_relative_position_score(
                        bbox_center_x, bbox_center_y, x1, y1, x2, y2, region_name
                    )
                    region_activities[region_name] += relative_score
        
        # 영역별 평균 활동도 계산
        region_scores = {}
        for region, activity in region_activities.items():
            avg_activity = activity / total_frames if total_frames > 0 else 0
            weighted_score = avg_activity * self.region_weights[region]
            region_scores[region] = weighted_score
        
        final_score = max(region_scores.values()) if region_scores else 0.0
        
        return final_score, region_scores
    
    def _calculate_relative_position_score(self, x, y, x1, y1, x2, y2, region_name):
        """영역 내에서의 상대적 위치 기반 점수"""
        region_width = x2 - x1
        region_height = y2 - y1
        
        rel_x = (x - x1) / region_width
        rel_y = (y - y1) / region_height
        
        if region_name == 'center_overlap':
            return self._center_region_score(rel_x, rel_y)
        else:
            return self._corner_region_score(rel_x, rel_y)
    
    def _corner_region_score(self, rel_x, rel_y):
        """모서리 영역에서의 점수 계산"""
        distance_from_center = np.sqrt((rel_x - 0.5)**2 + (rel_y - 0.5)**2)
        max_distance = np.sqrt(0.5**2 + 0.5**2)
        
        return 1.0 - (distance_from_center / max_distance)
    
    def _center_region_score(self, rel_x, rel_y):
        """중앙 겹침 영역에서의 점수 계산"""
        edge_distance = min(rel_x, 1-rel_x, rel_y, 1-rel_y)
        return 0.8 + 0.2 * (edge_distance / 0.5)


class AdaptiveRegionImportance:
    """비디오별로 적응적으로 영역 중요도를 학습하는 시스템"""
    
    def __init__(self, convergence_threshold=0.05):
        self.convergence_threshold = convergence_threshold
        self.region_importance_history = []
        
    def calculate_video_specific_importance(self, all_tracks_data, img_shape, iterations=5):
        """특정 비디오에 대해 반복적으로 영역 중요도 학습"""
        region_scorer = RegionBasedPositionScorer(img_shape[1], img_shape[0])
        
        current_weights = region_scorer.region_weights.copy()
        
        for iteration in range(iterations):
            track_scores = {}
            for track_id, track_data in all_tracks_data.items():
                bbox_history = [data['bbox'] for data in track_data.values()]
                
                region_scorer.region_weights = current_weights
                position_score, region_breakdown = region_scorer.calculate_position_score(bbox_history)
                
                movement_score = self._calculate_movement_intensity(track_data)
                
                combined_score = position_score * 0.6 + movement_score * 0.4
                track_scores[track_id] = {
                    'combined_score': combined_score,
                    'region_breakdown': region_breakdown,
                    'movement_score': movement_score
                }
            
            top_tracks = self._get_top_scoring_tracks(track_scores, top_k=5)
            new_weights = self._update_weights_from_top_tracks(
                top_tracks, current_weights
            )
            
            weight_change = sum(abs(new_weights[k] - current_weights[k]) 
                              for k in current_weights.keys())
            
            if weight_change < self.convergence_threshold:
                break
                
            current_weights = new_weights
            self.region_importance_history.append(current_weights.copy())
        
        return current_weights, track_scores
    
    def _calculate_movement_intensity(self, track_data):
        """움직임 강도 계산"""
        if len(track_data) < 2:
            return 0.0
        
        frame_indices = sorted(track_data.keys())
        intensities = []
        
        for i in range(1, len(frame_indices)):
            prev_frame = frame_indices[i-1]
            curr_frame = frame_indices[i]
            
            if 'keypoints' in track_data[prev_frame] and 'keypoints' in track_data[curr_frame]:
                try:
                    prev_kpts = np.array(track_data[prev_frame]['keypoints'])
                    curr_kpts = np.array(track_data[curr_frame]['keypoints'])
                    
                    # 배열 크기 확인
                    if prev_kpts.shape == curr_kpts.shape and prev_kpts.size > 0:
                        movement = np.linalg.norm(curr_kpts - prev_kpts, axis=-1)
                        if movement.size > 0:
                            intensities.append(float(np.mean(movement)))
                except Exception as e:
                    print(f"Error calculating movement intensity: {str(e)}")
                    continue
        
        return np.mean(intensities) if intensities else 0.0
    
    def _get_top_scoring_tracks(self, track_scores, top_k=5):
        """상위 K개 트랙 선별"""
        sorted_tracks = sorted(
            track_scores.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        )
        return dict(sorted_tracks[:top_k])
    
    def _update_weights_from_top_tracks(self, top_tracks, current_weights, alpha=0.3):
        """상위 트랙들의 영역 분포를 기반으로 가중치 업데이트"""
        region_usage = defaultdict(float)
        total_score = 0
        
        for track_id, track_info in top_tracks.items():
            track_score = track_info['combined_score']
            region_breakdown = track_info['region_breakdown']
            
            for region, region_score in region_breakdown.items():
                region_usage[region] += region_score * track_score
            total_score += track_score
        
        if total_score > 0:
            for region in region_usage:
                region_usage[region] /= total_score
        
        new_weights = {}
        for region, current_weight in current_weights.items():
            observed_importance = region_usage.get(region, 0)
            new_weights[region] = (1 - alpha) * current_weight + alpha * observed_importance
        
        return new_weights


class EnhancedFightInvolvementScorer:
    """개선된 싸움 참여도 점수 계산기"""
    
    def __init__(self, img_shape, enable_adaptive=True, weights=None):
        self.img_shape = img_shape
        self.enable_adaptive = enable_adaptive
        
        self.position_scorer = RegionBasedPositionScorer(img_shape[1], img_shape[0])
        
        if enable_adaptive:
            self.adaptive_analyzer = AdaptiveRegionImportance()

        # 가중치 설정
        if weights and len(weights) == 5:
            self.weights = {
                'movement': weights[0],
                'position': weights[1],
                'interaction': weights[2],
                'temporal_consistency': weights[3],
                'persistence': weights[4]
            }
        else:
            self.weights = {
                'movement': 0.30,
                'position': 0.35,
                'interaction': 0.20,
                'temporal_consistency': 0.10,
                'persistence': 0.05
            }
    
    def calculate_enhanced_fight_score(self, track_data, all_tracks_data=None):
        """개선된 복합 점수 계산"""
        # 1. 움직임 강도 점수
        movement_score = self._calculate_movement_intensity(track_data)
        
        # 2. 개선된 5영역 기반 위치 점수
        bbox_history = [data['bbox'] for data in track_data.values()]
        
        if self.enable_adaptive and all_tracks_data:
            adaptive_weights, _ = self.adaptive_analyzer.calculate_video_specific_importance(
                all_tracks_data, self.img_shape
            )
            self.position_scorer.region_weights = adaptive_weights
        
        position_score, region_breakdown = self.position_scorer.calculate_position_score(bbox_history)
        
        # 3. 상호작용 점수
        interaction_score = self._calculate_enhanced_interaction(track_data, all_tracks_data)
        
        # 4. 시간적 일관성 점수
        temporal_consistency = self._calculate_temporal_consistency(track_data)
        
        # 5. 지속성 점수
        persistence_score = len(track_data) / self._get_total_frames(all_tracks_data)
        
        # 최종 가중 점수 계산
        composite_score = (
            movement_score * self.weights['movement'] +
            position_score * self.weights['position'] +
            interaction_score * self.weights['interaction'] +
            temporal_consistency * self.weights['temporal_consistency'] +
            persistence_score * self.weights['persistence']
        )
        
        return {
            'composite_score': composite_score,
            'breakdown': {
                'movement': movement_score,
                'position': position_score,
                'interaction': interaction_score,
                'temporal_consistency': temporal_consistency,
                'persistence': persistence_score
            },
            'region_breakdown': region_breakdown
        }
    
    def _calculate_movement_intensity(self, track_data):
        """움직임 강도 계산"""
        if len(track_data) < 2:
            return 0.0
        
        frame_indices = sorted(track_data.keys())
        intensities = []
        
        for i in range(1, len(frame_indices)):
            prev_frame = frame_indices[i-1]
            curr_frame = frame_indices[i]
            
            if 'keypoints' in track_data[prev_frame] and 'keypoints' in track_data[curr_frame]:
                try:
                    prev_kpts = np.array(track_data[prev_frame]['keypoints'])
                    curr_kpts = np.array(track_data[curr_frame]['keypoints'])
                    
                    # 배열 크기 확인
                    if prev_kpts.shape == curr_kpts.shape and prev_kpts.size > 0:
                        movement = np.linalg.norm(curr_kpts - prev_kpts, axis=-1)
                        if movement.size > 0:
                            threshold = movement.mean() + 2*movement.std()
                            rapid_movement = np.sum(movement > threshold)
                            intensities.append(float(rapid_movement / len(movement)))
                except Exception as e:
                    print(f"Error calculating rapid movement: {str(e)}")
                    continue
        
        return np.mean(intensities) if intensities else 0.0
    
    def _calculate_enhanced_interaction(self, track_data, all_tracks_data):
        """개선된 상호작용 점수 계산"""
        if not all_tracks_data or len(all_tracks_data) <= 1:
            return 0.0
        
        interaction_intensity = 0.0
        interaction_count = 0
        
        current_track_id = None
        for tid, tdata in all_tracks_data.items():
            # 딕셔너리의 메모리 주소로 비교 (같은 객체인지 확인)
            if id(tdata) == id(track_data):
                current_track_id = tid
                break
        
        if current_track_id is None:
            return 0.0
        
        for other_id, other_data in all_tracks_data.items():
            if other_id == current_track_id:
                continue
            
            common_frames = set(track_data.keys()) & set(other_data.keys())
            
            for frame in common_frames:
                distance_score = self._calculate_proximity_interaction(
                    track_data[frame]['bbox'], 
                    other_data[frame]['bbox']
                )
                
                sync_score = self._calculate_movement_synchronization(
                    track_data, other_data, frame
                )
                
                interaction_intensity += (distance_score + sync_score) / 2
                interaction_count += 1
        
        return interaction_intensity / interaction_count if interaction_count > 0 else 0.0
    
    def _calculate_proximity_interaction(self, bbox1, bbox2):
        """근접도 기반 상호작용 점수"""
        center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
        center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
        
        distance = np.linalg.norm(center1 - center2)
        
        # 바운딩박스 크기 기반 정규화
        size1 = np.sqrt((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        size2 = np.sqrt((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
        avg_size = (size1 + size2) / 2
        
        normalized_distance = distance / avg_size if avg_size > 0 else float('inf')
        
        # 거리가 가까울수록 높은 점수 (임계값 3.0)
        return max(0, 1.0 - normalized_distance / 3.0)
    
    def _calculate_movement_synchronization(self, track_data1, track_data2, current_frame):
        """움직임 동기화 점수"""
        frame_indices1 = sorted(track_data1.keys())
        frame_indices2 = sorted(track_data2.keys())
        
        if current_frame not in frame_indices1 or current_frame not in frame_indices2:
            return 0.0
        
        idx1 = frame_indices1.index(current_frame)
        idx2 = frame_indices2.index(current_frame)
        
        if idx1 == 0 or idx2 == 0:
            return 0.0
        
        prev_frame1 = frame_indices1[idx1 - 1]
        prev_frame2 = frame_indices2[idx2 - 1]
        
        if prev_frame1 not in track_data1 or prev_frame2 not in track_data2:
            return 0.0
        
        # 움직임 벡터 계산
        movement1 = self._get_movement_vector(track_data1[prev_frame1], track_data1[current_frame])
        movement2 = self._get_movement_vector(track_data2[prev_frame2], track_data2[current_frame])
        
        return self._calculate_vector_similarity(movement1, movement2)
    
    def _calculate_temporal_consistency(self, track_data):
        """시간적 일관성 점수"""
        if len(track_data) < 3:
            return 1.0
        
        frame_indices = sorted(track_data.keys())
        consistency_scores = []
        
        for i in range(2, len(frame_indices)):
            prev_prev = frame_indices[i-2]
            prev = frame_indices[i-1] 
            curr = frame_indices[i]
            
            movement1 = self._get_movement_vector(
                track_data[prev_prev], track_data[prev]
            )
            
            movement2 = self._get_movement_vector(
                track_data[prev], track_data[curr]
            )
            
            consistency = self._calculate_vector_similarity(movement1, movement2)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _get_movement_vector(self, prev_data, curr_data):
        """움직임 벡터 계산"""
        try:
            if 'keypoints' in prev_data and 'keypoints' in curr_data:
                prev_kpts = np.array(prev_data['keypoints'])
                curr_kpts = np.array(curr_data['keypoints'])
                
                # 배열 크기 확인
                if prev_kpts.shape == curr_kpts.shape and prev_kpts.size > 0:
                    movement = curr_kpts - prev_kpts
                    return np.mean(movement, axis=0)  # 평균 움직임 벡터
            
            # 바운딩박스 중심점 기반 움직임
            if 'bbox' in prev_data and 'bbox' in curr_data:
                prev_center = np.array([(prev_data['bbox'][0] + prev_data['bbox'][2])/2,
                                       (prev_data['bbox'][1] + prev_data['bbox'][3])/2])
                curr_center = np.array([(curr_data['bbox'][0] + curr_data['bbox'][2])/2,
                                       (curr_data['bbox'][1] + curr_data['bbox'][3])/2])
                return curr_center - prev_center
                
            return np.zeros(2)  # 기본값
            
        except Exception as e:
            print(f"Error calculating movement vector: {str(e)}")
            return np.zeros(2)
    
    def _calculate_vector_similarity(self, vec1, vec2):
        """벡터 유사도 계산 (코사인 유사도)"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return max(0, float(cosine_sim))  # 음수 유사도는 0으로 처리
            
        except Exception as e:
            print(f"Error calculating vector similarity: {str(e)}")
            return 0.0
    
    def _get_total_frames(self, all_tracks_data):
        """전체 프레임 수 계산"""
        if not all_tracks_data:
            return 1
        
        all_frames = set()
        for track_data in all_tracks_data.values():
            all_frames.update(track_data.keys())
        
        return len(all_frames) if all_frames else 1

# =============================================================================
# Enhanced Annotation Creator
# =============================================================================

def create_detection_results(pose_result):
    """포즈 추정 결과에서 detection 형태로 변환"""
    pred_instances = pose_result._pred_instances
    
    if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
        return np.empty((0, 5))
    
    bboxes = pred_instances.bboxes.cpu().numpy() if hasattr(pred_instances.bboxes, 'cpu') else pred_instances.bboxes
    if hasattr(pred_instances, 'bbox_scores'):
        scores = pred_instances.bbox_scores.cpu().numpy() if hasattr(pred_instances.bbox_scores, 'cpu') else pred_instances.bbox_scores
        scores = scores.reshape(-1, 1)
    else:
        scores = np.ones((len(bboxes), 1))
    
    detections = np.concatenate([bboxes, scores], axis=1)
    return detections


def assign_track_ids_from_bytetrack(pose_result, active_tracks, iou_threshold=0.5):
    """ByteTrack 결과를 기반으로 pose_result에 track_id 할당"""
    frame_result = pose_result._pred_instances
    if not hasattr(frame_result, 'bboxes') or len(frame_result.bboxes) == 0:
        frame_result.track_ids = np.array([])
        return pose_result

    pose_bboxes = frame_result.bboxes.cpu().numpy() if hasattr(frame_result.bboxes, 'cpu') else frame_result.bboxes
    track_ids = np.full(len(pose_bboxes), -1, dtype=int)
    
    for i, pose_bbox in enumerate(pose_bboxes):
        best_iou = 0
        best_track_id = -1
        
        for track in active_tracks:
            current_iou = calculate_iou(pose_bbox, track.bbox)
            if current_iou > best_iou and current_iou > iou_threshold:
                best_iou = current_iou
                best_track_id = track.track_id
        
        track_ids[i] = best_track_id

    frame_result.track_ids = track_ids
    return pose_result


def calculate_iou(box1, box2):
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


def get_num_keypoints_from_model(pose_model):
    """모델 설정에서 키포인트 개수 추출"""
    try:
        if hasattr(pose_model, 'dataset_meta') and pose_model.dataset_meta is not None:
            if 'num_keypoints' in pose_model.dataset_meta:
                return pose_model.dataset_meta['num_keypoints']
            elif 'keypoint_info' in pose_model.dataset_meta:
                return len(pose_model.dataset_meta['keypoint_info'])
            elif 'keypoints' in pose_model.dataset_meta:
                return len(pose_model.dataset_meta['keypoints'])
        
        if hasattr(pose_model, 'cfg'):
            if hasattr(pose_model.cfg, 'model') and hasattr(pose_model.cfg.model, 'num_keypoints'):
                return pose_model.cfg.model.num_keypoints
            elif hasattr(pose_model.cfg, 'num_keypoints'):
                return pose_model.cfg.num_keypoints
    except Exception as e:
        print(f"Warning: Could not extract keypoint number from model: {e}")
    
    print("Warning: Using default keypoint number (17).")
    return 17


def collect_all_tracks_data(pose_results, min_track_length=10):
    """모든 트랙 데이터 수집"""
    all_tracks_data = defaultdict(dict)
    
    for f_idx, result in enumerate(pose_results):
        pred_instances = result._pred_instances
        if not hasattr(pred_instances, 'track_ids'):
            continue
        
        instance_track_ids = pred_instances.track_ids
        instance_keypoints = pred_instances.keypoints
        instance_scores = pred_instances.keypoint_scores
        instance_bboxes = pred_instances.bboxes
        
        for p_idx in range(len(instance_track_ids)):
            tid = instance_track_ids[p_idx]
            # track_id가 numpy array인 경우 스칼라 값으로 추출
            if isinstance(tid, np.ndarray):
                tid = tid.item() if tid.size == 1 else tid[0]
            # 정수형으로 변환
            tid = int(tid) if tid is not None else -1
            if tid >= 0:  # 유효한 track_id만
                all_tracks_data[tid][f_idx] = {
                    'keypoints': instance_keypoints[p_idx].cpu().numpy() if hasattr(instance_keypoints[p_idx], 'cpu') else instance_keypoints[p_idx],
                    'scores': instance_scores[p_idx].cpu().numpy() if hasattr(instance_scores[p_idx], 'cpu') else instance_scores[p_idx],
                    'bbox': instance_bboxes[p_idx].cpu().numpy() if hasattr(instance_bboxes[p_idx], 'cpu') else instance_bboxes[p_idx]
                }
    
    # 최소 길이 필터링
    filtered_tracks = {tid: data for tid, data in all_tracks_data.items() 
                      if len(data) >= min_track_length}
    
    return filtered_tracks


def apply_advanced_interpolation(keypoints, scores, confidence_threshold=0.3):
    """고급 보간 알고리즘 적용"""
    num_persons, num_frames, num_keypoints, _ = keypoints.shape
    
    for p_idx in range(num_persons):
        for k_idx in range(num_keypoints):
            # 현재 키포인트의 시간축 데이터
            kpt_x = keypoints[p_idx, :, k_idx, 0]
            kpt_y = keypoints[p_idx, :, k_idx, 1]
            kpt_scores = scores[p_idx, :, k_idx]
            
            # 신뢰도가 높은 프레임 찾기
            valid_frames = np.where(kpt_scores > confidence_threshold)[0]
            
            if len(valid_frames) < 2:
                continue  # 유효한 프레임이 너무 적으면 보간 스킵
            
            # 스플라인 보간
            try:
                interp_x = interp1d(valid_frames, kpt_x[valid_frames], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(valid_frames, kpt_y[valid_frames], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                
                # 모든 프레임에 대해 보간값 적용
                all_frames = np.arange(num_frames)
                keypoints[p_idx, :, k_idx, 0] = interp_x(all_frames)
                keypoints[p_idx, :, k_idx, 1] = interp_y(all_frames)
                
                # 보간된 프레임의 신뢰도는 낮게 설정
                interpolated_frames = np.setdiff1d(all_frames, valid_frames)
                scores[p_idx, interpolated_frames, k_idx] = confidence_threshold * 0.8
                
            except Exception as e:
                # 스플라인 실패 시 선형 보간 사용
                interp_x = interp1d(valid_frames, kpt_x[valid_frames], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(valid_frames, kpt_y[valid_frames], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                
                all_frames = np.arange(num_frames)
                keypoints[p_idx, :, k_idx, 0] = interp_x(all_frames)
                keypoints[p_idx, :, k_idx, 1] = interp_y(all_frames)
                
                interpolated_frames = np.setdiff1d(all_frames, valid_frames)
                scores[p_idx, interpolated_frames, k_idx] = confidence_threshold * 0.8
    
    return keypoints, scores


def create_enhanced_annotation(pose_results, video_path, pose_model, 
                             min_track_length=10, quality_threshold=0.3, weights=None):
    """개선된 어노테이션 생성 (모든 객체 랭킹)"""
    if not pose_results:
        return None, "No pose results"
    
    # 1. 모든 Track ID 수집 및 기본 필터링
    all_tracks_data = collect_all_tracks_data(pose_results, min_track_length)
    
    if len(all_tracks_data) == 0:
        return None, f"No tracks with minimum length {min_track_length}"
    
    # 2. 이미지 크기 추출
    img_shape = pose_results[0].img_shape
    
    # 3. 개선된 점수 계산기 초기화 (가중치 전달)
    scorer = EnhancedFightInvolvementScorer(img_shape, enable_adaptive=True, weights=weights)
    
    # 4. 각 Track ID에 대해 복합 점수 계산
    scored_tracks = []
    for track_id, track_data in all_tracks_data.items():
        score_info = scorer.calculate_enhanced_fight_score(track_data, all_tracks_data)
        scored_tracks.append((track_id, score_info, track_data))
    
    # 5. 점수순 정렬 (내림차순)
    scored_tracks.sort(key=lambda x: x[1]['composite_score'], reverse=True)
    
    # 6. 품질 필터링
    quality_tracks = []
    for track_id, score_info, track_data in scored_tracks:
        track_quality = _calculate_track_quality(track_data)
        if track_quality >= quality_threshold:
            quality_tracks.append((track_id, score_info, track_data, track_quality))
    
    if len(quality_tracks) == 0:
        return None, f"No tracks meet quality threshold {quality_threshold}"
    
    # 7. 모든 객체에 대한 어노테이션 생성
    num_keypoints = get_num_keypoints_from_model(pose_model)
    all_annotations = {}
    
    for rank, (track_id, score_info, track_data, track_quality) in enumerate(quality_tracks):
        person_annotation = create_single_person_annotation(
            track_id, track_data, pose_results, num_keypoints
        )
        
        all_annotations[f'person_{rank:02d}'] = {
            'track_id': track_id,
            'composite_score': score_info['composite_score'],
            'score_breakdown': score_info['breakdown'],
            'region_breakdown': score_info['region_breakdown'],
            'track_quality': track_quality,
            'rank': rank + 1,
            'annotation': person_annotation
        }
    
    # 8. 전체 메타데이터 추가
    final_annotation = {
        'total_persons': len(quality_tracks),
        'video_info': {
            'frame_dir': os.path.splitext(os.path.basename(video_path))[0],
            'total_frames': len(pose_results),
            'img_shape': img_shape,
            'original_shape': pose_results[0].ori_shape,
            'label': 1 if '/Fight/' in video_path else 0
        },
        'persons': all_annotations,
        'score_weights': {
            'movement_intensity': 0.30,
            'position_5region': 0.35,
            'interaction': 0.20,
            'temporal_consistency': 0.10,
            'persistence': 0.05
        },
        'quality_threshold': quality_threshold,
        'min_track_length': min_track_length
    }
    
    return final_annotation, "Success"


def create_single_person_annotation(track_id, track_data, pose_results, num_keypoints):
    """단일 Track ID에 대한 어노테이션 생성"""
    num_frames = len(pose_results)
    
    # 초기화
    keypoints = np.zeros((1, num_frames, num_keypoints, 2), dtype=np.float32)
    scores = np.zeros((1, num_frames, num_keypoints), dtype=np.float32)
    
    # 데이터 채우기
    for f_idx in range(num_frames):
        if f_idx in track_data:
            frame_data = track_data[f_idx]
            available_kpts = min(frame_data['keypoints'].shape[0], num_keypoints)
            keypoints[0, f_idx, :available_kpts] = frame_data['keypoints'][:available_kpts]
            scores[0, f_idx, :available_kpts] = frame_data['scores'][:available_kpts]
    
    # 고급 보간 적용
    keypoints, scores = apply_advanced_interpolation(keypoints, scores)
    
    return {
        'keypoint': keypoints,
        'keypoint_score': scores,
        'num_keypoints': num_keypoints,
        'track_id': track_id
    }


def _calculate_track_quality(track_data):
    """트랙 품질 점수 계산"""
    if not track_data:
        return 0.0
    
    all_scores = []
    for frame_data in track_data.values():
        if 'scores' in frame_data:
            all_scores.extend(frame_data['scores'])
    
    return np.mean(all_scores) if all_scores else 0.0


# =============================================================================
# File Operations
# =============================================================================





def draw_track_ids(frame, pose_result, track_id_to_rank: Dict[int, int], top_track_ids: Optional[List[int]] = None):
    """프레임에 TrackID와 전체 순위(Rank) 표시"""
    try:
        if hasattr(pose_result, 'pred_instances') and hasattr(pose_result.pred_instances, 'track_ids'):
            track_ids = pose_result.pred_instances.track_ids
            keypoints = pose_result.pred_instances.keypoints
            
            for i, track_id in enumerate(track_ids):
                # track_id가 numpy array인 경우 스칼라 값으로 추출
                if isinstance(track_id, np.ndarray):
                    track_id = track_id.item() if track_id.size == 1 else track_id[0]
                # 정수형으로 변환
                track_id = int(track_id) if track_id is not None else -1
                
                if track_id >= 0 and track_id in track_id_to_rank:
                    if len(keypoints[i]) > 0:
                        kpts = keypoints[i]
                        if len(kpts.shape) == 2 and kpts.shape[1] >= 2:
                            if kpts.shape[1] >= 3:
                                valid_kpts = kpts[kpts[:, 2] > 0.5]
                            else:
                                valid_kpts = kpts[~np.isnan(kpts[:, 0]) & ~np.isnan(kpts[:, 1])]
                            
                            if len(valid_kpts) > 0:
                                head_kpt = valid_kpts[np.argmin(valid_kpts[:, 1])]
                                x, y = int(head_kpt[0]), int(head_kpt[1])
                                
                                # 전체 순위(Rank) 조회
                                rank = track_id_to_rank[track_id]
                                
                                # 텍스트 형식: "ID: tracker_id, Rank: rank"
                                text = f"ID: {track_id}, Rank: {rank}"
                                
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.4
                                thickness = 1
                                
                                # 상위 N명은 다른 색으로 강조 (top_track_ids가 주어진 경우)
                                if top_track_ids and track_id in top_track_ids:
                                    bg_color = (0, 255, 0)  # 형광 녹색 배경 (BGR 순서)
                                    text_color = (0, 0, 0)  # 검은색 텍스트 (가독성 향상)
                                else:
                                    bg_color = (50, 50, 50)
                                    text_color = (255, 255, 255)

                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                
                                cv2.rectangle(frame, 
                                            (x - 3, y - text_height - 8), 
                                            (x + text_width + 3, y + 3), 
                                            bg_color, -1)
                                
                                cv2.putText(frame, text, (x, y - 3), font, font_scale, 
                                          text_color, thickness)
        
        return frame
        
    except Exception as e:
        print(f"Warning: Failed to draw track IDs: {e}")
        return frame# =============================================================================
# Enhanced RTMO Pose Extractor Class
# =============================================================================

class EnhancedRTMOPoseExtractor:
    """개선된 RTMO 포즈 추출기 클래스"""
    
    def __init__(self, config_file, checkpoint_file, device='cuda:0', 
                 gpu_ids=None, multi_gpu=False,
                 score_thr=0.3, nms_thr=0.35,
                 track_high_thresh=0.6, track_low_thresh=0.1,
                 track_max_disappeared=30, track_min_hits=3,
                 quality_threshold=0.3, min_track_length=10,
                 weights=None):
        """
        Args:
            config_file: RTMO 설정 파일 경로
            checkpoint_file: RTMO 체크포인트 파일 경로
            device: 추론에 사용할 디바이스
            gpu_ids: 사용할 GPU ID 리스트
            multi_gpu: 멀티 GPU 사용 여부
            score_thr: 포즈 검출 점수 임계값
            nms_thr: NMS 임계값
            track_high_thresh: ByteTracker 높은 임계값
            track_low_thresh: ByteTracker 낮은 임계값
            track_max_disappeared: 트랙 최대 소실 프레임
            track_min_hits: 트랙 최소 히트 수
            quality_threshold: 품질 임계값
            min_track_length: 최소 트랙 길이
            weights: 복합점수 가중치
        """
        self.config_path = config_file
        self.checkpoint_path = checkpoint_file
        self.device = device
        self.gpu_ids = gpu_ids or []
        self.multi_gpu = multi_gpu
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        
        # 트래킹 설정
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_max_disappeared = track_max_disappeared
        self.track_min_hits = track_min_hits
        self.quality_threshold = quality_threshold
        self.min_track_length = min_track_length
        self.weights = weights or [0.45, 0.10, 0.30, 0.10, 0.05]
        
        # 모델을 생성자에서 한 번만 초기화
        print(f"Initializing RTMO model: {config_file}")
        self.pose_model = init_model(config_file, checkpoint_file, device=device)
        
        # 모델 설정 적용
        self._configure_model()
        print("RTMO model initialized successfully")
    
    def _configure_model(self):
        """모델 설정 적용"""
        if hasattr(self.pose_model.cfg, 'model'):
            if hasattr(self.pose_model.cfg.model, 'test_cfg'):
                self.pose_model.cfg.model.test_cfg.score_thr = self.score_thr
                self.pose_model.cfg.model.test_cfg.nms_thr = self.nms_thr
            else:
                self.pose_model.cfg.model.test_cfg = dict(score_thr=self.score_thr, nms_thr=self.nms_thr)
        
        if hasattr(self.pose_model, 'head') and hasattr(self.pose_model.head, 'test_cfg'):
            self.pose_model.head.test_cfg.score_thr = self.score_thr
            self.pose_model.head.test_cfg.nms_thr = self.nms_thr
    
    def extract_poses_only(self, video_path, failure_logger=None):
        """전체 비디오에 대해 포즈 추정만 수행 (트래킹 제외)"""
        try:
            print(f"Extracting poses from: {video_path}")
            
            # CUDA 메모리 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # 이미 초기화된 모델 사용 (매번 로드하지 않음)
            pose_model = self.pose_model
        
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if failure_logger:
                    failure_logger.log_failure(video_path, "Cannot open video file")
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            pose_results = []
            
            # 포즈 추정만 수행 (트래킹 없음)
            print(f"Running pose estimation on {total_frames} frames...")
            pbar = tqdm(total=total_frames, desc="Extracting poses")

            while True:
                success, frame = cap.read()
                if not success:
                    break

                # 포즈 추정만 수행
                batch_pose_results = inference_bottomup(pose_model, frame)
                pose_result = batch_pose_results[0]
                
                pose_results.append(pose_result)
                pbar.update(1)

            cap.release()
            pbar.close()
            
            if not pose_results:
                return None
            
            print(f"Extracted poses from {len(pose_results)} frames")
            return pose_results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if failure_logger:
                failure_logger.log_failure(video_path, f"Pose extraction error: {str(e)}")
            return None
    
    def apply_tracking_to_poses(self, poses_data: list, start_frame: int, end_frame: int, window_idx: int):
        """저장된 포즈 데이터에 트래킹 및 복합점수 적용"""
        try:
            if not poses_data or len(poses_data) == 0:
                return None
            
            print(f"    Debug: Window {window_idx} processing {len(poses_data)} frames")
            
            # ByteTracker 초기화
            tracker = ByteTracker(
                high_thresh=self.track_high_thresh,
                low_thresh=self.track_low_thresh,
                max_disappeared=self.track_max_disappeared,
                min_hits=self.track_min_hits
            )
            
            # 트래킹 적용
            tracked_poses = []
            for frame_idx, pose_result in enumerate(poses_data):
                try:
                    # PoseDataSample 객체를 표준 형식으로 변환
                    converted_pose = self._convert_pose_data_sample(pose_result)
                    
                    # ByteTracker에 전달할 detection 생성
                    detections = np.array([p['bbox'] for p in converted_pose if 'bbox' in p]) if converted_pose else np.empty((0, 5))
                    
                    # 트래커 업데이트 및 활성 트랙 가져오기
                    tracks = tracker.update(detections)
                    
                    if converted_pose and len(converted_pose) > 0:
                        # 트래킹 결과를 포즈 데이터에 매핑
                        tracked_frame = self._map_tracks_to_poses(converted_pose, tracks)
                        tracked_poses.append(tracked_frame)
                    else:
                        # 포즈가 없는 프레임도 빈 리스트로 추가
                        tracked_poses.append([])
                        
                except Exception as e:
                    print(f"      Error in frame {frame_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    tracked_poses.append([])
            
            # 복합점수 계산 및 어노테이션 생성
            # 간단한 어노테이션 생성
            annotation = self._generate_simple_annotation(tracked_poses)
            
            if not annotation or 'persons' not in annotation or not annotation['persons']:
                return None
            
            # 윈도우 결과 구성
            window_result = {
                'window_idx': window_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'annotation': annotation,
                'tracking_applied': True,
                'frame_count': len(tracked_poses)
            }
            
            return window_result
            
        except Exception as e:
            print(f"Error applying tracking to poses: {str(e)}")
            return None
    
    def _convert_pose_data_sample(self, pose_data_sample):
        """PoseDataSample 객체를 표준 형식으로 변환"""
        try:
            # MMPose의 PoseDataSample 형식인지 확인
            if hasattr(pose_data_sample, 'pred_instances'):
                instances = pose_data_sample.pred_instances
                converted_poses = []
                
                # 각 인스턴스를 표준 형식으로 변환
                if hasattr(instances, 'keypoints') and hasattr(instances, 'bboxes'):
                    # torch tensor이면 cpu로 이동, numpy array이면 그대로 사용
                    if hasattr(instances.keypoints, 'cpu'):
                        keypoints = instances.keypoints.cpu().numpy()
                    else:
                        keypoints = np.array(instances.keypoints)
                    
                    if hasattr(instances.bboxes, 'cpu'):
                        bboxes = instances.bboxes.cpu().numpy()
                    else:
                        bboxes = np.array(instances.bboxes)
                    
                    # keypoint_scores 처리 (visibility scores)
                    keypoint_scores = None
                    if hasattr(instances, 'keypoint_scores'):
                        if hasattr(instances.keypoint_scores, 'cpu'):
                            keypoint_scores = instances.keypoint_scores.cpu().numpy()
                        else:
                            keypoint_scores = np.array(instances.keypoint_scores)
                    
                    # bbox_scores 처리
                    bbox_scores = None
                    if hasattr(instances, 'bbox_scores'):
                        if hasattr(instances.bbox_scores, 'cpu'):
                            bbox_scores = instances.bbox_scores.cpu().numpy()
                        else:
                            bbox_scores = np.array(instances.bbox_scores)
                    
                    for i in range(len(keypoints)):
                        person_keypoints = keypoints[i]  # 원본 keypoints
                        
                        # keypoints가 (17, 2) 형태인 경우 (17, 3)으로 확장
                        if person_keypoints.shape[-1] == 2:
                            # visibility scores 추가
                            if keypoint_scores is not None and i < len(keypoint_scores):
                                # keypoint_scores가 있으면 사용
                                visibility = keypoint_scores[i]
                                person_keypoints = np.concatenate([
                                    person_keypoints, 
                                    visibility.reshape(-1, 1)
                                ], axis=1)
                            else:
                                # keypoint_scores가 없으면 기본값 1.0 사용
                                visibility = np.ones((person_keypoints.shape[0], 1), dtype=np.float32)
                                person_keypoints = np.concatenate([person_keypoints, visibility], axis=1)
                        
                        person_data = {
                            'keypoints': person_keypoints,  # (17, 3) 형태로 보장
                            'bbox': list(bboxes[i]) + [bbox_scores[i] if bbox_scores is not None else 1.0],
                            'score': bbox_scores[i] if bbox_scores is not None else 1.0
                        }
                        converted_poses.append(person_data)
                
                return converted_poses
            
            # 이미 표준 형식인 경우
            elif isinstance(pose_data_sample, list):
                return pose_data_sample
            
            # 빈 결과
            else:
                return []
                
        except Exception as e:
            print(f"Error converting PoseDataSample: {str(e)}")
            return []
    
    def _safe_array_check(self, arr, min_length):
        """배열 크기를 안전하게 체크"""
        try:
            if isinstance(arr, (list, tuple)):
                return len(arr) >= min_length
            elif isinstance(arr, np.ndarray):
                return arr.shape[0] >= min_length
            else:
                return False
        except Exception:
            return False
    
    def _has_valid_keypoints(self, keypoints):
        """키포인트가 유효한지 안전하게 확인"""
        try:
            if keypoints is None:
                return False
            
            if isinstance(keypoints, (list, tuple)):
                return len(keypoints) > 0
            elif isinstance(keypoints, np.ndarray):
                return keypoints.size > 0 and not np.all(keypoints == 0)
            else:
                return bool(keypoints) if keypoints is not None else False
        except Exception:
            return False
    
    def _generate_simple_annotation(self, tracked_poses):
        """간단한 어노테이션 생성"""
        try:
            if not tracked_poses:
                return None
            
            # 트랙 ID별로 데이터 수집
            track_data = {}
            
            for frame_idx, frame_poses in enumerate(tracked_poses):
                if frame_poses:
                    for person in frame_poses:
                        if 'track_id' in person:
                            track_id = person['track_id']
                            if track_id not in track_data:
                                track_data[track_id] = []
                            
                            track_data[track_id].append({
                                'frame_idx': frame_idx,
                                'keypoints': person.get('keypoints', []),
                                'bbox': person.get('bbox', []),
                                'score': person.get('score', 0.0)
                            })
            
            if not track_data:
                return None
            
            # 각 트랙에 대해 복합 점수 계산 (간단화)
            persons = {}
            for track_id, frames in track_data.items():
                if len(frames) >= self.min_track_length:
                    # 평균 점수 계산 (안전한 스칼라 변환)
                    scores = []
                    for f in frames:
                        score = f['score']
                        if isinstance(score, np.ndarray):
                            score = float(score.item()) if score.size == 1 else float(score.mean())
                        scores.append(float(score))
                    avg_score = np.mean(scores)
                    
                    # 키포인트 데이터 구성
                    num_frames = len(tracked_poses)
                    keypoints_data = np.zeros((1, num_frames, 17, 2), dtype=np.float32)
                    scores_data = np.zeros((1, num_frames, 17), dtype=np.float32)
                    
                    for frame_data in frames:
                        frame_idx = frame_data['frame_idx']
                        # 키포인트 존재 여부를 안전하게 확인
                        keypoints = frame_data.get('keypoints', [])
                        if self._has_valid_keypoints(keypoints):
                            try:
                                kpts = np.array(keypoints)
                                
                                # 키포인트 형태 확인 및 처리
                                if kpts.ndim == 1 and len(kpts) >= 51:
                                    # 평면화된 경우 (51,) -> (17, 3)으로 변형
                                    kpts = kpts.reshape(-1, 3)
                                elif kpts.ndim == 2:
                                    # 이미 (17, 3) 또는 (N, 3) 형태인 경우
                                    if kpts.shape[0] >= 17 and kpts.shape[1] >= 3:
                                        pass  # 올바른 형태
                                    else:
                                        print(f"Invalid 2D keypoint shape for track {track_id}, frame {frame_idx}: {kpts.shape}")
                                        continue
                                elif kpts.ndim == 3:
                                    # (1, 17, 3) 형태인 경우 squeeze
                                    kpts = kpts.squeeze(0)
                                else:
                                    print(f"Unsupported keypoint dimension for track {track_id}, frame {frame_idx}: {kpts.ndim}")
                                    continue
                                
                                if self._safe_array_check(kpts, 17):
                                    # 안전한 인덱싱 및 배열 크기 검증
                                    try:
                                        # 배열 크기 체크
                                        kpts_subset = kpts[:17]
                                        if kpts_subset.ndim >= 2 and kpts_subset.shape[1] >= 3:
                                            keypoints_data[0, frame_idx, :17, 0] = kpts_subset[:, 0]
                                            keypoints_data[0, frame_idx, :17, 1] = kpts_subset[:, 1]
                                            scores_data[0, frame_idx, :17] = kpts_subset[:, 2]
                                        else:
                                            print(f"Invalid keypoint subset shape for track {track_id}, frame {frame_idx}: {kpts_subset.shape}")
                                    except (IndexError, ValueError) as idx_err:
                                        print(f"Index error for track {track_id}, frame {frame_idx}: {str(idx_err)}")
                                        continue
                            except Exception as e:
                                print(f"Error processing keypoints for track {track_id}, frame {frame_idx}: {str(e)}")
                                continue
                    
                    persons[track_id] = {
                        'keypoint': keypoints_data,
                        'keypoint_score': scores_data,
                        'num_keypoints': 17,
                        'track_id': track_id,
                        'composite_score': float(avg_score)
                    }
            
            if not persons:
                return None
            
            return {
                'persons': persons,
                'metadata': {
                    'num_persons': len(persons),
                    'frame_count': len(tracked_poses),
                    'quality_threshold': self.quality_threshold
                }
            }
            
        except Exception as e:
            print(f"Error generating annotation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _map_tracks_to_poses(self, pose_result: list, tracks: list) -> list:
        """트래킹 결과를 포즈 데이터에 매핑"""
        try:
            if not tracks:
                return pose_result
            
            tracked_poses = []
            
            # 각 트랙에 대해 가장 가까운 포즈 찾기
            for track in tracks:
                try:
                    track_bbox = track.to_bbox()
                    track_center = [(track_bbox[0] + track_bbox[2]) / 2, 
                                   (track_bbox[1] + track_bbox[3]) / 2]
                    
                    best_match = None
                    best_distance = float('inf')
                    
                    for person in pose_result:
                        if 'bbox' in person:
                            person_bbox = person['bbox'][:4]  # [x1, y1, x2, y2]만 사용
                            person_center = [(person_bbox[0] + person_bbox[2]) / 2,
                                           (person_bbox[1] + person_bbox[3]) / 2]
                            
                            # 중심점 간 거리 계산
                            distance = np.sqrt((track_center[0] - person_center[0])**2 + 
                                             (track_center[1] - person_center[1])**2)
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_match = person.copy()
                    
                    if best_match is not None:
                        # 트랙 ID 추가
                        best_match['track_id'] = int(track.track_id) if hasattr(track, 'track_id') else -1
                        # numpy 배열을 안전하게 리스트로 변환
                        if isinstance(track_bbox, np.ndarray):
                            best_match['track_bbox'] = track_bbox.tolist()
                        else:
                            best_match['track_bbox'] = list(track_bbox)
                        tracked_poses.append(best_match)
                        
                except Exception as e:
                    print(f"        Error processing track {track.track_id}: {str(e)}")
                    continue
            
            return tracked_poses
            
        except Exception as e:
            print(f"Error mapping tracks to poses: {str(e)}")
            return pose_result



