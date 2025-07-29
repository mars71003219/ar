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
import logging
import glob
import time
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import cv2
import mmcv
import mmengine
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS


# =============================================================================
# Core Tracking Classes (기존 유지)
# =============================================================================

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
                prev_kpts = track_data[prev_frame]['keypoints']
                curr_kpts = track_data[curr_frame]['keypoints']
                
                movement = np.linalg.norm(curr_kpts - prev_kpts, axis=1)
                intensities.append(np.mean(movement))
        
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
                prev_kpts = track_data[prev_frame]['keypoints']
                curr_kpts = track_data[curr_frame]['keypoints']
                
                movement = np.linalg.norm(curr_kpts - prev_kpts, axis=1)
                rapid_movement = np.sum(movement > movement.mean() + 2*movement.std())
                intensities.append(rapid_movement / len(movement))
        
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
        if 'keypoints' in prev_data and 'keypoints' in curr_data:
            prev_kpts = prev_data['keypoints']
            curr_kpts = curr_data['keypoints']
            movement = curr_kpts - prev_kpts
            return np.mean(movement, axis=0)  # 평균 움직임 벡터
        else:
            # 바운딩박스 중심점 기반 움직임
            prev_center = np.array([(prev_data['bbox'][0] + prev_data['bbox'][2])/2,
                                   (prev_data['bbox'][1] + prev_data['bbox'][3])/2])
            curr_center = np.array([(curr_data['bbox'][0] + curr_data['bbox'][2])/2,
                                   (curr_data['bbox'][1] + curr_data['bbox'][3])/2])
            return curr_center - prev_center
    
    def _calculate_vector_similarity(self, vec1, vec2):
        """벡터 유사도 계산 (코사인 유사도)"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return max(0, cosine_sim)  # 음수 유사도는 0으로 처리
    
    def _get_total_frames(self, all_tracks_data):
        """전체 프레임 수 계산"""
        if not all_tracks_data:
            return 1
        
        all_frames = set()
        for track_data in all_tracks_data.values():
            all_frames.update(track_data.keys())
        
        return len(all_frames) if all_frames else 1


# =============================================================================
# Failure Logging System
# =============================================================================

class FailureLogger:
    """실패 케이스 로깅 시스템"""
    
    FAILURE_CATEGORIES = {
        'NO_TRACKS': "No valid tracks found",
        'INSUFFICIENT_LENGTH': "Tracks too short (< min_track_length)",
        'LOW_QUALITY': "All tracks below quality threshold", 
        'PROCESSING_ERROR': "Technical processing error",
        'EMPTY_VIDEO': "Empty or corrupted video file"
    }
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """로그 파일 초기화"""
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Enhanced STGCN++ Annotation Failure Log\n")
                f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Format: [timestamp] video_path | failure_reason\n\n")
    
    def log_failure(self, video_path, reason, additional_info=None):
        """실패 케이스 로깅"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {video_path} | {reason}"
        
        if additional_info:
            log_entry += f" | {additional_info}"
        
        log_entry += "\n"
        
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"FAILED: {os.path.basename(video_path)} - {reason}")
    
    def categorize_failure(self, tracks_data, min_length, quality_threshold):
        """실패 원인 자동 분류"""
        if len(tracks_data) == 0:
            return self.FAILURE_CATEGORIES['NO_TRACKS']
        
        valid_length_tracks = [t for t in tracks_data if len(t) >= min_length]
        if len(valid_length_tracks) == 0:
            return self.FAILURE_CATEGORIES['INSUFFICIENT_LENGTH']
        
        high_quality_tracks = [t for t in valid_length_tracks 
                              if self._calculate_track_quality(t) >= quality_threshold]
        if len(high_quality_tracks) == 0:
            return self.FAILURE_CATEGORIES['LOW_QUALITY']
        
        return "Unknown failure"
    
    def _calculate_track_quality(self, track_data):
        """트랙 품질 점수 계산"""
        if not track_data:
            return 0.0
        
        # 평균 키포인트 신뢰도로 품질 평가
        scores = []
        for frame_data in track_data.values():
            if 'scores' in frame_data:
                scores.extend(frame_data['scores'])
        
        return np.mean(scores) if scores else 0.0


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

def find_video_files(input_path):
    """비디오 파일 찾기"""
    if os.path.isfile(input_path):
        return [input_path]
    
    video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(input_path, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)


def get_video_extension(video_path):
    """비디오 파일의 확장자 반환"""
    return os.path.splitext(video_path)[1].lower()


def get_output_path(video_path, input_root, output_root, extension):
    """출력 경로 생성"""
    input_basename = os.path.basename(input_root.rstrip('/'))
    
    abs_video_path = os.path.abspath(video_path)
    abs_input_root = os.path.abspath(input_root)
    
    rel_path = os.path.relpath(abs_video_path, abs_input_root)
    rel_path_with_base = os.path.join(input_basename, rel_path)
    
    base_name = os.path.splitext(rel_path_with_base)[0]
    output_file = base_name + extension
    output_path = os.path.join(output_root, output_file)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return output_path


def create_overlay_video_streaming(video_path, input_root, output_root, pose_results, fps, width, height, pose_model, top_track_ids=None):
    """메모리 효율적인 스트리밍 방식 오버레이 비디오 생성"""
    """메모리 효율적인 스트리밍 방식 오버레이 비디오 생성"""
    try:
        # 비디오 확장자 확인
        input_ext = get_video_extension(video_path)
        
        # 출력 파일 경로 생성
        if input_ext == '.mp4':
            output_ext = '_enhanced_rtmo_overlay.mp4'
        elif input_ext == '.avi':
            output_ext = '_enhanced_rtmo_overlay.avi'
        elif input_ext == '.mov':
            output_ext = '_enhanced_rtmo_overlay.mov'
        else:
            output_ext = '_enhanced_rtmo_overlay.avi'  # 기본값
        
        video_output_path = get_output_path(video_path, input_root, output_root, output_ext)
        
        # 비디오 라이터 설정
        _, out_writer = setup_video_writer(video_output_path, input_ext, fps, width, height)
        if out_writer is None:
            print(f"Failed to create video writer for {video_output_path}")
            return False
        
        # Visualizer 초기화
        from mmpose.registry import VISUALIZERS
        visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
        visualizer.set_dataset_meta(pose_model.dataset_meta)
        
        # 원본 비디오 다시 읽기 (스트리밍)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return False
        
        print(f"Creating overlay video: {video_output_path}")
        vis_pbar = tqdm(total=len(pose_results), desc="Creating overlay video")
        
        # 각 프레임을 스트리밍으로 처리
        for idx, pose_result in enumerate(pose_results):
            success, frame = cap.read()
            if not success:
                print(f"Warning: Could not read frame {idx}")
                break
                
            try:
                # 포즈 시각화 (라벨 비활성화)
                visualizer.add_datasample(
                    'result',
                    frame,
                    data_sample=pose_result,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=False,  # 바운딩 박스와 라벨 비활성화
                    show_kpt_idx=False,
                    skeleton_style='mmpose'
                )
                
                # 시각화된 프레임 가져오기
                vis_frame = visualizer.get_image()
                
                # 프레임 크기 조정 (필요시)
                if vis_frame.shape[:2] != (height, width):
                    vis_frame = cv2.resize(vis_frame, (width, height))
                
                # TrackID 추가 표시 (상위 트랙 강조)
                vis_frame = draw_track_ids(vis_frame, pose_result, top_track_ids)
                
                # 비디오에 프레임 작성
                out_writer.write(vis_frame)
                
            except Exception as e:
                print(f"Warning: Failed to process frame {idx}: {e}")
                # 오리지널 프레임 사용
                out_writer.write(frame)
            
            vis_pbar.update(1)
        
        # 리소스 정리
        vis_pbar.close()
        out_writer.release()
        cap.release()
        
        print(f"Overlay video saved: {video_output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating overlay video: {e}")
        return False


def create_overlay_video(video_path, input_root, output_root, pose_results, frames, fps, width, height, pose_model, top_track_ids=None):
    """오버레이 비디오 생성"""
    """오버레이 비디오 생성"""
    try:
        # 비디오 확장자 확인
        input_ext = get_video_extension(video_path)
        
        # 출력 파일 경로 생성
        if input_ext == '.mp4':
            output_ext = '_enhanced_rtmo_overlay.mp4'
        elif input_ext == '.avi':
            output_ext = '_enhanced_rtmo_overlay.avi'
        elif input_ext == '.mov':
            output_ext = '_enhanced_rtmo_overlay.mov'
        else:
            output_ext = '_enhanced_rtmo_overlay.avi'  # 기본값
        
        video_output_path = get_output_path(video_path, input_root, output_root, output_ext)
        
        # 비디오 라이터 설정
        _, out_writer = setup_video_writer(video_output_path, input_ext, fps, width, height)
        if out_writer is None:
            print(f"Failed to create video writer for {video_output_path}")
            return False
        
        # Visualizer 초기화
        from mmpose.registry import VISUALIZERS
        visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
        visualizer.set_dataset_meta(pose_model.dataset_meta)
        
        print(f"Creating overlay video: {video_output_path}")
        vis_pbar = tqdm(total=len(pose_results), desc="Creating overlay video")
        
        # 각 프레임에 포즈 오버레이
        for idx, (pose_result, frame) in enumerate(zip(pose_results, frames)):
            try:
                # 포즈 시각화 (라벨 비활성화)
                visualizer.add_datasample(
                    'result',
                    frame,
                    data_sample=pose_result,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=False,  # 바운딩 박스와 라벨 비활성화
                    show_kpt_idx=False,
                    skeleton_style='mmpose'
                )
                
                # 시각화된 프레임 가져오기
                vis_frame = visualizer.get_image()
                
                # 프레임 크기 조정 (필요시)
                if vis_frame.shape[:2] != (height, width):
                    vis_frame = cv2.resize(vis_frame, (width, height))
                
                # TrackID 추가 표시 (상위 트랙 강조)
                vis_frame = draw_track_ids(vis_frame, pose_result, top_track_ids)
                
                # 비디오에 프레임 작성
                out_writer.write(vis_frame)
                
            except Exception as e:
                print(f"Warning: Failed to process frame {idx}: {e}")
                # 오리지널 프레임 사용
                out_writer.write(frame)
            
            vis_pbar.update(1)
        
        # 리소스 정리
        vis_pbar.close()
        out_writer.release()
        
        print(f"Overlay video saved: {video_output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating overlay video: {e}")
        return False


def draw_track_ids(frame, pose_result, top_track_ids: Optional[List[int]] = None):
    """프레임에 TrackID와 정렬번호 표시"""
    try:
        if hasattr(pose_result, 'pred_instances') and hasattr(pose_result.pred_instances, 'track_ids'):
            track_ids = pose_result.pred_instances.track_ids
            keypoints = pose_result.pred_instances.keypoints
            
            # 현재 프레임의 모든 track_id를 수집하고 정렬
            current_track_ids = []
            for i, track_id in enumerate(track_ids):
                # track_id가 numpy array인 경우 스칼라 값으로 추출
                if isinstance(track_id, np.ndarray):
                    track_id = track_id.item() if track_id.size == 1 else track_id[0]
                # 정수형으로 변환
                track_id = int(track_id) if track_id is not None else -1
                if track_id is not None and track_id >= 0:
                    current_track_ids.append(track_id)
            
            # track_id를 오름차순으로 정렬
            sorted_track_ids = sorted(current_track_ids)
            
            for i, track_id in enumerate(track_ids):
                # track_id가 numpy array인 경우 스칼라 값으로 추출
                if isinstance(track_id, np.ndarray):
                    track_id = track_id.item() if track_id.size == 1 else track_id[0]
                # 정수형으로 변환
                track_id = int(track_id) if track_id is not None else -1
                if track_id is not None and track_id >= 0:
                    if len(keypoints[i]) > 0:
                        # 키포인트 배열 형태 확인
                        kpts = keypoints[i]
                        if len(kpts.shape) == 2 and kpts.shape[1] >= 2:
                            # 신뢰도 점수가 있는 경우 (shape: [N, 3])
                            if kpts.shape[1] >= 3:
                                valid_kpts = kpts[kpts[:, 2] > 0.5]
                            else:
                                # 신뢰도 점수가 없는 경우 (shape: [N, 2]) - 모든 키포인트 사용
                                valid_kpts = kpts[~np.isnan(kpts[:, 0]) & ~np.isnan(kpts[:, 1])]
                            
                            if len(valid_kpts) > 0:
                                head_kpt = valid_kpts[np.argmin(valid_kpts[:, 1])]
                                x, y = int(head_kpt[0]), int(head_kpt[1])
                            
                                # 정렬번호 계산 (현재 프레임에서의 순위)
                                rank_number = sorted_track_ids.index(track_id) + 1  # 1부터 시작
                                
                                # 텍스트 형식: "ID: tracker_id, 정렬번호"
                                text = f"ID: {int(track_id)}, {rank_number}"
                                
                                # 폰트 설정 - 작고 선명하게
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.4  # 0.6에서 0.4로 축소
                                thickness = 1  # 2에서 1로 축소
                                
                                # 단일 배경색 사용 (기본 배경색)
                                bg_color = (50, 50, 50)  # BGR: 회색 배경
                                text_color = (255, 255, 255)  # 흰색 텍스트

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
        return frame


def setup_video_writer(output_path, input_ext, fps, width, height):
    """비디오 라이터 설정"""
    try:
        if input_ext == '.mp4':
            # MP4 포맷
            codecs_to_try = ['H264', 'avc1', 'mp4v']
        elif input_ext == '.avi':
            # AVI 포맷
            codecs_to_try = ['XVID', 'MJPG']
        elif input_ext == '.mov':
            # MOV 포맷
            codecs_to_try = ['mp4v', 'H264']
        else:
            # 기본 AVI
            codecs_to_try = ['XVID', 'MJPG']
        
        # 코덱 순서대로 시도
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
                
                # 테스트 프레임으로 검증
                test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                if out_writer.write(test_frame):
                    return fourcc, out_writer
                else:
                    out_writer.release()
                    
            except Exception:
                continue
        
        print(f"Warning: All codecs failed for {input_ext}, trying fallback")
        # 최종 fallback
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        return fourcc, out_writer
        
    except Exception as e:
        print(f"Failed to setup video writer: {e}")
        return None, None


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_video(video_path, args, failure_logger):
    """단일 비디오 처리"""
    try:
        print(f"\nProcessing: {video_path}")
        
        # CUDA 메모리 정리 (멀티프로세싱 환경에서 안정성 향상)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # ByteTracker 초기화
        tracker = ByteTracker(
            high_thresh=args.track_high_thresh,
            low_thresh=args.track_low_thresh,
            max_disappeared=args.track_max_disappeared,
            min_hits=args.track_min_hits
        )
        
        # 모델 초기화
        pose_model = init_model(args.config, args.checkpoint, device=args.device)
        
        # 모델 설정 적용
        if hasattr(pose_model.cfg, 'model'):
            if hasattr(pose_model.cfg.model, 'test_cfg'):
                pose_model.cfg.model.test_cfg.score_thr = args.score_thr
                pose_model.cfg.model.test_cfg.nms_thr = args.nms_thr
            else:
                pose_model.cfg.model.test_cfg = dict(score_thr=args.score_thr, nms_thr=args.nms_thr)
        
        if hasattr(pose_model, 'head') and hasattr(pose_model.head, 'test_cfg'):
            pose_model.head.test_cfg.score_thr = args.score_thr
            pose_model.head.test_cfg.nms_thr = args.nms_thr
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            failure_logger.log_failure(video_path, "Cannot open video file")
            return False
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        pose_results = []
        save_overlay = getattr(args, 'save_overlayfile', False)
        
        # Step 1: 포즈 추정 및 트래킹
        print(f"Running pose estimation and tracking... (save_overlay={save_overlay})")
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 포즈 추정
            batch_pose_results = inference_bottomup(pose_model, frame)
            pose_result = batch_pose_results[0]
            
            # Detection 결과 생성
            detections = create_detection_results(pose_result)
            
            # ByteTrack으로 트래킹 수행
            active_tracks = tracker.update(detections)
            
            # 포즈 결과에 트래킹 ID 할당
            pose_result = assign_track_ids_from_bytetrack(pose_result, active_tracks)
            
            pose_results.append(pose_result)
            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        
        if not pose_results:
            failure_logger.log_failure(video_path, "No pose results generated")
            return False
        
        # Step 2: 개선된 어노테이션 생성
        print("Creating enhanced annotation...")
        annotation, status_message = create_enhanced_annotation(
            pose_results, video_path, pose_model,
            min_track_length=args.min_track_length,
            quality_threshold=args.quality_threshold,
            weights=args.weights  # 가중치 전달
        )

        
        if annotation is None:
            failure_logger.log_failure(video_path, status_message)
            return False
        
        # Step 3: 결과 저장
        if os.path.isfile(args.input):
            input_root = os.path.dirname(args.input)
        else:
            input_root = args.input
        
        pkl_output_path = get_output_path(
            video_path, input_root, args.output_root, 
            '_enhanced_stgcn_annotation.pkl'
        )
        
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(annotation, f)
        
        print(f'Enhanced annotation saved: {pkl_output_path}')
        print(f'Total persons: {annotation["total_persons"]}')
        print(f'Quality threshold: {annotation["quality_threshold"]}')
        
        # Step 4: 오버레이 비디오 생성 (옵션) - 스트리밍 방식
        save_overlay = getattr(args, 'save_overlayfile', False)
        print(f"Debug: save_overlayfile = {save_overlay}")
        
        if save_overlay:
            print("Creating overlay visualization video...")
            
            # 상위 num_person개 트랙 ID 추출 (복합 점수 기준 정렬)
            num_person = getattr(args, 'num_person', 2)  # 기본값: 2명
            top_track_ids = []
            
            if 'keypoint' in annotation and annotation['keypoint']:
                # 트랙별 평균 복합점수 계산
                track_scores = {}
                for frame_data in annotation['keypoint']:
                    if isinstance(frame_data, list) and len(frame_data) > 0:
                        for person_data in frame_data:
                            if len(person_data) >= 3:  # [keypoints, scores, track_id] 구조 확인
                                track_id = person_data[2] if len(person_data) > 2 else -1
                                # track_id가 numpy array인 경우 스칼라 값으로 추출
                                if isinstance(track_id, np.ndarray):
                                    track_id = track_id.item() if track_id.size == 1 else track_id[0]
                                # 정수형으로 변환
                                track_id = int(track_id) if track_id is not None else -1
                                if track_id >= 0:
                                    # 복합 점수는 보통 추가 정보로 저장되거나 점수에서 계산
                                    # 여기서는 keypoint 점수의 평균을 사용
                                    if len(person_data) > 1 and person_data[1] is not None:
                                        avg_score = np.mean(person_data[1]) if isinstance(person_data[1], np.ndarray) else 0
                                        if track_id not in track_scores:
                                            track_scores[track_id] = []
                                        track_scores[track_id].append(avg_score)
                
                # 트랙별 평균 점수 계산 및 상위 선택
                if track_scores:
                    track_avg_scores = {tid: np.mean(scores) for tid, scores in track_scores.items()}
                    # 점수 높은 순으로 정렬하여 상위 num_person개 선택
                    sorted_tracks = sorted(track_avg_scores.items(), key=lambda x: x[1], reverse=True)
                    top_track_ids = [tid for tid, _ in sorted_tracks[:num_person]]
                    print(f"Top {num_person} track IDs selected: {top_track_ids}")
            
            success = create_overlay_video_streaming(
                video_path, input_root, args.output_root,
                pose_results, video_fps, width, height, pose_model, top_track_ids
            )
            if success:
                print("Overlay video creation completed successfully")
            else:
                print("Warning: Overlay video creation failed")
        else:
            print("Overlay video generation skipped (save_overlayfile=False)")
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        failure_logger.log_failure(video_path, f"Processing error: {str(e)}")
        return False


def parse_args():
    """인자 파싱"""
    parser = ArgumentParser(description='Enhanced RTMO pose estimation with advanced scoring')
    parser.add_argument('config', help='RTMO config file')
    parser.add_argument('checkpoint', help='RTMO checkpoint file')
    parser.add_argument('--input', type=str,
                       default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000',
                       help='Video file path or directory')
    parser.add_argument('--output-root', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output', help='Output directory')
    parser.add_argument('--save-overlayfile', action='store_true', default=True,
                       help='Save overlay visualization video file')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Detection score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.35, help='NMS threshold')
    
    # Enhanced parameters
    parser.add_argument('--min-track-length', type=int, default=10, 
                       help='Minimum track length for inclusion')
    parser.add_argument('--quality-threshold', type=float, default=0.3, 
                       help='Minimum track quality threshold')
    parser.add_argument('--num-person', type=int, default=2,
                       help='Number of top-ranked persons to highlight in overlay (default: 2)')
    
    # ByteTrack parameters
    parser.add_argument('--track-high-thresh', type=float, default=0.6)
    parser.add_argument('--track-low-thresh', type=float, default=0.1)
    parser.add_argument('--track-max-disappeared', type=int, default=30)
    parser.add_argument('--track-min-hits', type=int, default=3)
    
    # Performance parameters
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    
    # Composite score weights
    parser.add_argument('--weights', type=float, nargs=5, 
                       default=[0.30, 0.35, 0.20, 0.10, 0.05],
                       metavar=('W_MV', 'W_POS', 'W_INT', 'W_CONS', 'W_PERS'),
                       help='Weights for composite score in order: movement, position, interaction, temporal_consistency, persistence')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
    
    # 실패 로그 초기화
    failure_log_path = os.path.join(args.output_root, 'enhanced_failed_videos.txt')
    failure_logger = FailureLogger(failure_log_path)
    
    # 비디오 파일 목록 수집
    video_files = find_video_files(args.input)
    if not video_files:
        print(f"No video files found in {args.input}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Enhanced features enabled:")
    print(f"  - 5-region position scoring")
    print(f"  - Adaptive region weight learning") 
    print(f"  - Composite scoring system")
    print(f"  - Advanced interpolation")
    print(f"  - Quality-based filtering")
    print(f"  - Failure case logging")
    
    # 순차 처리 (병렬 처리는 다음 구현에서)
    success_count = 0
    for video_idx, video_path in enumerate(video_files):
        print(f"\n=== Processing video {video_idx + 1}/{len(video_files)} ===")
        
        success = process_single_video(video_path, args, failure_logger)
        if success:
            success_count += 1
        
        print(f"Progress: {video_idx + 1}/{len(video_files)} ({success_count} successful)")
    
    print(f"\n=== Processing Complete ===")
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(video_files) - success_count}")
    print(f"Failure log: {failure_log_path}")


if __name__ == '__main__':
    main()