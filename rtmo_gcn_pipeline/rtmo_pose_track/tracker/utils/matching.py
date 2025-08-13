#!/usr/bin/env python3
"""
Hungarian algorithm based matching utilities
MMTracking의 lap 라이브러리 대신 scipy 사용
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bbox_utils import compute_ious


def linear_assignment(cost_matrix: np.ndarray) -> tuple:
    """
    헝가리안 알고리즘을 사용한 선형 할당
    
    Args:
        cost_matrix: 비용 매트릭스 shape (N, M)
        
    Returns:
        (row_indices, col_indices): 매칭된 인덱스들
    """
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return row_indices, col_indices


def associate_detections_to_trackers(detections: np.ndarray, 
                                   trackers: np.ndarray,
                                   iou_threshold: float = 0.3) -> tuple:
    """
    헝가리안 알고리즘을 사용하여 detection과 tracker를 매칭
    
    Args:
        detections: shape (N, 4) - [x1, y1, x2, y2]
        trackers: shape (M, 4) - [x1, y1, x2, y2] 
        iou_threshold: IoU 임계값
        
    Returns:
        (matches, unmatched_dets, unmatched_trks): 
            - matches: shape (K, 2) - [[det_idx, trk_idx], ...]
            - unmatched_dets: 매칭되지 않은 detection 인덱스들
            - unmatched_trks: 매칭되지 않은 tracker 인덱스들
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))
    
    # IoU 매트릭스 계산
    iou_matrix = compute_ious(detections, trackers)
    
    # 비용 매트릭스 (1 - IoU)
    cost_matrix = 1 - iou_matrix
    
    # 헝가리안 알고리즘
    row_indices, col_indices = linear_assignment(cost_matrix)
    
    # IoU 임계값을 만족하는 매칭만 유효
    matches = []
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            matches.append([row, col])
    
    matches = np.array(matches).reshape(-1, 2)
    
    # 매칭되지 않은 detection과 tracker 찾기
    unmatched_dets = []
    for i in range(len(detections)):
        if i not in matches[:, 0]:
            unmatched_dets.append(i)
    
    unmatched_trks = []
    for i in range(len(trackers)):
        if i not in matches[:, 1]:
            unmatched_trks.append(i)
    
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


def weighted_iou_association(detections: np.ndarray,
                           trackers: np.ndarray, 
                           det_scores: np.ndarray,
                           iou_threshold: float = 0.3,
                           score_weight: float = 0.5) -> tuple:
    """
    Detection 점수로 가중치를 준 IoU 기반 매칭
    MMTracking ByteTracker의 weight_iou_with_det_scores 기능 구현
    
    Args:
        detections: shape (N, 4) - [x1, y1, x2, y2]
        trackers: shape (M, 4) - [x1, y1, x2, y2]
        det_scores: shape (N,) - detection 점수들
        iou_threshold: IoU 임계값
        score_weight: 점수 가중치 (0~1)
        
    Returns:
        (matches, unmatched_dets, unmatched_trks)
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))
    
    # IoU 매트릭스 계산
    iou_matrix = compute_ious(detections, trackers)
    
    # Detection 점수로 가중치 적용
    weighted_iou_matrix = iou_matrix * (1 + score_weight * det_scores[:, None])
    
    # 비용 매트릭스
    cost_matrix = 1 - weighted_iou_matrix
    
    # 헝가리안 알고리즘
    row_indices, col_indices = linear_assignment(cost_matrix)
    
    # IoU 임계값을 만족하는 매칭만 유효
    matches = []
    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            matches.append([row, col])
    
    matches = np.array(matches).reshape(-1, 2)
    
    # 매칭되지 않은 detection과 tracker 찾기
    unmatched_dets = []
    for i in range(len(detections)):
        if len(matches) == 0 or i not in matches[:, 0]:
            unmatched_dets.append(i)
    
    unmatched_trks = []
    for i in range(len(trackers)):
        if len(matches) == 0 or i not in matches[:, 1]:
            unmatched_trks.append(i)
    
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)