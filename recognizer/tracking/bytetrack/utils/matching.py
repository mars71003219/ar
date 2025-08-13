"""
트래킹을 위한 데이터 어소시에이션 유틸리티

Hungarian algorithm과 IoU 기반 매칭을 구현합니다.
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    헝가리안 알고리즘을 사용한 선형 할당
    
    Args:
        cost_matrix: 비용 행렬
        
    Returns:
        (row_indices, col_indices): 매칭된 인덱스들
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return row_indices, col_indices


def associate_detections_to_trackers(detections: List, trackers: List, 
                                   iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    검출과 트래커를 IoU 기반으로 연결합니다.
    
    Args:
        detections: 검출 결과 리스트
        trackers: 트래커 리스트  
        iou_threshold: IoU 임계값
        
    Returns:
        (matches, unmatched_dets, unmatched_trks): 매칭 결과
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    from .bbox_utils import iou_distance
    iou_matrix = iou_distance(trackers, detections)
    
    if iou_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(trackers))
    
    # Hungarian algorithm 적용
    matched_indices = linear_assignment(iou_matrix)
    
    # 매칭되지 않은 검출과 트래커 찾기
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[1]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[0]:
            unmatched_trackers.append(t)
    
    # IoU 임계값 이하인 매칭 제거
    matches = []
    for m in zip(matched_indices[0], matched_indices[1]):
        if iou_matrix[m[0], m[1]] > (1 - iou_threshold):
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m)
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def gate_cost_matrix(cost_matrix: np.ndarray, tracks: List, detections: List,
                    track_indices: List, detection_indices: List,
                    gated_cost: float = 100000.0, only_position: bool = False) -> np.ndarray:
    """
    게이팅을 통해 비용 행렬을 필터링합니다.
    
    Args:
        cost_matrix: 원본 비용 행렬
        tracks: 트랙 리스트
        detections: 검출 리스트  
        track_indices: 사용할 트랙 인덱스
        detection_indices: 사용할 검출 인덱스
        gated_cost: 게이팅된 경우의 비용
        only_position: 위치만 고려할지 여부
        
    Returns:
        게이팅이 적용된 비용 행렬
    """
    gated_cost_matrix = cost_matrix.copy()
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if track.kalman_filter is not None and track.mean is not None:
            measurements = np.array([detections[i] for i in detection_indices])
            
            # 게이팅 거리 계산
            gating_distances = track.kalman_filter.gating_distance(
                track.mean, track.covariance, measurements, only_position
            )
            
            # 임계값을 넘는 경우 높은 비용 할당 (카이제곱 분포, 95% 신뢰도)
            gating_threshold = 9.4877 if not only_position else 5.9915
            
            for col, gating_distance in enumerate(gating_distances):
                if gating_distance > gating_threshold:
                    gated_cost_matrix[row, col] = gated_cost
    
    return gated_cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: List) -> np.ndarray:
    """
    검출 점수와 비용을 융합합니다.
    
    Args:
        cost_matrix: 비용 행렬
        detections: 검출 리스트
        
    Returns:
        점수가 융합된 비용 행렬
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost