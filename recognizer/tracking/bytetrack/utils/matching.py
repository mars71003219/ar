"""
트래킹을 위한 데이터 어소시에이션 유틸리티

Hungarian algorithm과 IoU 기반 매칭을 구현합니다.
"""

import numpy as np
from typing import List, Tuple, Optional
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


def calculate_keypoint_center(keypoints: np.ndarray) -> Optional[np.ndarray]:
    """
    키포인트에서 중심점을 계산합니다.
    어깨(5,6)와 엉덩이(11,12) 중점의 평균을 사용합니다.
    
    Args:
        keypoints: (17, 3) 형태의 키포인트 배열 [x, y, confidence]
        
    Returns:
        중심점 [x, y] 또는 None (키포인트가 부족한 경우)
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return None
    
    # COCO 키포인트 인덱스: 어깨(5,6), 엉덩이(11,12)
    shoulder_indices = [5, 6]  # left_shoulder, right_shoulder
    hip_indices = [11, 12]    # left_hip, right_hip
    
    valid_points = []
    
    # 어깨 중점 계산
    shoulder_points = []
    for idx in shoulder_indices:
        if keypoints[idx, 2] > 0.3:  # confidence > 0.3
            shoulder_points.append(keypoints[idx, :2])
    
    if len(shoulder_points) >= 1:
        shoulder_center = np.mean(shoulder_points, axis=0)
        valid_points.append(shoulder_center)
    
    # 엉덩이 중점 계산
    hip_points = []
    for idx in hip_indices:
        if keypoints[idx, 2] > 0.3:  # confidence > 0.3
            hip_points.append(keypoints[idx, :2])
    
    if len(hip_points) >= 1:
        hip_center = np.mean(hip_points, axis=0)
        valid_points.append(hip_center)
    
    # 중심점 계산
    if len(valid_points) >= 1:
        return np.mean(valid_points, axis=0)
    
    return None


def keypoint_distance(trackers: List, detections: List) -> np.ndarray:
    """
    키포인트 기반 거리 행렬을 계산합니다.
    
    Args:
        trackers: 트래커 리스트
        detections: 검출 리스트
        
    Returns:
        거리 행렬 (낮을수록 더 가까움)
    """
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((len(trackers), len(detections)))
    
    distance_matrix = np.zeros((len(trackers), len(detections)))
    
    for t, tracker in enumerate(trackers):
        # 트래커에서 키포인트 정보 획득
        tracker_keypoints = getattr(tracker, 'keypoints', None)
        tracker_center = calculate_keypoint_center(tracker_keypoints)
        
        if tracker_center is None:
            # 키포인트가 없으면 바운딩 박스 중심 사용
            if hasattr(tracker, 'tlbr'):
                bbox = tracker.tlbr
            elif hasattr(tracker, 'bbox'):
                bbox = tracker.bbox
            else:
                continue
            tracker_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        for d, detection in enumerate(detections):
            # 검출에서 키포인트 정보 획득
            det_keypoints = getattr(detection, 'keypoints', None)
            det_center = calculate_keypoint_center(det_keypoints)
            
            if det_center is None:
                # 키포인트가 없으면 바운딩 박스 중심 사용
                if hasattr(detection, 'tlbr'):
                    bbox = detection.tlbr
                elif hasattr(detection, 'bbox'):
                    bbox = detection.bbox
                else:
                    distance_matrix[t, d] = 1000.0  # 매우 큰 거리
                    continue
                det_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            # 유클리드 거리 계산 (정규화)
            euclidean_dist = np.linalg.norm(tracker_center - det_center)
            
            # 이미지 크기로 정규화 (0-1 범위)
            # 가정: 1280x720 이미지 기준으로 정규화
            normalized_dist = euclidean_dist / np.sqrt(1280**2 + 720**2)
            distance_matrix[t, d] = normalized_dist
    
    return distance_matrix


def hybrid_distance(trackers: List, detections: List, 
                   iou_weight: float = 0.7, keypoint_weight: float = 0.3) -> np.ndarray:
    """
    IoU와 키포인트 거리를 결합한 하이브리드 거리 행렬을 계산합니다.
    
    Args:
        trackers: 트래커 리스트
        detections: 검출 리스트
        iou_weight: IoU 거리의 가중치
        keypoint_weight: 키포인트 거리의 가중치
        
    Returns:
        하이브리드 거리 행렬
    """
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((len(trackers), len(detections)))
    
    # IoU 거리 계산
    from .bbox_utils import iou_distance
    iou_dist = iou_distance(trackers, detections)
    
    # 키포인트 거리 계산
    keypoint_dist = keypoint_distance(trackers, detections)
    
    # 가중 결합
    hybrid_dist = iou_weight * iou_dist + keypoint_weight * keypoint_dist
    
    return hybrid_dist


def associate_detections_to_trackers_hybrid(detections: List, trackers: List, 
                                          iou_threshold: float = 0.3,
                                          use_hybrid: bool = True,
                                          iou_weight: float = 0.6,
                                          keypoint_weight: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    하이브리드 방식으로 검출과 트래커를 연결합니다.
    키포인트 정보가 있으면 IoU와 키포인트 거리를 모두 고려합니다.
    
    Args:
        detections: 검출 결과 리스트
        trackers: 트래커 리스트  
        iou_threshold: IoU 임계값
        use_hybrid: 하이브리드 매칭 사용 여부
        iou_weight: IoU 거리의 가중치
        keypoint_weight: 키포인트 거리의 가중치
        
    Returns:
        (matches, unmatched_dets, unmatched_trks): 매칭 결과
    """
    import logging
    # ID 안정성 개선을 위한 디버깅 로그 감소
    # logging.info(f"Hybrid matching called: det={len(detections)}, trk={len(trackers)}, use_hybrid={use_hybrid}")
    
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    # 거리 행렬 계산
    if use_hybrid:
        distance_matrix = hybrid_distance(trackers, detections, iou_weight, keypoint_weight)
        # logging.info(f"Hybrid distance matrix shape: {distance_matrix.shape}")
        # if distance_matrix.size > 0:
        #     logging.info(f"Distance matrix min/max: {distance_matrix.min():.3f}/{distance_matrix.max():.3f}")
    else:
        from .bbox_utils import iou_distance
        distance_matrix = iou_distance(trackers, detections)
    
    if distance_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(trackers))
    
    # Hungarian algorithm 적용
    matched_indices = linear_assignment(distance_matrix)
    
    # 매칭되지 않은 검출과 트래커 찾기
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[1]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[0]:
            unmatched_trackers.append(t)
    
    # 거리 임계값 확인 (ID 일관성을 위해 매우 관대한 임계값 사용)
    if use_hybrid:
        # ID 일관성을 위해 더욱 관대한 임계값 적용
        distance_threshold = (1 - iou_threshold) * 2.5  # 기존 2.0 -> 2.5로 증가
    else:
        distance_threshold = (1 - iou_threshold) * 1.5  # 순수 IoU도 더 관대하게
    # logging.info(f"Distance threshold: {distance_threshold:.3f} (iou_threshold: {iou_threshold}, hybrid: {use_hybrid})")
    
    # 임계값 이하인 매칭 제거
    matches = []
    for m in zip(matched_indices[0], matched_indices[1]):
        distance_val = distance_matrix[m[0], m[1]]
        if distance_val > distance_threshold:
            # logging.info(f"Rejecting match ({m[0]},{m[1]}): distance {distance_val:.3f} > threshold {distance_threshold:.3f}")
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            # logging.info(f"Accepting match ({m[0]},{m[1]}): distance {distance_val:.3f} <= threshold {distance_threshold:.3f}")
            matches.append(m)
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)