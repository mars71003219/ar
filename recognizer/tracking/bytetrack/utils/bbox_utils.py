"""
바운딩 박스 변환 유틸리티

ByteTracker에서 사용하는 바운딩 박스 형식 변환 함수들입니다.
"""

import numpy as np
from typing import Union


def convert_bbox_to_z(bbox: Union[list, np.ndarray]) -> np.ndarray:
    """
    [x1, y1, x2, y2] 형태의 바운딩 박스를 [center_x, center_y, aspect_ratio, height] 형태로 변환
    
    Args:
        bbox: [x1, y1, x2, y2] 형태의 바운딩 박스
        
    Returns:
        [center_x, center_y, aspect_ratio, height] 형태의 배열
    """
    bbox = np.array(bbox, dtype=np.float32)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h    # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, r, h], dtype=np.float32)


def convert_x_to_bbox(x: np.ndarray, score: float = None) -> np.ndarray:
    """
    [center_x, center_y, aspect_ratio, height] 형태를 [x1, y1, x2, y2] 형태로 변환
    
    Args:
        x: [center_x, center_y, aspect_ratio, height] 형태의 배열
        score: 선택적 점수 (사용하지 않음)
        
    Returns:
        [x1, y1, x2, y2] 형태의 배열
    """
    w = np.sqrt(x[2] * x[3])
    h = x[3] / w
    return np.array([
        x[0] - w / 2.,  # x1
        x[1] - h / 2.,  # y1  
        x[0] + w / 2.,  # x2
        x[1] + h / 2.   # y2
    ], dtype=np.float32)


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    두 트랙 리스트 간의 IoU 거리 행렬을 계산합니다. (벡터화 개선)
    
    Args:
        atracks: 첫 번째 트랙 리스트
        btracks: 두 번째 트랙 리스트
        
    Returns:
        IoU 거리 행렬 (1 - IoU)
    """
    if len(atracks) * len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    # 안전한 tlbr 접근
    atlbrs = []
    for track in atracks:
        try:
            tlbr = track.tlbr
            if tlbr is not None and len(tlbr) >= 4:
                atlbrs.append(tlbr)
            else:
                atlbrs.append(np.zeros(4, dtype=np.float32))
        except (AttributeError, IndexError):
            atlbrs.append(np.zeros(4, dtype=np.float32))
    
    btlbrs = []
    for track in btracks:
        try:
            tlbr = track.tlbr
            if tlbr is not None and len(tlbr) >= 4:
                btlbrs.append(tlbr)
            else:
                btlbrs.append(np.zeros(4, dtype=np.float32))
        except (AttributeError, IndexError):
            btlbrs.append(np.zeros(4, dtype=np.float32))
    
    atlbrs = np.asarray(atlbrs, dtype=np.float32)
    btlbrs = np.asarray(btlbrs, dtype=np.float32)
    
    # 벡터화된 IoU 계산
    ious = calculate_ious_vectorized(atlbrs, btlbrs)
    
    return 1 - ious  # 거리는 1 - IoU


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    두 바운딩 박스 간의 IoU를 계산합니다.
    
    Args:
        bbox1: [x1, y1, x2, y2] 형태의 바운딩 박스
        bbox2: [x1, y1, x2, y2] 형태의 바운딩 박스
        
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    # 교집합 영역 계산
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 각 박스의 면적
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # 합집합 면적
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_ious_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    두 바운딩 박스 배열 간의 IoU를 벡터화하여 계산합니다.
    
    Args:
        boxes1: [N, 4] 형태의 바운딩 박스 배열
        boxes2: [M, 4] 형태의 바운딩 박스 배열
        
    Returns:
        [N, M] 형태의 IoU 행렬
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # boxes1을 [N, 1, 4]로, boxes2를 [1, M, 4]로 확장
    boxes1_expanded = boxes1[:, np.newaxis, :]  # [N, 1, 4]
    boxes2_expanded = boxes2[np.newaxis, :, :]  # [1, M, 4]
    
    # 교집합 계산
    x1 = np.maximum(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    y1 = np.maximum(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    x2 = np.minimum(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    y2 = np.minimum(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
    
    # 교집합 면적
    intersection_width = np.maximum(0.0, x2 - x1)
    intersection_height = np.maximum(0.0, y2 - y1)
    intersection = intersection_width * intersection_height
    
    # 각 박스의 면적
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    
    # 브로드캐스팅을 위해 확장
    area1_expanded = area1[:, np.newaxis]  # [N, 1]
    area2_expanded = area2[np.newaxis, :]  # [1, M]
    
    # 합집합 면적
    union = area1_expanded + area2_expanded - intersection
    
    # IoU 계산 (0으로 나누기 방지)
    ious = np.where(union > 1e-6, intersection / union, 0.0)
    
    return ious


def normalize_keypoints(keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    키포인트를 바운딩 박스에 대해 정규화합니다.
    
    Args:
        keypoints: [N, 3] 형태의 키포인트 (x, y, visibility)
        bbox: [x1, y1, x2, y2] 형태의 바운딩 박스
        
    Returns:
        정규화된 키포인트
    """
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((17, 3))
    
    keypoints = np.array(keypoints, dtype=np.float32)
    
    # 바운딩 박스 정보
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # 키포인트 정규화 (바운딩 박스 기준으로 0~1 범위로)
    normalized_kpts = keypoints.copy()
    normalized_kpts[:, 0] = (keypoints[:, 0] - x1) / (width + 1e-6)
    normalized_kpts[:, 1] = (keypoints[:, 1] - y1) / (height + 1e-6)
    
    return normalized_kpts


def denormalize_keypoints(normalized_keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    정규화된 키포인트를 원래 좌표계로 변환합니다.
    
    Args:
        normalized_keypoints: 정규화된 키포인트
        bbox: [x1, y1, x2, y2] 형태의 바운딩 박스
        
    Returns:
        원래 좌표계의 키포인트
    """
    if normalized_keypoints is None or len(normalized_keypoints) == 0:
        return np.zeros((17, 3))
    
    keypoints = np.array(normalized_keypoints, dtype=np.float32)
    
    # 바운딩 박스 정보
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # 키포인트 비정규화
    denormalized_kpts = keypoints.copy()
    denormalized_kpts[:, 0] = keypoints[:, 0] * width + x1
    denormalized_kpts[:, 1] = keypoints[:, 1] * height + y1
    
    return denormalized_kpts