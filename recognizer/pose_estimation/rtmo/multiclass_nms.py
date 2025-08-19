#!/usr/bin/env python3
"""
Multi-class NMS 구현

rtmlib의 NMS 구현을 기반으로 한 Non-Maximum Suppression
"""

import numpy as np
from typing import Tuple, Optional


def multiclass_nms(boxes: np.ndarray, 
                   scores: np.ndarray,
                   nms_thr: float = 0.45,
                   score_thr: float = 0.1,
                   max_num: int = 1000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Multi-class NMS 적용
    
    Args:
        boxes: 바운딩 박스 배열 [N, 4] (x1, y1, x2, y2)
        scores: 신뢰도 점수 배열 [N, num_classes]
        nms_thr: NMS IoU 임계값
        score_thr: 점수 임계값
        max_num: 최대 검출 수
        
    Returns:
        dets: 필터링된 검출 결과 [K, 5] (x1, y1, x2, y2, score)
        keep: 유지된 인덱스 배열 [K] (None if no detections)
    """
    if boxes.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32), None
    
    # 점수가 임계값 이상인 박스 필터링
    if scores.ndim == 2:
        # Multi-class scores: 각 클래스별 최대 점수 사용
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
    else:
        # Single class scores
        max_scores = scores
        class_ids = np.zeros(len(scores), dtype=np.int32)
    
    valid_mask = max_scores >= score_thr
    if not np.any(valid_mask):
        return np.empty((0, 5), dtype=np.float32), None
    
    boxes = boxes[valid_mask]
    max_scores = max_scores[valid_mask]
    class_ids = class_ids[valid_mask]
    original_indices = np.where(valid_mask)[0]
    
    # 점수 내림차순 정렬
    order = np.argsort(max_scores)[::-1]
    boxes = boxes[order]
    max_scores = max_scores[order]
    class_ids = class_ids[order]
    original_indices = original_indices[order]
    
    # NMS 적용
    keep_indices = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
            
        keep_indices.append(i)
        
        if len(keep_indices) >= max_num:
            break
        
        # 현재 박스와 나머지 박스들의 IoU 계산
        current_box = boxes[i:i+1]
        remaining_boxes = boxes[i+1:]
        
        if len(remaining_boxes) == 0:
            break
            
        ious = compute_iou(current_box, remaining_boxes)
        
        # 같은 클래스이고 IoU가 임계값 이상인 박스들 억제
        for j, iou in enumerate(ious[0]):  # ious는 (1, N) 형태
            idx = i + 1 + j
            if not suppressed[idx] and class_ids[i] == class_ids[idx] and iou > nms_thr:
                suppressed[idx] = True
    
    if not keep_indices:
        return np.empty((0, 5), dtype=np.float32), None
    
    # 결과 구성
    keep_indices = np.array(keep_indices)
    final_boxes = boxes[keep_indices]
    final_scores = max_scores[keep_indices]
    
    # [x1, y1, x2, y2, score] 형식으로 결합
    dets = np.column_stack([final_boxes, final_scores])
    
    # 원본 인덱스 반환
    keep_original = original_indices[keep_indices]
    
    return dets, keep_original


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU (Intersection over Union) 계산
    
    Args:
        boxes1: 첫 번째 박스 집합 [M, 4] (x1, y1, x2, y2)
        boxes2: 두 번째 박스 집합 [N, 4] (x1, y1, x2, y2)
        
    Returns:
        ious: IoU 행렬 [M, N]
    """
    # 박스 영역 계산
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 교집합 계산을 위한 좌표
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])  # [M, N]
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])  # [M, N]
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])  # [M, N]
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])  # [M, N]
    
    # 교집합 영역
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 합집합 영역
    union = area1[:, np.newaxis] + area2 - intersection
    
    # IoU 계산 (0으로 나누기 방지)
    ious = intersection / np.maximum(union, 1e-8)
    
    return ious


def batched_nms(boxes: np.ndarray,
                scores: np.ndarray, 
                idxs: np.ndarray,
                nms_thr: float = 0.45) -> np.ndarray:
    """배치 처리 방식의 NMS
    
    Args:
        boxes: 바운딩 박스 배열 [N, 4]
        scores: 신뢰도 점수 배열 [N]
        idxs: 클래스 ID 배열 [N]
        nms_thr: NMS IoU 임계값
        
    Returns:
        keep: 유지된 인덱스 배열
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    max_coordinate = np.max(boxes)
    
    # 각 클래스의 박스를 오프셋을 주어 분리
    offsets = idxs * (max_coordinate + 1)
    offset_boxes = boxes + offsets[:, np.newaxis]
    
    # 단일 클래스 NMS 적용
    keep = nms(offset_boxes, scores, nms_thr)
    
    return keep


def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float = 0.45) -> np.ndarray:
    """단일 클래스 NMS
    
    Args:
        boxes: 바운딩 박스 배열 [N, 4] (x1, y1, x2, y2)
        scores: 신뢰도 점수 배열 [N]
        nms_thr: NMS IoU 임계값
        
    Returns:
        keep: 유지된 인덱스 배열
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    # 점수 내림차순 정렬
    order = np.argsort(scores)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # 현재 박스와 나머지 박스들의 IoU 계산
        current_box = boxes[i:i+1]
        remaining_boxes = boxes[order[1:]]
        
        ious = compute_iou(current_box, remaining_boxes)[0]
        
        # IoU가 임계값 이하인 박스들만 유지
        valid_indices = np.where(ious <= nms_thr)[0]
        order = order[valid_indices + 1]
    
    return np.array(keep, dtype=np.int64)