#!/usr/bin/env python3
"""
Bounding box utility functions for tracker
MMTracking의 bbox 유틸리티를 참고하여 구현
"""

import numpy as np
import torch


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    IoU (Intersection over Union) 계산
    
    Args:
        bbox1: [x1, y1, x2, y2] 형태의 바운딩 박스
        bbox2: [x1, y1, x2, y2] 형태의 바운딩 박스
        
    Returns:
        IoU 값 (0~1)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = bbox1_area + bbox2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_ious(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    여러 바운딩 박스들 간의 IoU 매트릭스 계산
    
    Args:
        bboxes1: shape (N, 4) - [x1, y1, x2, y2]
        bboxes2: shape (M, 4) - [x1, y1, x2, y2]
        
    Returns:
        IoU 매트릭스 shape (N, M)
    """
    ious = np.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            ious[i, j] = compute_iou(bbox1, bbox2)
    return ious


def bbox_xyxy_to_cxcyah(bboxes):
    """
    바운딩 박스를 [x1, y1, x2, y2] 에서 [cx, cy, aspect_ratio, height] 형태로 변환
    MMTracking의 구현을 참고
    
    Args:
        bboxes: tensor 또는 numpy array of shape (..., 4)
        
    Returns:
        변환된 바운딩 박스
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone()
    else:
        bboxes = bboxes.copy()
        
    if len(bboxes) == 0:
        return bboxes
    
    if isinstance(bboxes, torch.Tensor):
        x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        aspect_ratio = w / (h + 1e-6)
        
        return torch.stack([cx, cy, aspect_ratio, h], dim=-1)
    else:
        x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        aspect_ratio = w / (h + 1e-6)
        
        return np.stack([cx, cy, aspect_ratio, h], axis=-1)


def bbox_cxcyah_to_xyxy(bboxes):
    """
    바운딩 박스를 [cx, cy, aspect_ratio, height] 에서 [x1, y1, x2, y2] 형태로 변환
    MMTracking의 구현을 참고
    
    Args:
        bboxes: tensor 또는 numpy array of shape (..., 4)
        
    Returns:
        변환된 바운딩 박스
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone()
    else:
        bboxes = bboxes.copy()
        
    if len(bboxes) == 0:
        return bboxes
    
    if isinstance(bboxes, torch.Tensor):
        cx, cy, aspect_ratio, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        w = aspect_ratio * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        cx, cy, aspect_ratio, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        w = aspect_ratio * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return np.stack([x1, y1, x2, y2], axis=-1)


def bbox_area(bboxes):
    """바운딩 박스 면적 계산"""
    if isinstance(bboxes, torch.Tensor):
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        return w * h
    else:
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        return w * h


def clip_bbox(bboxes, img_shape):
    """바운딩 박스를 이미지 범위 내로 클리핑"""
    h, w = img_shape[:2]
    
    if isinstance(bboxes, torch.Tensor):
        bboxes[..., 0] = torch.clamp(bboxes[..., 0], 0, w)
        bboxes[..., 1] = torch.clamp(bboxes[..., 1], 0, h) 
        bboxes[..., 2] = torch.clamp(bboxes[..., 2], 0, w)
        bboxes[..., 3] = torch.clamp(bboxes[..., 3], 0, h)
    else:
        bboxes[..., 0] = np.clip(bboxes[..., 0], 0, w)
        bboxes[..., 1] = np.clip(bboxes[..., 1], 0, h)
        bboxes[..., 2] = np.clip(bboxes[..., 2], 0, w)
        bboxes[..., 3] = np.clip(bboxes[..., 3], 0, h)
    
    return bboxes