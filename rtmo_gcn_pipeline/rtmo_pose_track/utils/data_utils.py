#!/usr/bin/env python3
"""
Data processing utilities
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from scipy.interpolate import interp1d


def apply_temporal_padding(keypoint_sequence: np.ndarray, target_length: int = 100) -> np.ndarray:
    """시간적 패딩 적용"""
    current_length = keypoint_sequence.shape[0]
    
    if current_length == target_length:
        return keypoint_sequence
    elif current_length > target_length:
        # 다운샘플링
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return keypoint_sequence[indices]
    else:
        # 업샘플링 (보간)
        try:
            old_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            
            interpolated = np.zeros((target_length, *keypoint_sequence.shape[1:]))
            
            for joint_idx in range(keypoint_sequence.shape[1]):
                for coord_idx in range(keypoint_sequence.shape[2]):
                    f = interp1d(old_indices, keypoint_sequence[:, joint_idx, coord_idx], 
                               kind='linear', fill_value='extrapolate')
                    interpolated[:, joint_idx, coord_idx] = f(new_indices)
            
            return interpolated
        except Exception as e:
            # 실패시 단순 패딩
            padded = np.zeros((target_length, *keypoint_sequence.shape[1:]))
            padded[:current_length] = keypoint_sequence
            padded[current_length:] = keypoint_sequence[-1]
            return padded


def convert_to_stgcn_format(keypoint_data: np.ndarray, num_person: int = 2) -> Dict[str, Any]:
    """STGCN 형식으로 변환"""
    # keypoint_data shape: (frames, persons, joints, coords)
    T, M_actual, V, C = keypoint_data.shape
    
    # num_person으로 패딩
    if M_actual < num_person:
        padded = np.zeros((T, num_person, V, C))
        padded[:, :M_actual] = keypoint_data
        keypoint_data = padded
    
    # STGCN 형식: (1, C, T, V, M)
    stgcn_data = keypoint_data.transpose(3, 0, 2, 1)  # (C, T, V, M)
    stgcn_data = np.expand_dims(stgcn_data, axis=0)   # (1, C, T, V, M)
    
    return {
        'keypoint': stgcn_data,
        'total_frames': T,
        'img_shape': (720, 1280),  # 기본값
        'original_shape': (720, 1280),
        'label': 0  # 기본값, 나중에 설정
    }


def split_dataset(data: List[Dict], train_split: float = 0.7, 
                 val_split: float = 0.2) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """데이터셋을 train/val/test로 분할"""
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def merge_stgcn_samples(samples: List[Dict]) -> List[Dict]:
    """STGCN 샘플들을 병합"""
    merged = []
    for sample in samples:
        if 'keypoint' in sample and sample['keypoint'] is not None:
            merged.append(sample)
    return merged