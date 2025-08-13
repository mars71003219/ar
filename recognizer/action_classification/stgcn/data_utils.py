#!/usr/bin/env python3
"""
Data processing utilities for STGCN
기존 rtmo_gcn_pipeline의 data_utils를 이전한 버전입니다.
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


def convert_poses_to_stgcn_format(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    포즈 어노테이션을 STGCN++ 형식으로 변환
    기존 rtmo_gcn_pipeline와 호환성을 위한 함수
    """
    stgcn_samples = []
    
    for annotation in annotations:
        try:
            if 'persons' not in annotation or not annotation['persons']:
                continue
            
            persons_data = annotation['persons']
            
            # 각 person별 데이터 추출
            all_keypoints = []
            all_scores = []
            
            for person_id, person_data in persons_data.items():
                if 'keypoint' in person_data and 'keypoint_score' in person_data:
                    keypoints = person_data['keypoint']  # (1, T, V, C)
                    scores = person_data['keypoint_score']  # (1, T, V)
                    
                    if keypoints.shape[1] > 0:  # 프레임이 있는지 확인
                        all_keypoints.append(keypoints[0])  # (T, V, C)
                        all_scores.append(scores[0])  # (T, V)
            
            if not all_keypoints:
                continue
                
            # 다중 인원 데이터 구성
            max_frames = max(kpts.shape[0] for kpts in all_keypoints)
            num_persons = len(all_keypoints)
            num_joints = all_keypoints[0].shape[1]
            num_coords = all_keypoints[0].shape[2] if all_keypoints[0].ndim > 2 else 2
            
            # 모든 person을 동일한 프레임 수로 맞춤
            aligned_keypoints = np.zeros((max_frames, num_persons, num_joints, num_coords))
            
            for p_idx, keypoints in enumerate(all_keypoints):
                # 시간 패딩 적용
                padded_kpts = apply_temporal_padding(keypoints, max_frames)
                aligned_keypoints[:, p_idx] = padded_kpts
            
            # STGCN 형식으로 변환
            stgcn_sample = convert_to_stgcn_format(aligned_keypoints, num_person=2)
            
            # 메타데이터 추가
            if 'metadata' in annotation:
                metadata = annotation['metadata']
                stgcn_sample.update({
                    'num_persons': metadata.get('num_persons', num_persons),
                    'frame_count': metadata.get('frame_count', max_frames),
                    'quality_threshold': metadata.get('quality_threshold', 0.3)
                })
            
            stgcn_samples.append(stgcn_sample)
            
        except Exception as e:
            print(f"Error converting annotation to STGCN format: {str(e)}")
            continue
    
    return stgcn_samples


def create_window_annotation(persons_data: Dict[str, Any], 
                           window_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    윈도우 어노테이션 생성
    기존 rtmo_gcn_pipeline의 create_window_annotation과 호환
    """
    annotation = {
        'persons': persons_data,
        'metadata': {
            'num_persons': len(persons_data),
            'frame_count': window_metadata.get('frame_count', 0),
            'quality_threshold': window_metadata.get('quality_threshold', 0.3),
            'window_idx': window_metadata.get('window_idx', 0),
            'start_frame': window_metadata.get('start_frame', 0),
            'end_frame': window_metadata.get('end_frame', 0)
        }
    }
    
    return annotation