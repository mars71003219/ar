#!/usr/bin/env python3
"""
Annotation processing utilities
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def create_simple_annotation(persons_data: Dict, frame_idx: int, img_shape: Tuple[int, int]) -> Dict:
    """간단한 어노테이션 생성 (프레임별)"""
    annotation = {
        'frame_ind': frame_idx,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'persons': {}
    }
    
    for person_id, person_data in persons_data.items():
        annotation['persons'][person_id] = {
            'keypoint': person_data.get('keypoints', np.array([])),
            'keypoint_score': person_data.get('keypoint_scores', np.array([])),
            'bbox': person_data.get('bbox', np.array([])),
            'bbox_score': person_data.get('bbox_score', 0.0),
            'rank': person_data.get('rank', float('inf'))
        }
    
    return annotation


def extract_persons_ranking(all_tracks_data: Dict, frame_idx: int) -> List[Tuple[str, float]]:
    """프레임별 사람 객체 랭킹 추출"""
    rankings = []
    
    for person_id, track_data in all_tracks_data.items():
        if frame_idx in track_data:
            score = track_data[frame_idx].get('composite_score', 0.0)
            rankings.append((person_id, score))
    
    # 점수 기준 내림차순 정렬
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def calculate_composite_score(track_data: Dict, weights: List[float] = None) -> float:
    """복합 점수 계산"""
    if weights is None:
        weights = [0.3, 0.35, 0.2, 0.1, 0.05]  # [movement, position, interaction, temporal, persistence]
    
    scores = [
        track_data.get('movement_score', 0.0),
        track_data.get('position_score', 0.0), 
        track_data.get('interaction_score', 0.0),
        track_data.get('temporal_consistency', 0.0),
        track_data.get('persistence_score', 0.0)
    ]
    
    composite = sum(s * w for s, w in zip(scores, weights))
    return float(np.clip(composite, 0.0, 1.0))


def sort_windows_by_score(windows: List[Dict], score_key: str = 'composite_score') -> List[Dict]:
    """윈도우들을 점수 기준으로 정렬"""
    return sorted(windows, key=lambda w: w.get(score_key, 0.0), reverse=True)