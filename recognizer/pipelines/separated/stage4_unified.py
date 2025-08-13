"""
Stage 4: STGCN 훈련용 통합 데이터셋 생성
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from ...utils.file_utils import ensure_directory
from .data_structures import STGCNData, StageResult
from .stage3_classification import load_stage3_result


def process_stage4_unified_dataset(
    stage3_results: List[str],
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    quality_filter: bool = True,
    min_confidence: float = 0.5
) -> StageResult:
    """
    Stage 4: 여러 비디오의 Stage 3 결과를 STGCN 훈련용 통합 데이터셋으로 변환
    
    Args:
        stage3_results: Stage 3 결과 PKL 파일 리스트
        output_dir: 출력 디렉토리
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율  
        test_ratio: 테스트 데이터 비율
        quality_filter: 품질 필터링 적용 여부
        min_confidence: 최소 신뢰도
        
    Returns:
        Stage 4 처리 결과
    """
    output_path = Path(output_dir)
    ensure_directory(output_path)
    
    logging.info(f"Stage 4: Creating unified STGCN dataset from {len(stage3_results)} videos")
    
    # 모든 분류 결과 수집
    all_stgcn_data = []
    label_counts = defaultdict(int)
    
    for pkl_path in stage3_results:
        video_name = Path(pkl_path).stem.replace('_stage3_classification', '')
        
        try:
            frames, classification_results = load_stage3_result(pkl_path)
            
            for result in classification_results:
                # 품질 필터링
                if quality_filter:
                    composite_score = result.metadata.get('composite_score', 0)
                    if composite_score < min_confidence:
                        continue
                
                # WindowAnnotation을 STGCNData로 변환
                if hasattr(result, 'window_annotation') and result.window_annotation:
                    stgcn_data = STGCNData.from_window_annotation(
                        result.window_annotation, video_name
                    )
                    stgcn_data.label = result.predicted_class
                    stgcn_data.confidence = result.confidence
                    stgcn_data.quality_score = composite_score
                    
                    all_stgcn_data.append(stgcn_data)
                    label_counts[result.predicted_class] += 1
                    
        except Exception as e:
            logging.error(f"Failed to process {pkl_path}: {e}")
            continue
    
    if not all_stgcn_data:
        raise ValueError("No valid STGCN data found")
    
    # 데이터셋 분할
    train_data, val_data, test_data = _split_dataset(
        all_stgcn_data, train_ratio, val_ratio, test_ratio
    )
    
    # STGCN 호환 형식으로 변환 및 저장
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    saved_files = {}
    for split_name, data in datasets.items():
        if not data:
            continue
            
        # STGCN 형식으로 변환
        stgcn_format = _convert_to_stgcn_format(data)
        
        # 저장
        output_file = output_path / f"{split_name}_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(stgcn_format, f)
        
        saved_files[split_name] = str(output_file)
        logging.info(f"Saved {len(data)} samples to {output_file}")
    
    # 메타데이터 저장
    metadata = {
        'total_samples': len(all_stgcn_data),
        'label_counts': dict(label_counts),
        'split_info': {
            'train': len(train_data),
            'val': len(val_data), 
            'test': len(test_data)
        },
        'quality_filter': quality_filter,
        'min_confidence': min_confidence
    }
    
    metadata_file = output_path / "dataset_info.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    logging.info(f"Stage 4 completed: Generated STGCN dataset with {len(all_stgcn_data)} samples")
    
    return StageResult(
        stage_name="stage4_unified",
        input_path=str(stage3_results),
        output_path=str(output_path),
        processing_time=0,  # 계산 생략
        metadata=metadata
    )


def _split_dataset(
    data: List[STGCNData], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float
) -> Tuple[List[STGCNData], List[STGCNData], List[STGCNData]]:
    """데이터셋을 train/val/test로 분할 (라벨별 균등 분할)"""
    
    # 라벨별로 데이터 그룹화
    label_groups = defaultdict(list)
    for item in data:
        label_groups[item.label].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    for label, items in label_groups.items():
        # 셔플
        np.random.shuffle(items)
        
        n_total = len(items)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    return train_data, val_data, test_data


def _convert_to_stgcn_format(data: List[STGCNData]) -> Dict[str, Any]:
    """STGCN 훈련 형식으로 변환"""
    
    keypoints_list = []
    labels_list = []
    
    for item in data:
        keypoints_list.append(item.keypoints_sequence)
        labels_list.append(item.label)
    
    return {
        'keypoints': np.array(keypoints_list),  # [N, T, V, C]
        'labels': np.array(labels_list),        # [N]
        'metadata': {
            'video_names': [item.video_name for item in data],
            'confidences': [item.confidence for item in data],
            'quality_scores': [item.quality_score for item in data]
        }
    }


def load_stgcn_dataset(dataset_dir: str, split: str = 'train') -> Dict[str, Any]:
    """STGCN 데이터셋 로드"""
    dataset_path = Path(dataset_dir) / f"{split}_data.pkl"
    
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


def validate_stage4_result(output_dir: str) -> bool:
    """Stage 4 결과 유효성 검사"""
    try:
        output_path = Path(output_dir)
        
        # 필수 파일 확인
        required_files = ['train_data.pkl', 'dataset_info.pkl']
        for file_name in required_files:
            if not (output_path / file_name).exists():
                return False
        
        # 데이터 형식 확인
        train_data = load_stgcn_dataset(output_dir, 'train')
        if 'keypoints' not in train_data or 'labels' not in train_data:
            return False
        
        return True
    except Exception as e:
        logging.error(f"Stage 4 validation failed: {e}")
        return False