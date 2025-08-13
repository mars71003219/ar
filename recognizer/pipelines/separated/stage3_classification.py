"""
Stage 3: 분류 및 복합점수 정렬 처리
"""

import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

import sys
from pathlib import Path as PathUtil

# recognizer 모듈 경로 추가
recognizer_root = PathUtil(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

from utils.factory import ModuleFactory
from utils.data_structure import FramePoses, WindowAnnotation, ClassificationResult

def ensure_directory(path):
    """디렉토리 존재 확인 및 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)
from .data_structures import StageResult, VisualizationData
from .stage2_tracking import load_stage2_result


def process_stage3_classification(
    pkl_file_path: str, 
    classification_config_dict: Dict[str, Any],
    window_size: int,
    window_stride: int,
    output_dir: str,
    save_visualization: bool = True
) -> StageResult:
    """
    Stage 3: 윈도우 기반 분류 및 복합점수 정렬
    
    Args:
        pkl_file_path: Stage 2 결과 PKL 파일
        classification_config_dict: 분류 설정
        window_size: 윈도우 크기
        window_stride: 윈도우 스트라이드
        output_dir: 출력 디렉토리
        save_visualization: 시각화 데이터 저장 여부
        
    Returns:
        Stage 3 처리 결과
    """
    start_time = time.time()
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    ensure_directory(output_path)
    
    pkl_path = Path(pkl_file_path)
    video_name = pkl_path.stem.replace('_stage2_tracking', '')
    
    # Stage 2 결과 로드
    scored_frames = load_stage2_result(pkl_file_path)
    
    # 윈도우 프로세서 및 분류기 생성
    window_processor = ModuleFactory.create_window_processor({
        'window_size': window_size,
        'window_stride': window_stride
    })
    classifier = ModuleFactory.create_classifier(classification_config_dict)
    
    logging.info(f"Stage 3: Processing classification for {video_name}")
    
    # 윈도우 처리 및 분류
    windows = window_processor.process_frames(scored_frames)
    
    classification_results = []
    for window in windows:
        result = classifier.classify_window(window)
        classification_results.append(result)
    
    # 복합점수 계산 및 정렬
    for i, result in enumerate(classification_results):
        # 포즈 품질, 트래킹 안정성, 분류 신뢰도를 종합한 복합점수
        pose_quality = _calculate_pose_quality(windows[i])
        tracking_stability = _calculate_tracking_stability(windows[i])
        classification_confidence = result.confidence
        
        # 가중 평균으로 복합점수 계산
        composite_score = (
            pose_quality * 0.3 + 
            tracking_stability * 0.3 + 
            classification_confidence * 0.4
        )
        
        # 메타데이터에 추가
        result.metadata['pose_quality'] = pose_quality
        result.metadata['tracking_stability'] = tracking_stability
        result.metadata['composite_score'] = composite_score
    
    # 복합점수로 정렬
    classification_results.sort(key=lambda x: x.metadata.get('composite_score', 0), reverse=True)
    
    # 결과 저장
    output_pkl_path = output_path / f"{video_name}_stage3_classification.pkl"
    
    if save_visualization:
        # 시각화용 데이터 생성
        viz_data = VisualizationData(
            video_name=video_name,
            frame_data=scored_frames,
            stage_info={
                'stage': 'classification_scoring',
                'total_windows': len(classification_results),
                'window_size': window_size,
                'window_stride': window_stride,
                'classification_config': classification_config_dict
            },
            poses_with_scores=scored_frames,
            classification_results=classification_results,
            scoring_info={
                'total_windows': len(classification_results),
                'avg_composite_score': sum(r.metadata.get('composite_score', 0) for r in classification_results) / len(classification_results),
                'config': classification_config_dict
            }
        )
        
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(viz_data, f)
    else:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump({
                'frames': scored_frames,
                'windows': windows,
                'classification_results': classification_results
            }, f)
    
    processing_time = time.time() - start_time
    
    logging.info(f"Stage 3 completed: {video_name} -> {output_pkl_path} ({processing_time:.2f}s)")
    
    return StageResult(
        stage_name="stage3_classification",
        input_path=pkl_file_path,
        output_path=str(output_pkl_path),
        processing_time=processing_time,
        metadata={
            'total_windows': len(classification_results),
            'avg_composite_score': sum(r.metadata.get('composite_score', 0) for r in classification_results) / len(classification_results),
            'classification_config': classification_config_dict
        }
    )


def load_stage3_result(pkl_path: str) -> tuple:
    """Stage 3 결과 로드"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, VisualizationData):
        return data.poses_with_scores, data.classification_results
    elif isinstance(data, dict):
        return data['frames'], data['classification_results']
    else:
        raise ValueError(f"Unexpected data type in {pkl_path}: {type(data)}")


def _calculate_pose_quality(window: WindowAnnotation) -> float:
    """포즈 품질 점수 계산"""
    if window.keypoints_sequence is None or len(window.keypoints_sequence) == 0:
        return 0.0
    
    # 키포인트 신뢰도 평균
    keypoints = window.keypoints_sequence
    if len(keypoints.shape) == 3:  # [T, V, C]
        confidence_scores = keypoints[:, :, 2]  # visibility/confidence
        avg_confidence = float(confidence_scores.mean())
        return min(avg_confidence, 1.0)
    
    return 0.5  # 기본값


def _calculate_tracking_stability(window: WindowAnnotation) -> float:
    """트래킹 안정성 점수 계산"""
    if window.person_id is None:
        return 0.0
    
    # 트래킹 길이와 일관성 기반으로 계산
    if hasattr(window, 'metadata') and window.metadata:
        tracking_info = window.metadata.get('tracking_info', {})
        track_length = tracking_info.get('track_length', 0)
        track_gaps = tracking_info.get('track_gaps', 0)
        
        if track_length > 0:
            stability = (track_length - track_gaps) / track_length
            return max(0.0, min(stability, 1.0))
    
    return 0.7  # 기본값


def validate_stage3_result(pkl_path: str) -> bool:
    """Stage 3 결과 유효성 검사"""
    try:
        frames, results = load_stage3_result(pkl_path)
        
        if not frames or not results:
            return False
        
        # 분류 결과 확인
        if isinstance(results[0], ClassificationResult):
            return True
        return False
    except Exception as e:
        logging.error(f"Stage 3 validation failed for {pkl_path}: {e}")
        return False