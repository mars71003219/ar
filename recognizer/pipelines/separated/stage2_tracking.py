"""
Stage 2: 트래킹 및 스코어링 처리
"""

import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

from ...utils.factory import ModuleFactory
from ...utils.file_utils import ensure_directory
from ...utils.data_structure import FramePoses
from .data_structures import StageResult, VisualizationData
from .stage1_poses import load_stage1_result


def process_stage2_tracking_scoring(
    pkl_file_path: str, 
    tracking_config_dict: Dict[str, Any],
    scoring_config_dict: Dict[str, Any],
    output_dir: str,
    save_visualization: bool = True
) -> StageResult:
    """
    Stage 2: 트래킹 및 스코어링 수행
    
    Args:
        pkl_file_path: Stage 1 결과 PKL 파일
        tracking_config_dict: 트래킹 설정
        scoring_config_dict: 스코어링 설정
        output_dir: 출력 디렉토리
        save_visualization: 시각화 데이터 저장 여부
        
    Returns:
        Stage 2 처리 결과
    """
    start_time = time.time()
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    ensure_directory(output_path)
    
    pkl_path = Path(pkl_file_path)
    video_name = pkl_path.stem.replace('_stage1_poses', '')
    
    # Stage 1 결과 로드
    frame_poses_list = load_stage1_result(pkl_file_path)
    
    # 트래커 및 스코어링 모듈 생성
    tracker = ModuleFactory.create_tracker(tracking_config_dict)
    scorer = ModuleFactory.create_scorer(scoring_config_dict)
    
    logging.info(f"Stage 2: Processing tracking and scoring for {video_name}")
    
    # 트래킹 수행
    tracked_frames = []
    tracker.reset()
    
    for frame_poses in frame_poses_list:
        tracked_frame = tracker.track_frame_poses(frame_poses)
        tracked_frames.append(tracked_frame)
    
    # 스코어링 수행
    scored_frames = []
    for tracked_frame in tracked_frames:
        scored_frame = scorer.score_frame_poses(tracked_frame)
        scored_frames.append(scored_frame)
    
    # 결과 저장
    output_pkl_path = output_path / f"{video_name}_stage2_tracking.pkl"
    
    if save_visualization:
        # 시각화용 데이터 생성
        viz_data = VisualizationData(
            video_name=video_name,
            frame_data=scored_frames,
            stage_info={
                'stage': 'tracking_scoring',
                'total_frames': len(scored_frames),
                'tracking_config': tracking_config_dict,
                'scoring_config': scoring_config_dict
            },
            poses_with_tracking=scored_frames,
            tracking_info={
                'total_tracks': tracker.get_track_info().get('total_tracks', 0) if hasattr(tracker, 'get_track_info') else 0,
                'config': tracking_config_dict
            }
        )
        
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(viz_data, f)
    else:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(scored_frames, f)
    
    processing_time = time.time() - start_time
    
    logging.info(f"Stage 2 completed: {video_name} -> {output_pkl_path} ({processing_time:.2f}s)")
    
    return StageResult(
        stage_name="stage2_tracking",
        input_path=pkl_file_path,
        output_path=str(output_pkl_path),
        processing_time=processing_time,
        metadata={
            'total_frames': len(scored_frames),
            'tracking_config': tracking_config_dict,
            'scoring_config': scoring_config_dict
        }
    )


def load_stage2_result(pkl_path: str) -> List[FramePoses]:
    """Stage 2 결과 로드"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, VisualizationData):
        return data.poses_with_tracking
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected data type in {pkl_path}: {type(data)}")


def validate_stage2_result(pkl_path: str) -> bool:
    """Stage 2 결과 유효성 검사"""
    try:
        data = load_stage2_result(pkl_path)
        if not data or not isinstance(data, list):
            return False
        
        # 트래킹 정보 확인
        if data and isinstance(data[0], FramePoses):
            # 트래킹된 person_id가 있는지 확인
            if data[0].poses and data[0].poses[0].person_id is not None:
                return True
        return False
    except Exception as e:
        logging.error(f"Stage 2 validation failed for {pkl_path}: {e}")
        return False