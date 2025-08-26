"""
Stage 1: 포즈 추정 처리
"""

import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path as PathUtil

# recognizer 모듈 경로 추가
recognizer_root = PathUtil(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

from utils.factory import ModuleFactory
from utils.data_structure import FramePoses

def ensure_directory(path):
    """디렉토리 존재 확인 및 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)
from .data_structures import StageResult, VisualizationData


def process_stage1_pose_extraction(
    video_path: str, 
    pose_config_dict: Dict[str, Any], 
    output_dir: str,
    save_visualization: bool = True
) -> StageResult:
    """
    Stage 1: 비디오에서 포즈 추정 수행
    
    Args:
        video_path: 입력 비디오 경로
        pose_config_dict: 포즈 추정 설정
        output_dir: 출력 디렉토리
        save_visualization: 시각화 데이터 저장 여부
        
    Returns:
        Stage 1 처리 결과
    """
    start_time = time.time()
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    ensure_directory(output_path)
    
    video_path = Path(video_path)
    video_name = video_path.stem
    
    # 포즈 추정기 생성 (inference_mode 설정을 고려)
    pose_estimator = ModuleFactory.create_pose_estimator_from_inference_config(pose_config_dict)
    
    # 포즈 추정 수행
    logging.info(f"Stage 1: Processing pose estimation for {video_name}")
    frame_poses_list = pose_estimator.extract_video_poses(str(video_path))
    
    # 결과 저장
    pkl_path = output_path / f"{video_name}_stage1_poses.pkl"
    
    if save_visualization:
        # 원본 경로에서 라벨 정보 추출 (RWF-2000 구조)
        original_path_str = str(video_path)
        original_label = None
        if '/Fight/' in original_path_str:
            original_label = 1  # Fight
        elif '/NonFight/' in original_path_str:
            original_label = 0  # NonFight
        
        # 시각화용 데이터 생성 (원본 경로 정보 포함)
        viz_data = VisualizationData(
            video_name=video_name,
            frame_data=frame_poses_list,
            stage_info={
                'stage': 'pose_estimation',
                'total_frames': len(frame_poses_list),
                'config': pose_config_dict,
                'original_path': original_path_str,  # 원본 경로 보존
                'original_label': original_label    # 원본 라벨 보존
            },
            poses_only=frame_poses_list
        )
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(viz_data, f)
    else:
        # 기본 데이터만 저장
        with open(pkl_path, 'wb') as f:
            pickle.dump(frame_poses_list, f)
    
    processing_time = time.time() - start_time
    
    logging.info(f"Stage 1 completed: {video_name} -> {pkl_path} ({processing_time:.2f}s)")
    
    return StageResult(
        stage_name="stage1_poses",
        input_path=str(video_path),
        output_path=str(pkl_path),
        processing_time=processing_time,
        metadata={
            'total_frames': len(frame_poses_list),
            'avg_poses_per_frame': sum(len(fp.persons) for fp in frame_poses_list) / len(frame_poses_list),
            'config': pose_config_dict
        }
    )


def load_stage1_result(pkl_path: str) -> List[FramePoses]:
    """Stage 1 결과 로드"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, VisualizationData):
        return data.poses_only
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected data type in {pkl_path}: {type(data)}")


def validate_stage1_result(pkl_path: str) -> bool:
    """Stage 1 결과 유효성 검사"""
    try:
        data = load_stage1_result(pkl_path)
        if not data or not isinstance(data, list):
            return False
        
        # 첫 번째 프레임 검사
        if data and isinstance(data[0], FramePoses):
            return True
        return False
    except Exception as e:
        logging.error(f"Stage 1 validation failed for {pkl_path}: {e}")
        return False