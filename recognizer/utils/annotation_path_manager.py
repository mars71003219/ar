"""
어노테이션 경로 관리 유틸리티

입력 폴더 구조를 보존하고 설정 기반으로 폴더를 생성하는 시스템
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AnnotationPathManager:
    """어노테이션 경로 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.annotation_config = config.get('annotation', {})
        
        # 기본 경로 설정
        self.input_path = Path(self.annotation_config.get('input', ''))
        self.output_base = Path(self.annotation_config.get('output_dir', 'output'))
        
    def get_input_folder_name(self) -> str:
        """입력 폴더명 또는 파일명 추출"""
        if self.input_path.is_file():
            return self.input_path.parent.name
        else:
            return self.input_path.name
    
    def get_base_output_dir(self) -> Path:
        """기본 출력 디렉토리 (입력 구조 보존)"""
        input_folder_name = self.get_input_folder_name()
        return self.output_base / input_folder_name
    
    def _generate_config_hash(self, config_dict: Dict[str, Any], keys_to_hash: list) -> str:
        """설정 딕셔너리에서 지정된 키들의 해시 생성"""
        config_subset = {}
        for key in keys_to_hash:
            if key in config_dict:
                config_subset[key] = config_dict[key]
        
        # 딕셔너리를 정렬된 JSON 문자열로 변환 후 해시
        config_str = json.dumps(config_subset, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
    
    def _get_pose_config_folder_name(self) -> str:
        """포즈 추정 설정 기반 폴더명 생성: 모델명(사이즈)_score_nms"""
        pose_config = self.config.get('models', {}).get('pose_estimation', {})
        
        # 주요 설정들
        model_name = pose_config.get('model_name', 'unknown')
        score_threshold = pose_config.get('score_threshold', 0.2)
        nms_threshold = pose_config.get('nms_threshold', 0.65)
        
        # 체크포인트 경로에서 모델 크기 추출 (예: rtmo-m, rtmo-l)
        checkpoint_path = pose_config.get('checkpoint_path', '')
        model_size = ''
        if 'rtmo-m' in checkpoint_path:
            model_size = '-m'
        elif 'rtmo-l' in checkpoint_path:
            model_size = '-l'
        elif 'rtmo-s' in checkpoint_path:
            model_size = '-s'
        
        return f"{model_name}{model_size}_s{score_threshold}_n{nms_threshold}"
    
    def _get_tracking_config_folder_name(self) -> str:
        """트래킹 설정 기반 폴더명 생성: trackername_high_thresh_low_track_thresh"""
        tracking_config = self.config.get('models', {}).get('tracking', {})
        
        # 주요 설정들
        tracker_name = tracking_config.get('tracker_name', 'unknown')
        track_high_thresh = tracking_config.get('track_high_thresh', 0.3)
        track_low_thresh = tracking_config.get('track_low_thresh', 0.1) 
        track_thresh = tracking_config.get('track_thresh', 0.2)
        
        return f"{tracker_name}_h{track_high_thresh}_l{track_low_thresh}_t{track_thresh}"
    
    def _get_dataset_config_folder_name(self) -> str:
        """데이터셋 설정 기반 폴더명 생성: stage1+stage2 정보 모두 포함"""
        # Stage1과 Stage2 폴더명 가져오기
        pose_folder = self._get_pose_config_folder_name()
        tracking_folder = self._get_tracking_config_folder_name()
        
        # Stage3 분할 비율 정보
        stage3_config = self.annotation_config.get('stage3', {})
        split_ratios = stage3_config.get('split_ratios', {})
        train_ratio = split_ratios.get('train', 0.7)
        val_ratio = split_ratios.get('val', 0.15)
        test_ratio = split_ratios.get('test', 0.15)
        
        return f"{pose_folder}_{tracking_folder}_split{train_ratio}-{val_ratio}-{test_ratio}"
    
    def get_stage1_output_dir(self) -> Path:
        """Stage 1 출력 디렉토리"""
        base_dir = self.get_base_output_dir()
        config_folder = self._get_pose_config_folder_name()
        return base_dir / "stage1_poses" / config_folder
    
    def get_stage2_output_dir(self) -> Path:
        """Stage 2 출력 디렉토리"""
        base_dir = self.get_base_output_dir()
        config_folder = self._get_tracking_config_folder_name()
        return base_dir / "stage2_tracking" / config_folder
    
    def get_stage3_output_dir(self) -> Path:
        """Stage 3 출력 디렉토리"""
        base_dir = self.get_base_output_dir()
        config_folder = self._get_dataset_config_folder_name()
        return base_dir / "stage3_dataset" / config_folder
    
    def get_stage2_input_dir(self) -> Path:
        """Stage 2 입력 디렉토리 (Stage 1의 최신 출력)"""
        stage1_base = self.get_base_output_dir() / "stage1_poses"
        
        # 현재 설정에 맞는 폴더 사용
        config_folder = self._get_pose_config_folder_name()
        preferred_dir = stage1_base / config_folder
        
        if preferred_dir.exists():
            return preferred_dir
        
        # 없으면 가장 최근 폴더 사용
        if stage1_base.exists():
            subdirs = [d for d in stage1_base.iterdir() if d.is_dir()]
            if subdirs:
                latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
                logger.warning(f"Using latest stage1 output: {latest_dir}")
                return latest_dir
        
        # 기본 경로 반환
        return preferred_dir
    
    def get_stage3_input_dir(self) -> Path:
        """Stage 3 입력 디렉토리 (Stage 2의 최신 출력)"""
        stage2_base = self.get_base_output_dir() / "stage2_tracking"
        
        # 현재 설정에 맞는 폴더 사용
        config_folder = self._get_tracking_config_folder_name()
        preferred_dir = stage2_base / config_folder
        
        if preferred_dir.exists():
            return preferred_dir
        
        # 없으면 가장 최근 폴더 사용
        if stage2_base.exists():
            subdirs = [d for d in stage2_base.iterdir() if d.is_dir()]
            if subdirs:
                latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
                logger.warning(f"Using latest stage2 output: {latest_dir}")
                return latest_dir
        
        # 기본 경로 반환
        return preferred_dir
    
    def create_directories(self, stage: str) -> Path:
        """지정된 스테이지의 디렉토리 생성"""
        if stage == "stage1":
            output_dir = self.get_stage1_output_dir()
        elif stage == "stage2":
            output_dir = self.get_stage2_output_dir()
        elif stage == "stage3":
            output_dir = self.get_stage3_output_dir()
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created {stage} output directory: {output_dir}")
        return output_dir
    
    def get_path_summary(self) -> Dict[str, Any]:
        """경로 설정 요약 정보 반환"""
        return {
            'input_path': str(self.input_path),
            'input_folder_name': self.get_input_folder_name(),
            'base_output_dir': str(self.get_base_output_dir()),
            'stage1_output': str(self.get_stage1_output_dir()),
            'stage2_input': str(self.get_stage2_input_dir()),
            'stage2_output': str(self.get_stage2_output_dir()),
            'stage3_input': str(self.get_stage3_input_dir()),
            'stage3_output': str(self.get_stage3_output_dir()),
            'config_folders': {
                'pose': self._get_pose_config_folder_name(),
                'tracking': self._get_tracking_config_folder_name(),
                'dataset': self._get_dataset_config_folder_name()
            }
        }
    
    def save_path_info(self, stage: str, output_dir: Path):
        """경로 정보를 JSON 파일로 저장"""
        path_info = {
            'stage': stage,
            'output_dir': str(output_dir),
            'config_summary': self.get_path_summary(),
            'timestamp': str(Path().stat().st_mtime) if Path().exists() else None
        }
        
        info_file = output_dir / f"{stage}_path_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(path_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved path info: {info_file}")


def create_path_manager(config: Dict[str, Any]) -> AnnotationPathManager:
    """경로 관리자 생성 팩토리 함수"""
    return AnnotationPathManager(config)