"""
어노테이션 모드들
1. Stage 1 - 포즈 추정 결과 PKL 생성
2. Stage 2 - 트래킹 및 정렬 결과 PKL 생성  
3. Stage 3 - train/val/test 통합 PKL 생성
4. 시각화 - stage별 오버레이
"""

import logging
from typing import Dict, Any
from pathlib import Path

from .mode_manager import BaseMode

logger = logging.getLogger(__name__)


class Stage1Mode(BaseMode):
    """Stage 1 - 포즈 추정 결과 생성"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('stage1', {})
    
    def execute(self) -> bool:
        """Stage 1 실행"""
        if not self._validate_config(['input', 'output_dir']):
            return False
        
        from pipelines.separated import process_stage1_pose_extraction
        import os
        from pathlib import Path
        
        input_path = self.mode_config.get('input')
        output_dir = self.mode_config.get('output_dir')
        
        logger.info(f"Stage 1: Processing poses from {input_path}")
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 경로가 파일인지 폴더인지 자동 감지
        path_obj = Path(input_path)
        
        if path_obj.is_file():
            # 단일 파일 처리
            if path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                video_files = [path_obj]
            else:
                logger.error(f"Unsupported file format: {path_obj.suffix}")
                return False
        elif path_obj.is_dir():
            # 폴더 처리
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(path_obj.glob(f"*{ext}"))
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return False
        
        processed_count = 0
        failed_count = 0
        
        for video_file in video_files:
            try:
                output_path = Path(output_dir) / f"{video_file.stem}_poses.pkl"
                result = process_stage1_pose_extraction(str(video_file), str(output_path), self.config)
                
                if result:
                    processed_count += 1
                    logger.info(f"Processed: {video_file.name}")
                else:
                    failed_count += 1
                    logger.error(f"Failed: {video_file.name}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {video_file.name}: {e}")
        
        total_videos = len(video_files)
        success = failed_count == 0
        
        logger.info(f"Stage 1 completed: {processed_count}/{total_videos} videos processed")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count} videos")
        
        return success


class Stage2Mode(BaseMode):
    """Stage 2 - 트래킹 및 정렬 결과 생성"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('stage2', {})
    
    def execute(self) -> bool:
        """Stage 2 실행"""
        if not self._validate_config(['poses_dir', 'output_dir']):
            return False
        
        from pipelines.separated import process_stage2_tracking_scoring
        from pathlib import Path
        
        poses_dir = self.mode_config.get('poses_dir')
        output_dir = self.mode_config.get('output_dir')
        
        logger.info(f"Stage 2: Processing tracking from {poses_dir}")
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # PKL 파일 찾기
        pkl_files = list(Path(poses_dir).glob("*_poses.pkl"))
        
        processed_count = 0
        failed_count = 0
        
        for pkl_file in pkl_files:
            try:
                output_path = Path(output_dir) / f"{pkl_file.stem.replace('_poses', '_tracking')}.pkl"
                result = process_stage2_tracking_scoring(str(pkl_file), str(output_path), self.config)
                
                if result:
                    processed_count += 1
                    logger.info(f"Processed: {pkl_file.name}")
                else:
                    failed_count += 1
                    logger.error(f"Failed: {pkl_file.name}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {pkl_file.name}: {e}")
        
        total_files = len(pkl_files)
        success = failed_count == 0
        
        logger.info(f"Stage 2 completed: {processed_count}/{total_files} files processed")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count} files")
        
        return success


class Stage3Mode(BaseMode):
    """Stage 3 - train/val/test 통합 데이터셋 생성"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('stage3', {})
    
    def execute(self) -> bool:
        """Stage 3 실행"""
        if not self._validate_config(['tracking_dir', 'output_dir', 'split_ratios']):
            return False
        
        from pathlib import Path
        import pickle
        import json
        import random
        
        tracking_dir = self.mode_config.get('tracking_dir')
        output_dir = self.mode_config.get('output_dir')
        split_ratios = self.mode_config.get('split_ratios')
        
        logger.info(f"Stage 3: Integrating dataset from {tracking_dir}")
        
        # 출력 디렉토리 생성
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 트래킹 결과 파일 찾기
        tracking_files = list(Path(tracking_dir).glob("*_tracking.pkl"))
        
        if not tracking_files:
            logger.error(f"No tracking files found in {tracking_dir}")
            return False
        
        # 데이터 수집
        all_data = []
        for tracking_file in tracking_files:
            try:
                with open(tracking_file, 'rb') as f:
                    data = pickle.load(f)
                    all_data.append({
                        'filename': tracking_file.name,
                        'data': data
                    })
                logger.info(f"Loaded: {tracking_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {tracking_file.name}: {e}")
        
        # 데이터 분할
        random.shuffle(all_data)
        total_count = len(all_data)
        
        train_count = int(total_count * split_ratios['train'])
        val_count = int(total_count * split_ratios['val'])
        test_count = total_count - train_count - val_count
        
        train_data = all_data[:train_count]
        val_data = all_data[train_count:train_count + val_count]
        test_data = all_data[train_count + val_count:]
        
        # 분할된 데이터 저장
        splits = [
            ('train', train_data, train_count),
            ('val', val_data, val_count),
            ('test', test_data, test_count)
        ]
        
        for split_name, split_data, count in splits:
            if count > 0:
                output_path = Path(output_dir) / f"{split_name}.pkl"
                with open(output_path, 'wb') as f:
                    pickle.dump(split_data, f)
                logger.info(f"Saved {split_name}.pkl: {count} samples")
        
        # 메타데이터 저장
        metadata = {
            'total_files': total_count,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'split_ratios': split_ratios
        }
        
        metadata_path = Path(output_dir) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Stage 3 completed: Dataset integration successful")
        logger.info(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        return True


class AnnotationVisualizeMode(BaseMode):
    """어노테이션 시각화 모드"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('visualize', {})
    
    def execute(self) -> bool:
        """어노테이션 시각화 실행"""
        if not self._validate_config(['stage', 'results_dir', 'video_dir']):
            return False
        
        stage = self.mode_config.get('stage')  # stage1, stage2, stage3
        results_dir = self.mode_config.get('results_dir')
        video_dir = self.mode_config.get('video_dir')
        save_mode = self.mode_config.get('save_mode', False)
        save_dir = self.mode_config.get('save_dir', 'annotation_overlay')
        
        from pathlib import Path
        
        logger.info(f"Annotation visualization - {stage}")
        logger.info(f"Results dir: {results_dir}")
        logger.info(f"Video dir: {video_dir}")
        
        # 단순한 구현: 파일 존재 확인만 수행
        results_path = Path(results_dir)
        video_path = Path(video_dir)
        
        if not results_path.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return False
        
        if not video_path.exists():
            logger.error(f"Video directory not found: {video_dir}")
            return False
        
        # Stage별 파일 패턴 확인
        if stage == 'stage1':
            pattern = "*_poses.pkl"
        elif stage == 'stage2':
            pattern = "*_tracking.pkl"
        elif stage == 'stage3':
            pattern = "*.pkl"
        else:
            logger.error(f"Unknown stage: {stage}")
            return False
        
        result_files = list(results_path.glob(pattern))
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(video_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(result_files)} result files")
        logger.info(f"Found {len(video_files)} video files")
        
        if len(result_files) == 0:
            logger.warning(f"No result files found with pattern {pattern}")
        
        if len(video_files) == 0:
            logger.warning("No video files found")
        
        # 시각화는 준비되어 있다고 가정 (실제 구현은 추후)
        logger.info(f"Annotation visualization completed for {stage}")
        logger.info(f"Note: Actual visualization implementation pending")
        
        return True