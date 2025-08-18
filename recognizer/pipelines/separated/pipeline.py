"""
분리형 파이프라인 메인 클래스 
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# # from .config import SeparatedPipelineConfig  # 통합 설정 시스템으로 대체
from .data_structures import StageResult
from .stage1_poses import process_stage1_pose_extraction, validate_stage1_result
from .stage2_tracking import process_stage2_tracking_scoring, validate_stage2_result
# from .stage3_classification import process_stage3_classification, validate_stage3_result  # 삭제된 모듈
# from .stage4_unified import process_stage4_unified_dataset, validate_stage4_result  # 삭제된 모듈

import sys

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

from utils.multiprocess_manager import MultiprocessManager

def ensure_directory(path):
    """디렉토리 존재 확인 및 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)


class SeparatedPipeline:
    """
    4단계 분리형 파이프라인
    
    각 단계별 독립 실행 가능하며, resume 기능과 시각화 데이터 생성을 지원합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, List[StageResult]] = {
            'stage1': [],
            'stage2': [],
            'stage3': [],
            'stage4': []
        }
        
        # 출력 디렉토리 생성
        for output_dir in [
            self.config.get('stage1_output_dir', 'output/stage1'),
            self.config.get('stage2_output_dir', 'output/stage2'),
            self.config.get('stage3_output_dir', 'output/stage3'),
            self.config.get('stage4_output_dir', 'output/stage4')
        ]:
            ensure_directory(output_dir)
    
    def run_full_pipeline(self, video_paths: List[str]) -> Dict[str, List[StageResult]]:
        """전체 파이프라인 실행"""
        logging.info(f"Starting separated pipeline for {len(video_paths)} videos")
        
        # Stage 1: 포즈 추정
        stages_to_run = self.config.get('stages_to_run', ['stage1', 'stage2', 'stage3', 'stage4'])
        if 'stage1' in stages_to_run:
            self._run_stage1(video_paths)
        
        # Stage 2: 트래킹 및 스코어링
        if 'stage2' in stages_to_run:
            self._run_stage2()
        
        # Stage 3: 분류 및 복합점수
        if 'stage3' in stages_to_run:
            self._run_stage3()
        
        # Stage 4: 통합 데이터셋
        if 'stage4' in stages_to_run:
            self._run_stage4()
        
        return self.results
    
    def _run_stage1(self, video_paths: List[str]):
        """Stage 1: 포즈 추정 실행"""
        logging.info("=== Stage 1: Pose Estimation ===")
        
        if self.config.enable_multiprocessing and len(video_paths) > 1:
            self._run_stage1_multiprocess(video_paths)
        else:
            self._run_stage1_sequential(video_paths)
    
    def _run_stage1_sequential(self, video_paths: List[str]):
        """Stage 1 순차 실행"""
        for video_path in video_paths:
            video_name = Path(video_path).stem
            output_file = Path(self.config.stage1_output_dir) / f"{video_name}_stage1_poses.pkl"
            
            # Resume 체크
            if self.config.enable_resume and output_file.exists():
                if validate_stage1_result(str(output_file)):
                    logging.info(f"Stage 1 skipped (resume): {video_name}")
                    continue
            
            # 처리 실행
            result = process_stage1_pose_extraction(
                video_path=video_path,
                pose_config_dict=self.config.pose_config.__dict__,
                output_dir=self.config.stage1_output_dir,
                save_visualization=self.config.save_visualizations
            )
            self.results['stage1'].append(result)
    
    def _run_stage1_multiprocess(self, video_paths: List[str]):
        """Stage 1 멀티프로세스 실행"""
        tasks = []
        for video_path in video_paths:
            video_name = Path(video_path).stem
            output_file = Path(self.config.stage1_output_dir) / f"{video_name}_stage1_poses.pkl"
            
            # Resume 체크
            if self.config.enable_resume and output_file.exists():
                if validate_stage1_result(str(output_file)):
                    continue
            
            tasks.append({
                'func': process_stage1_pose_extraction,
                'args': (
                    video_path,
                    self.config.pose_config.__dict__,
                    self.config.stage1_output_dir,
                    self.config.save_visualizations
                )
            })
        
        if tasks:
            with MultiprocessManager(num_workers=self.config.num_workers) as manager:
                results = manager.run_tasks(tasks)
                self.results['stage1'].extend(results)
    
    def _run_stage2(self):
        """Stage 2: 트래킹 및 스코어링 실행"""
        logging.info("=== Stage 2: Tracking & Scoring ===")
        
        # Stage 1 결과 파일 찾기
        stage1_files = list(Path(self.config.stage1_output_dir).glob("*_stage1_poses.pkl"))
        
        if self.config.enable_multiprocessing and len(stage1_files) > 1:
            self._run_stage2_multiprocess(stage1_files)
        else:
            self._run_stage2_sequential(stage1_files)
    
    def _run_stage2_sequential(self, stage1_files: List[Path]):
        """Stage 2 순차 실행"""
        for pkl_file in stage1_files:
            video_name = pkl_file.stem.replace('_stage1_poses', '')
            output_file = Path(self.config.stage2_output_dir) / f"{video_name}_stage2_tracking.pkl"
            
            # Resume 체크
            if self.config.enable_resume and output_file.exists():
                if validate_stage2_result(str(output_file)):
                    logging.info(f"Stage 2 skipped (resume): {video_name}")
                    continue
            
            # 처리 실행
            result = process_stage2_tracking_scoring(
                pkl_file_path=str(pkl_file),
                tracking_config_dict=self.config.tracking_config.__dict__,
                scoring_config_dict=self.config.scoring_config.__dict__,
                output_dir=self.config.stage2_output_dir,
                save_visualization=self.config.save_visualizations
            )
            self.results['stage2'].append(result)
    
    def _run_stage2_multiprocess(self, stage1_files: List[Path]):
        """Stage 2 멀티프로세스 실행"""
        tasks = []
        for pkl_file in stage1_files:
            video_name = pkl_file.stem.replace('_stage1_poses', '')
            output_file = Path(self.config.stage2_output_dir) / f"{video_name}_stage2_tracking.pkl"
            
            # Resume 체크
            if self.config.enable_resume and output_file.exists():
                if validate_stage2_result(str(output_file)):
                    continue
            
            tasks.append({
                'func': process_stage2_tracking_scoring,
                'args': (
                    str(pkl_file),
                    self.config.tracking_config.__dict__,
                    self.config.scoring_config.__dict__,
                    self.config.stage2_output_dir,
                    self.config.save_visualizations
                )
            })
        
        if tasks:
            with MultiprocessManager(num_workers=self.config.num_workers) as manager:
                results = manager.run_tasks(tasks)
                self.results['stage2'].extend(results)
    
    def _run_stage3(self):
        """Stage 3: 분류 및 복합점수 실행"""
        logging.info("=== Stage 3: Classification & Scoring ===")
        
        # Stage 2 결과 파일 찾기
        stage2_files = list(Path(self.config.stage2_output_dir).glob("*_stage2_tracking.pkl"))
        
        if self.config.enable_multiprocessing and len(stage2_files) > 1:
            self._run_stage3_multiprocess(stage2_files)
        else:
            self._run_stage3_sequential(stage2_files)
    
    def _run_stage3_sequential(self, stage2_files: List[Path]):
        """Stage 3 순차 실행"""
        for pkl_file in stage2_files:
            video_name = pkl_file.stem.replace('_stage2_tracking', '')
            output_file = Path(self.config.stage3_output_dir) / f"{video_name}_stage3_classification.pkl"
            
            # Resume 체크 (stage3 삭제됨 - 주석 처리)
            logging.warning(f"Stage 3 classification module deleted, skipping: {video_name}")
            # if self.config.enable_resume and output_file.exists():
            #     if validate_stage3_result(str(output_file)):
            #         logging.info(f"Stage 3 skipped (resume): {video_name}")
            #         continue
            # 
            # # 처리 실행
            # result = process_stage3_classification(
            #     pkl_file_path=str(pkl_file),
            #     classification_config_dict=self.config.classification_config.__dict__,
            #     window_size=self.config.window_size,
            #     window_stride=self.config.window_stride,
            #     output_dir=self.config.stage3_output_dir,
            #     save_visualization=self.config.save_visualizations
            # )
            # self.results['stage3'].append(result)
    
    def _run_stage3_multiprocess(self, stage2_files: List[Path]):
        """Stage 3 멀티프로세스 실행"""
        tasks = []
        for pkl_file in stage2_files:
            video_name = pkl_file.stem.replace('_stage2_tracking', '')
            output_file = Path(self.config.stage3_output_dir) / f"{video_name}_stage3_classification.pkl"
            
            # Resume 체크 (stage3 삭제됨 - 주석 처리)  
            logging.warning(f"Stage 3 classification module deleted, skipping: {pkl_file.stem}")
            continue
            # if self.config.enable_resume and output_file.exists():
            #     if validate_stage3_result(str(output_file)):
            #         continue
            # 
            # tasks.append({
            #     'func': process_stage3_classification,
            #     'args': (
            #         str(pkl_file),
            #         self.config.classification_config.__dict__,
            #         self.config.window_size,
            #         self.config.window_stride,
            #         self.config.stage3_output_dir,
            #         self.config.save_visualizations
            #     )
            # })
        
        # if tasks:
        #     with MultiprocessManager(num_workers=self.config.num_workers) as manager:
        #         results = manager.run_tasks(tasks)
        #         self.results['stage3'].extend(results)
    
    def _run_stage4(self):
        """Stage 4: 통합 데이터셋 생성"""
        logging.info("=== Stage 4: Unified Dataset Creation ===")
        
        # Stage 3 결과 파일 찾기
        stage3_files = list(Path(self.config.stage3_output_dir).glob("*_stage3_classification.pkl"))
        
        if not stage3_files:
            logging.warning("No Stage 3 results found for Stage 4")
            return
        
        # Resume 체크
        if self.config.enable_resume and validate_stage4_result(self.config.stage4_output_dir):
            logging.info("Stage 4 skipped (resume): unified dataset already exists")
            return
        
        # 처리 실행
        result = process_stage4_unified_dataset(
            stage3_results=[str(f) for f in stage3_files],
            output_dir=self.config.stage4_output_dir,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            quality_filter=self.config.enable_quality_filter,
            min_confidence=self.config.min_confidence
        )
        self.results['stage4'].append(result)
    
    def run_single_stage(self, stage_name: str, inputs: List[str]) -> List[StageResult]:
        """단일 스테이지 실행"""
        if stage_name == 'stage1':
            self._run_stage1(inputs)
        elif stage_name == 'stage2':
            self._run_stage2()
        elif stage_name == 'stage3':
            self._run_stage3()
        elif stage_name == 'stage4':
            self._run_stage4()
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        return self.results[stage_name]
    
    def get_stage_outputs(self, stage_name: str) -> List[str]:
        """스테이지 출력 파일 경로 반환"""
        output_dirs = {
            'stage1': self.config.stage1_output_dir,
            'stage2': self.config.stage2_output_dir,
            'stage3': self.config.stage3_output_dir,
            'stage4': self.config.stage4_output_dir
        }
        
        if stage_name not in output_dirs:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        output_dir = Path(output_dirs[stage_name])
        if stage_name == 'stage4':
            # Stage 4는 디렉토리 자체가 결과
            return [str(output_dir)] if output_dir.exists() else []
        else:
            # 다른 스테이지는 PKL 파일들
            pattern_map = {
                'stage1': "*_stage1_poses.pkl",
                'stage2': "*_stage2_tracking.pkl", 
                'stage3': "*_stage3_classification.pkl"
            }
            return [str(f) for f in output_dir.glob(pattern_map[stage_name])]