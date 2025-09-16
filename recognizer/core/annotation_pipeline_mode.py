"""
어노테이션 파이프라인 모드

Stage 1-3를 연속으로 실행하는 파이프라인 모드
"""

import logging
from typing import Dict, Any
from pathlib import Path

from .mode_manager import BaseMode
from .annotation_modes import Stage1Mode, Stage2Mode, Stage3Mode
from utils.annotation_path_manager import create_path_manager

logger = logging.getLogger(__name__)


class AnnotationPipelineMode(BaseMode):
    """어노테이션 파이프라인 모드 (Stage 1-3 연속 실행)"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {})
    
    def execute(self) -> bool:
        """파이프라인 모드 실행"""
        if not self._validate_config(['input', 'output_dir']):
            return False
        
        # 경로 관리자 생성
        path_manager = create_path_manager(self.config)
        
        logger.info("=== ANNOTATION PIPELINE MODE START ===")
        logger.info("Executing Stage 1-3 sequentially")
        
        # 경로 정보 출력
        path_summary = path_manager.get_path_summary()
        logger.info(f"Input: {path_summary['input_path']}")
        logger.info(f"Base output: {path_summary['base_output_dir']}")
        
        # Stage별 활성화 확인
        stage_configs = {
            'stage1': self.mode_config.get('stage1', {}),
            'stage2': self.mode_config.get('stage2', {}),
            'stage3': self.mode_config.get('stage3', {})
        }
        
        enabled_stages = []
        for stage_name, stage_config in stage_configs.items():
            if stage_config.get('enabled', True):
                enabled_stages.append(stage_name)
        
        logger.info(f"Enabled stages: {enabled_stages}")
        
        # 스테이지별 실행
        results = {}

        # Stage1 실행 (멀티프로세스 지원)
        if 'stage1' in enabled_stages:
            try:
                logger.info(f"\n--- Executing STAGE1 ---")
                updated_config = self._prepare_stage_config('stage1', path_manager)
                success = self._execute_stage1_with_multiprocess(updated_config)
                results['stage1'] = success

                if success:
                    logger.info(f"✓ STAGE1 completed successfully")
                else:
                    logger.error(f"✗ STAGE1 failed")
                    logger.error("Pipeline stopped due to stage1 failure")
                    return False

            except Exception as e:
                logger.error(f"Error in stage1: {e}")
                results['stage1'] = False
                return False

        # Stage2/3 실행 (단일 프로세스)
        for stage_name in ['stage2', 'stage3']:
            if stage_name not in enabled_stages:
                continue

            try:
                logger.info(f"\n--- Executing {stage_name.upper()} ---")
                updated_config = self._prepare_stage_config(stage_name, path_manager)

                if stage_name == 'stage2':
                    stage_mode = Stage2Mode(updated_config)
                    success = stage_mode.execute()
                elif stage_name == 'stage3':
                    stage_mode = Stage3Mode(updated_config)
                    success = stage_mode.execute()

                results[stage_name] = success

                if success:
                    logger.info(f"✓ {stage_name.upper()} completed successfully")
                else:
                    logger.error(f"✗ {stage_name.upper()} failed")
                    if not self._should_continue_on_failure(stage_name):
                        logger.error("Pipeline stopped due to stage failure")
                        break

            except Exception as e:
                logger.error(f"Error in {stage_name}: {e}")
                results[stage_name] = False
                break
        
        # 결과 요약
        self._print_pipeline_summary(results, path_manager)
        
        # 모든 활성화된 스테이지가 성공했는지 확인
        success_count = sum(1 for stage in enabled_stages if results.get(stage, False))
        total_enabled = len(enabled_stages)
        
        overall_success = success_count == total_enabled
        
        if overall_success:
            logger.info("=== ANNOTATION PIPELINE COMPLETED SUCCESSFULLY ===")
        else:
            logger.error(f"=== ANNOTATION PIPELINE FAILED ({success_count}/{total_enabled} stages succeeded) ===")
        
        return overall_success
    
    def _prepare_stage_config(self, stage_name: str, path_manager) -> Dict[str, Any]:
        """스테이지별 설정 준비"""
        # 기본 설정 복사
        updated_config = self.config.copy()
        
        # 어노테이션 설정 업데이트
        annotation_config = updated_config.get('annotation', {}).copy()
        
        if stage_name == 'stage1':
            # Stage 1 설정
            stage1_config = annotation_config.get('stage1', {}).copy()
            stage1_config['input'] = str(path_manager.input_path)
            stage1_config['output_dir'] = str(path_manager.get_stage1_output_dir())
            annotation_config['stage1'] = stage1_config
            
        elif stage_name == 'stage2':
            # Stage 2 설정
            stage2_config = annotation_config.get('stage2', {}).copy()
            stage2_config['poses_dir'] = str(path_manager.get_stage2_input_dir())
            stage2_config['output_dir'] = str(path_manager.get_stage2_output_dir())
            annotation_config['stage2'] = stage2_config
            
        elif stage_name == 'stage3':
            # Stage 3 설정
            stage3_config = annotation_config.get('stage3', {}).copy()
            stage3_config['tracking_dir'] = str(path_manager.get_stage3_input_dir())
            stage3_config['output_dir'] = str(path_manager.get_stage3_output_dir())
            annotation_config['stage3'] = stage3_config
        
        updated_config['annotation'] = annotation_config
        return updated_config

    def _execute_stage1_with_multiprocess(self, config: Dict[str, Any]) -> bool:
        """Stage1 멀티프로세스 지원 실행"""
        try:
            # stage1 멀티프로세스 설정 확인
            annotation_config = config.get('annotation', {})
            stage1_config = annotation_config.get('stage1', {})
            stage1_multi_process = stage1_config.get('multi_process', {})
            should_run_multi_process = stage1_multi_process.get('enabled', False)

            if should_run_multi_process:
                logger.info("Stage1 - using multi-process mode in pipeline")
                # 멀티프로세스 실행
                from main import run_multi_process_annotation

                # 가짜 args 객체 생성 (멀티프로세스 함수 호환성)
                class FakeArgs:
                    def __init__(self):
                        self.multi_process = False  # config 기반으로 실행
                        self.num_processes = stage1_multi_process.get('num_processes', 6)
                        self.gpus = ','.join(map(str, stage1_multi_process.get('gpus', [0, 1])))
                        self.config = 'configs/config.yaml'  # 기본 설정 파일 경로

                fake_args = FakeArgs()
                return run_multi_process_annotation(config, fake_args)
            else:
                logger.info("Stage1 - using single process mode in pipeline")
                # 단일 프로세스 실행
                stage_mode = Stage1Mode(config)
                return stage_mode.execute()

        except Exception as e:
            logger.error(f"Error in stage1 execution: {e}")
            return False
    
    def _should_continue_on_failure(self, failed_stage: str) -> bool:
        """실패 시 계속 진행할지 결정"""
        # Stage 1이 실패하면 다음 스테이지는 실행할 수 없음
        if failed_stage == 'stage1':
            return False
        
        # Stage 2가 실패하면 Stage 3은 실행할 수 없음
        if failed_stage == 'stage2':
            return False
        
        # 기본적으로는 계속 진행하지 않음
        return False
    
    def _print_pipeline_summary(self, results: Dict[str, bool], path_manager):
        """파이프라인 실행 결과 요약 출력"""
        logger.info("\n=== PIPELINE EXECUTION SUMMARY ===")
        
        # 스테이지별 결과
        for stage_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{stage_name.upper()}: {status}")
        
        # 출력 경로 정보
        logger.info("\n=== OUTPUT DIRECTORIES ===")
        path_summary = path_manager.get_path_summary()
        
        if results.get('stage1', False):
            logger.info(f"Stage 1 poses: {path_summary['stage1_output']}")
        
        if results.get('stage2', False):
            logger.info(f"Stage 2 tracking: {path_summary['stage2_output']}")
        
        if results.get('stage3', False):
            logger.info(f"Stage 3 dataset: {path_summary['stage3_output']}")
        
        # 설정 폴더 정보
        logger.info("\n=== CONFIGURATION FOLDERS ===")
        config_folders = path_summary['config_folders']
        logger.info(f"Pose config: {config_folders['pose']}")
        logger.info(f"Tracking config: {config_folders['tracking']}")
        logger.info(f"Dataset config: {config_folders['dataset']}")


def should_use_pipeline_mode(config: Dict[str, Any]) -> bool:
    """파이프라인 모드 사용 여부 확인"""
    annotation_config = config.get('annotation', {})
    return annotation_config.get('pipeline_mode', False)