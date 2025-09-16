"""
분석 전용 파이프라인
실시간 로직 없이 전체 비디오를 완전히 분석하여 JSON/PKL 파일만 생성
"""

import time
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..base import BasePipeline
from utils.folder_processor import FolderVideoProcessor, should_use_folder_processing

logger = logging.getLogger(__name__)


class AnalysisPipeline(BasePipeline):
    """분석 전용 파이프라인 - JSON/PKL 파일만 생성"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.processing_mode = 'analysis'
    
    def initialize_pipeline(self) -> bool:
        """파이프라인 초기화"""
        logger.info("Initializing analysis pipeline")
        self._initialized = True
        return True
        
    def run_analysis(self, input_source: str, output_dir: str, is_folder: bool = False) -> bool:
        """분석 실행"""
        if is_folder:
            return self._run_folder_analysis(input_source, output_dir)
        else:
            return self._run_single_file_analysis(input_source, output_dir)
    
    def _run_folder_analysis(self, input_dir: str, output_dir: str) -> bool:
        """폴더 분석"""
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
            
        logger.info(f"Analysis mode - folder processing: {input_dir}")
        
        # 출력 디렉토리 사용 (이미 inference_modes.py에서 input_folder_name이 추가됨)
        base_output_dir = Path(output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Base output directory: {base_output_dir}")
        
        # config에 input_dir 설정
        if hasattr(self.config, '__dict__'):
            self.config.input_dir = input_dir
        else:
            self.config['input_dir'] = input_dir
        
        # 파이프라인 초기화
        from pipelines.dual_service import create_dual_service_pipeline
        pipeline = create_dual_service_pipeline(self.config)
        
        if not pipeline.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return False
        
        # 모든 비디오 파일 찾기 (상대 경로 포함)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # 각 비디오 파일을 개별 처리 (디렉토리 구조 보존)
        processed_count = 0
        failed_count = 0
        
        for video_file in video_files:
            try:
                # 상대 경로 계산 (input_path 기준)
                relative_path = video_file.relative_to(input_path)
                
                # 출력 디렉토리 구조 생성 (동일한 하위 구조 보존)
                video_output_dir = base_output_dir / relative_path.parent / relative_path.stem
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Processing video: {relative_path}")
                logger.info(f"Output directory: {video_output_dir}")
                
                success = self._process_single_file_with_output_dir(str(video_file), str(video_output_dir), pipeline)
                
                if success:
                    processed_count += 1
                    logger.info(f"Successfully processed: {relative_path}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to process: {relative_path}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {video_file.name}: {e}")
        
        total_videos = len(video_files)
        success_rate = (processed_count / total_videos * 100) if total_videos > 0 else 0
        
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Total videos: {total_videos}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return failed_count == 0
    
    def _run_single_file_analysis(self, input_file: str, output_dir: str) -> bool:
        """단일 파일 분석"""
        logger.info(f"Analysis mode - single file: {input_file}")
        
        # 파이프라인 초기화
        from pipelines.dual_service import create_dual_service_pipeline
        pipeline = create_dual_service_pipeline(self.config)
        
        if not pipeline.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return False
        
        # 단일 파일 분석 처리
        return self._process_single_file(input_file, output_dir, pipeline)
    
    def _process_single_file(self, input_file: str, output_dir: str, pipeline) -> bool:
        """단일 파일 분석 처리 (새로운 분석 모드 사용)"""
        logger.info(f"Starting analysis for: {input_file}")
        
        # 새로운 분석 모드 메서드 사용
        result = pipeline.process_video_analysis_mode(input_file)
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Analysis failed: {error_msg}")
            return False
        
        # 결과 저장
        from utils.result_saver import ResultSaver
        return ResultSaver.save_analysis_results(input_file, output_dir, result)
    
    def _process_single_file_with_output_dir(self, input_file: str, specific_output_dir: str, pipeline) -> bool:
        """특정 출력 디렉토리를 지정한 단일 파일 분석 처리 (새로운 분석 모드 사용)"""
        logger.info(f"Starting analysis for: {input_file}")
        
        # 새로운 분석 모드 메서드 사용
        result = pipeline.process_video_analysis_mode(input_file)
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Analysis failed: {error_msg}")
            return False
        
        # 결과 저장 (지정된 출력 디렉토리 사용)
        from utils.result_saver import ResultSaver
        return ResultSaver.save_analysis_results(input_file, specific_output_dir, result)
    
    
    
    


class BatchAnalysisProcessor:
    """배치 분석 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = AnalysisPipeline(config)
    
    def process_folder(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """폴더 배치 처리"""
        logger.info("=== BATCH ANALYSIS START ===")
        start_time = time.time()
        
        success = self.pipeline.run_analysis(input_dir, output_dir, is_folder=True)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'success': success,
            'elapsed_time': elapsed_time,
            'input_dir': input_dir,
            'output_dir': output_dir
        }
        
        logger.info(f"=== BATCH ANALYSIS COMPLETE ({elapsed_time:.1f}s) ===")
        return result
    
    def process_file(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """단일 파일 처리"""
        logger.info("=== SINGLE FILE ANALYSIS START ===")
        start_time = time.time()
        
        success = self.pipeline.run_analysis(input_file, output_dir, is_folder=False)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'success': success,
            'elapsed_time': elapsed_time,
            'input_file': input_file,
            'output_dir': output_dir
        }
        
        logger.info(f"=== SINGLE FILE ANALYSIS COMPLETE ({elapsed_time:.1f}s) ===")
        return result