"""
추론 모드들
1. 분석 모드 - JSON/PKL 파일 생성
2. 실시간 모드 - 실시간 디스플레이
3. 시각화 모드 - PKL 기반 오버레이
"""

import logging
from typing import Dict, Any
from pathlib import Path

from .mode_manager import BaseMode

logger = logging.getLogger(__name__)


class AnalysisMode(BaseMode):
    """분석 모드 - JSON/PKL 파일만 생성"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('inference', {}).get('analysis', {})
    
    def execute(self) -> bool:
        """분석 실행"""
        if not self._validate_config(['input', 'output_dir']):
            return False
        
        from pipelines.analysis import BatchAnalysisProcessor
        from pathlib import Path
        
        processor = BatchAnalysisProcessor(self.config)
        
        input_path = self.mode_config.get('input')
        output_dir = self.mode_config.get('output_dir')
        
        # 경로가 파일인지 폴더인지 자동 감지
        path_obj = Path(input_path)
        
        if path_obj.is_file():
            # 단일 파일 처리 - 파일명 기반 폴더 생성
            logger.info(f"Processing single file: {input_path}")
            
            # 파일명으로 출력 폴더 생성
            video_name = path_obj.stem
            file_output_dir = Path(output_dir) / video_name
            
            logger.info(f"Creating output directory: {file_output_dir}")
            result = processor.process_file(input_path, str(file_output_dir))
        elif path_obj.is_dir():
            # 폴더 처리
            logger.info(f"Processing folder: {input_path}")
            result = processor.process_folder(input_path, output_dir)
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return False
        
        return result['success']


class RealtimeMode(BaseMode):
    """실시간 모드 - 실시간 디스플레이"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('inference', {}).get('realtime', {})
    
    def execute(self) -> bool:
        """실시간 실행 (폴더 입력 지원 및 오버레이 개선)"""
        if not self._validate_config(['input']):
            return False
        
        from pipelines.inference.pipeline import InferencePipeline
        from pathlib import Path
        
        # 파이프라인 초기화
        pipeline = InferencePipeline(self.config)
        if not pipeline.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return False
        
        input_source = self.mode_config.get('input')
        save_output = self.mode_config.get('save_output', False)
        output_dir = self.mode_config.get('output_path', 'output')
        display_width = self.mode_config.get('display_width', 1280)
        display_height = self.mode_config.get('display_height', 720)
        
        # 입력이 폴더인지 파일인지 확인
        input_path = Path(input_source)
        
        if input_path.is_dir():
            # 폴더 처리: 비디오 파일들을 찾아서 각각 처리
            logger.info(f"Processing folder: {input_source}")
            return self._process_realtime_folder(pipeline, input_path, output_dir, 
                                               display_width, display_height, save_output)
        elif input_path.is_file():
            # 단일 파일 처리
            logger.info(f"Processing single file: {input_source}")
            
            # 출력 경로 생성 (파일명_오버레이.mp4)
            if save_output:
                video_name = input_path.stem
                output_path = Path(output_dir) / f"{video_name}_overlay.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = None
            
            success = pipeline.start_realtime_display(
                input_source=input_source,
                display_width=display_width,
                display_height=display_height,
                save_output=save_output,
                output_path=str(output_path) if output_path else None
            )
            
            if success:
                logger.info("Realtime mode completed successfully")
            else:
                logger.error("Realtime mode failed")
            
            return success
        else:
            logger.error(f"Input path does not exist: {input_source}")
            return False
    
    def _process_realtime_folder(self, pipeline, input_folder: Path, output_dir: str,
                               display_width: int, display_height: int, save_output: bool) -> bool:
        """폴더 내 비디오 파일들을 실시간 모드로 처리"""
        # 비디오 파일 확장자
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        # 모든 비디오 파일 찾기 (디렉토리 구조 보존)
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_folder.glob(f"**/*{ext}"))
        
        if not video_files:
            logger.error(f"No video files found in: {input_folder}")
            return False
        
        logger.info(f"Found {len(video_files)} video files")
        
        success_count = 0
        for video_file in video_files:
            logger.info(f"Processing: {video_file.name}")
            
            # 출력 경로 생성 (디렉토리 구조 보존)
            if save_output:
                # 상대 경로 계산
                relative_path = video_file.relative_to(input_folder)
                video_name = video_file.stem
                
                # 출력 경로 생성 (구조 보존)
                output_subdir = Path(output_dir) / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{video_name}_overlay.mp4"
            else:
                output_path = None
            
            try:
                # 각 비디오 처리 전에 파이프라인 상태 초기화
                pipeline.reset_pipeline_state()
                
                success = pipeline.start_realtime_display(
                    input_source=str(video_file),
                    display_width=display_width,
                    display_height=display_height,
                    save_output=save_output,
                    output_path=str(output_path) if output_path else None
                )
                
                if success:
                    success_count += 1
                    if save_output:
                        logger.info(f"Saved overlay video: {output_path}")
                else:
                    logger.warning(f"Failed to process: {video_file.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
        
        logger.info(f"Realtime folder processing complete: {success_count}/{len(video_files)} successful")
        return success_count > 0


class VisualizeMode(BaseMode):
    """분석 모드 시각화 - PKL 기반 오버레이 (기존 PKLVisualizer 활용)"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('inference', {}).get('visualize', {})
    
    def execute(self) -> bool:
        """시각화 실행"""
        if not self._validate_config(['results_dir', 'input']):
            return False
        
        from visualization.pkl_visualizer import PKLVisualizer
        from pathlib import Path
        
        results_dir = self.mode_config.get('results_dir')
        input_path = self.mode_config.get('input')  # input 경로 사용
        save_mode = self.mode_config.get('save_mode', False)
        save_dir = self.mode_config.get('save_dir', 'overlay_output')
        
        # overlay 폴더 자동 생성
        if save_mode:
            overlay_dir = Path(save_dir) / 'overlay'
            overlay_dir.mkdir(parents=True, exist_ok=True)
            save_dir = str(overlay_dir)
        
        # input 경로가 파일인지 폴더인지 자동 감지
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return False
        
        visualizer = PKLVisualizer(self.config)
        
        if input_path_obj.is_file():
            # 단일 파일 시각화
            logger.info(f"Visualizing single file: {input_path}")
            return visualizer.visualize_single_file(
                str(input_path_obj), Path(results_dir), save_mode, save_dir
            )
        elif input_path_obj.is_dir():
            # 폴더 시각화
            logger.info(f"Visualizing folder: {input_path}")
            return visualizer.visualize_folder(
                str(input_path_obj), Path(results_dir), save_mode, save_dir
            )
        else:
            logger.error(f"Invalid input path: {input_path}")
            return False