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
        
        # 입력 경로의 전체 구조를 출력 디렉토리에 보존
        input_folder_name = input_path.name
        base_output_dir = Path(output_dir) / input_folder_name
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Base output directory: {base_output_dir}")
        
        # config에 input_dir 설정
        if hasattr(self.config, '__dict__'):
            self.config.input_dir = input_dir
        else:
            self.config['input_dir'] = input_dir
        
        # 파이프라인 초기화
        from pipelines.inference.pipeline import InferencePipeline
        pipeline = InferencePipeline(self.config)
        
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
        from pipelines.inference.pipeline import InferencePipeline
        pipeline = InferencePipeline(self.config)
        
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
        return self._save_analysis_results_from_dict(input_file, output_dir, result)
    
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
        return self._save_analysis_results_from_dict(input_file, specific_output_dir, result)
    
    def _calculate_processing_duration(self, video_file: str) -> float:
        """비디오 길이 기반 처리 시간 계산"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                logger.warning(f"Cannot open video for duration calculation: {video_file}")
                return 60.0  # 기본값
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps <= 0 or frame_count <= 0:
                logger.warning(f"Invalid video properties: fps={fps}, frames={frame_count}")
                return 60.0
            
            # 비디오 실제 길이
            video_duration = frame_count / fps
            
            # 처리 시간: 비디오 길이 * 1.5 + 10초 버퍼 (충분한 시간 확보)
            processing_duration = video_duration * 1.5 + 10.0
            
            logger.info(f"Video: {video_duration:.1f}s, Processing: {processing_duration:.1f}s")
            return processing_duration
            
        except Exception as e:
            logger.warning(f"Error calculating duration for {video_file}: {e}")
            return 60.0  # 기본값
    
    def _save_analysis_results(self, input_file: str, output_dir: str, pipeline) -> bool:
        """분석 결과 저장"""
        # 결과 수집
        final_stats = pipeline.get_performance_stats()
        classification_results = pipeline.get_classification_results()
        frame_poses_results = pipeline.get_frame_poses_results()
        rtmo_poses_results = pipeline.get_rtmo_poses_results()
        
        # 출력 경로 설정
        input_path = Path(input_file)
        video_name = input_path.stem
        
        # 비디오별 폴더 생성 (사용자 요구사항에 맞게)
        base_output_dir = Path(output_dir)
        video_output_dir = base_output_dir / video_name
        
        # 디렉토리 생성
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 경로 (각 비디오 폴더 내에 json, pkl 파일)
        result_file = video_output_dir / f"{video_name}_results.json"
        poses_file = video_output_dir / f"{video_name}_frame_poses.pkl"
        rtmo_poses_file = video_output_dir / f"{video_name}_rtmo_poses.pkl"
        
        try:
            # JSON 결과 저장
            with open(result_file, 'w') as f:
                json.dump({
                    'input_video': input_file,
                    'video_name': video_name,
                    'performance_stats': final_stats,
                    'classification_results': classification_results,
                    'summary': {
                        'total_classifications': len(classification_results),
                        'fight_predictions': len([r for r in classification_results if r.get('prediction') == 1]),
                        'non_fight_predictions': len([r for r in classification_results if r.get('prediction') == 0]),
                        'total_pose_frames': len(frame_poses_results)
                    },
                    'timestamp': time.time()
                }, f, indent=2)
            
            # PKL 파일 저장
            with open(poses_file, 'wb') as f:
                pickle.dump(frame_poses_results, f)
            
            with open(rtmo_poses_file, 'wb') as f:
                pickle.dump(rtmo_poses_results, f)
            
            logger.info(f"Analysis complete - Results saved:")
            logger.info(f"  JSON: {result_file}")
            logger.info(f"  Frame poses: {poses_file}")
            logger.info(f"  RTMO poses: {rtmo_poses_file}")
            logger.info(f"  Classifications: {len(classification_results)}")
            logger.info(f"  Pose frames: {len(frame_poses_results)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def _save_analysis_results_to_specific_dir(self, input_file: str, specific_output_dir: str, pipeline) -> bool:
        """지정된 특정 디렉토리에 분석 결과 저장"""
        # 결과 수집
        final_stats = pipeline.get_performance_stats()
        classification_results = pipeline.get_classification_results()
        frame_poses_results = pipeline.get_frame_poses_results()
        rtmo_poses_results = pipeline.get_rtmo_poses_results()
        
        # 출력 경로 설정
        input_path = Path(input_file)
        video_name = input_path.stem
        
        # 지정된 디렉토리 사용
        video_output_dir = Path(specific_output_dir)
        
        # 파일 경로 (각 비디오 폴더 내에 json, pkl 파일)
        result_file = video_output_dir / f"{video_name}_results.json"
        poses_file = video_output_dir / f"{video_name}_frame_poses.pkl"
        rtmo_poses_file = video_output_dir / f"{video_name}_rtmo_poses.pkl"
        
        try:
            # JSON 결과 저장
            with open(result_file, 'w') as f:
                json.dump({
                    'input_video': input_file,
                    'video_name': video_name,
                    'performance_stats': final_stats,
                    'classification_results': classification_results,
                    'summary': {
                        'total_classifications': len(classification_results),
                        'fight_predictions': len([r for r in classification_results if r.get('prediction') == 1]),
                        'non_fight_predictions': len([r for r in classification_results if r.get('prediction') == 0]),
                        'total_pose_frames': len(frame_poses_results)
                    },
                    'timestamp': time.time()
                }, f, indent=2)
            
            # PKL 파일 저장
            with open(poses_file, 'wb') as f:
                pickle.dump(frame_poses_results, f)
            
            with open(rtmo_poses_file, 'wb') as f:
                pickle.dump(rtmo_poses_results, f)
            
            logger.info(f"Analysis complete - Results saved:")
            logger.info(f"  JSON: {result_file}")
            logger.info(f"  Frame poses: {poses_file}")
            logger.info(f"  RTMO poses: {rtmo_poses_file}")
            logger.info(f"  Classifications: {len(classification_results)}")
            logger.info(f"  Pose frames: {len(frame_poses_results)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def _save_analysis_results_from_dict(self, input_file: str, output_dir: str, result_dict: Dict[str, Any]) -> bool:
        """새로운 분석 결과 딕셔너리로부터 결과 저장"""
        try:
            from pathlib import Path
            import json
            import pickle
            
            video_path = Path(input_file)
            video_name = video_path.stem
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 윈도우 프레임 범위 정보 추가
            classifications_with_frames = []
            for i, classification in enumerate(result_dict.get('classifications', [])):
                # 윈도우 프레임 범위 계산
                window_start = i * 50  # stride=50 기준
                window_end = window_start + 100 - 1  # window_size=100 기준
                
                # 분류 결과에 프레임 범위 추가
                enhanced_classification = classification.copy()
                enhanced_classification.update({
                    'window_id': i,
                    'window_start_frame': window_start,
                    'window_end_frame': window_end,
                    'frame_range': f"{window_start}-{window_end}"
                })
                classifications_with_frames.append(enhanced_classification)
            
            # JSON 결과 생성
            json_result = {
                'input_video': input_file,
                'video_name': video_name,
                'performance_stats': result_dict.get('performance_stats', {}),
                'classification_results': classifications_with_frames,
                'window_analysis': {
                    'total_windows': len(classifications_with_frames),
                    'window_size': 100,
                    'window_stride': 50,
                    'windows_info': [
                        {
                            'window_id': i,
                            'frame_range': f"{i * 50}-{i * 50 + 99}",
                            'prediction': cls.get('predicted_label', 'Unknown'),
                            'confidence': cls.get('confidence', 0.0)
                        } for i, cls in enumerate(classifications_with_frames)
                    ]
                },
                'summary': {
                    'total_classifications': len(classifications_with_frames),
                    'fight_predictions': sum(1 for c in classifications_with_frames if c.get('prediction') == 1),
                    'non_fight_predictions': sum(1 for c in classifications_with_frames if c.get('prediction') == 0),
                    'total_pose_frames': result_dict.get('total_frames', 0)
                },
                'timestamp': time.time()
            }
            
            # JSON 파일 저장
            json_file = output_path / f"{video_name}_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            # PKL 파일들 저장
            if 'raw_pose_results' in result_dict:
                # 1. 포즈추정 PKL 파일 (원본 포즈 데이터)
                rtmo_poses_file = output_path / f"{video_name}_rtmo_poses.pkl"
                with open(rtmo_poses_file, 'wb') as f:
                    pickle.dump(result_dict['raw_pose_results'], f)
                logger.info(f"Raw pose results saved to: {rtmo_poses_file}")
            
            if 'processed_frame_poses' in result_dict:
                # 2. 트래킹+복합점수 계산 후 정렬된 PKL 파일
                frame_poses_file = output_path / f"{video_name}_frame_poses.pkl"
                with open(frame_poses_file, 'wb') as f:
                    pickle.dump(result_dict['processed_frame_poses'], f)
                logger.info(f"Processed frame poses saved to: {frame_poses_file}")
            
            logger.info(f"Analysis results saved to: {json_file}")
            logger.info(f"Total windows classified: {len(classifications_with_frames)}")
            logger.info(f"Total frames processed: {result_dict.get('total_frames', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            return False


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