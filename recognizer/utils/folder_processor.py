"""
폴더 단위 비디오 처리 유틸리티

입력 폴더의 구조를 유지하면서 모든 비디오를 재귀적으로 처리합니다.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import time

logger = logging.getLogger(__name__)


class FolderVideoProcessor:
    """폴더 단위 비디오 처리기"""
    
    def __init__(self, config):
        """
        Args:
            config: UnifiedConfig 객체
        """
        self.config = config
        self.input_dir = Path(config.input_dir) if config.input_dir else None
        self.output_dir = Path(config.output_dir)
        self.video_extensions = [ext.lower() for ext in config.video_extensions]
        self.processing_duration = config.processing_duration
        
        # 통계
        self.total_videos = 0
        self.processed_videos = 0
        self.failed_videos = 0
        self.skipped_videos = 0
        
    def find_all_videos(self) -> List[Tuple[Path, Path]]:
        """
        입력 디렉토리에서 모든 비디오 파일을 재귀적으로 찾습니다.
        
        Returns:
            List[Tuple[Path, Path]]: (절대경로, 상대경로) 튜플 리스트
        """
        if not self.input_dir or not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        video_files = []
        logger.info(f"Scanning for videos in: {self.input_dir}")
        logger.info(f"Supported extensions: {self.video_extensions}")
        
        # 재귀적으로 모든 파일 검사
        for root, dirs, files in os.walk(self.input_dir):
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                if file_path.suffix.lower() in self.video_extensions:
                    # 입력 디렉토리 기준 상대 경로
                    relative_path = file_path.relative_to(self.input_dir)
                    video_files.append((file_path, relative_path))
                    logger.debug(f"Found video: {relative_path}")
        
        logger.info(f"Total videos found: {len(video_files)}")
        return video_files
    
    def create_output_path(self, relative_video_path: Path) -> Tuple[Path, Path, Path, Path]:
        """
        출력 경로를 생성합니다.
        
        예시:
        - input_dir: /workspace/raw_videos
        - relative_video_path: Fight/cam04_06.mp4
        - output: output/raw_videos/Fight/overlay|pkl|json/
        
        Args:
            relative_video_path: 비디오의 상대 경로
            
        Returns:
            Tuple[Path, Path, Path, Path]: (base_dir, overlay_dir, pkl_dir, json_dir)
        """
        # 입력 폴더의 마지막 이름 (예: raw_videos)
        input_folder_name = self.input_dir.name
        
        # 비디오 파일의 상위 디렉토리 (예: Fight)
        video_parent_dir = relative_video_path.parent
        
        # output/raw_videos/Fight/ 구조 생성
        base_output_dir = self.output_dir / input_folder_name / video_parent_dir
        
        # 하위 디렉토리들
        overlay_dir = base_output_dir / "overlay"
        pkl_dir = base_output_dir / "pkl"
        json_dir = base_output_dir / "json"
        
        # 모든 디렉토리 생성
        for dir_path in [overlay_dir, pkl_dir, json_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created output structure: {base_output_dir}")
        return base_output_dir, overlay_dir, pkl_dir, json_dir
    
    def is_already_processed(self, video_path: Path, json_dir: Path) -> bool:
        """비디오가 이미 처리되었는지 확인"""
        video_name = video_path.stem
        result_file = json_dir / f"{video_name}_results.json"
        return result_file.exists()
    
    def process_single_video(self, video_path: Path, relative_path: Path, pipeline) -> bool:
        """단일 비디오 처리"""
        try:
            logger.info(f"Processing: {relative_path}")
            start_time = time.time()
            
            # 출력 경로 생성
            base_output_dir, overlay_dir, pkl_dir, json_dir = self.create_output_path(relative_path)
            
            # 이미 처리된 경우 스킵
            if self.is_already_processed(video_path, json_dir):
                logger.info(f"Already processed, skipping: {video_path.name}")
                self.skipped_videos += 1
                return True
            
            # 비디오 정보 분석 및 동적 처리 시간 계산
            dynamic_duration = self._calculate_processing_duration(video_path)
            logger.info(f"Calculated processing duration: {dynamic_duration:.1f}s for {video_path.name}")
            
            # 트래커 초기화 (새 비디오 처리 전에 필수)
            if hasattr(pipeline, 'tracker') and hasattr(pipeline.tracker, 'reset_tracker'):
                pipeline.tracker.reset_tracker()
                logger.debug(f"Reset tracker for new video: {video_path.name}")
            
            # 파이프라인 처리
            logger.info(f"Starting processing for: {video_path}")
            pipeline.start_realtime_processing(str(video_path))
            
            # 동적 계산된 시간동안 처리 + 완료 조건 확인
            actual_duration = self._wait_for_processing_completion(pipeline, dynamic_duration)
            
            # 처리 중지
            pipeline.stop_realtime_processing()
            
            # 결과 수집
            final_stats = pipeline.get_performance_stats()
            classification_results = pipeline.get_classification_results()
            frame_poses_results = pipeline.get_frame_poses_results()
            rtmo_poses_results = pipeline.get_rtmo_poses_results()
            
            # 결과 저장
            self._save_results(
                video_path, relative_path, base_output_dir,
                overlay_dir, pkl_dir, json_dir,
                final_stats, classification_results,
                frame_poses_results, rtmo_poses_results
            )
            
            # 시각화 생성 (활성화된 경우)
            if getattr(self.config, 'enable_visualization', False):
                self._create_visualization(
                    video_path, overlay_dir,
                    classification_results, frame_poses_results, rtmo_poses_results
                )
            
            elapsed = time.time() - start_time
            logger.info(f"Completed {video_path.name} in {elapsed:.1f}s (actual processing: {actual_duration:.1f}s)")
            self.processed_videos += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.failed_videos += 1
            return False
    
    def _save_results(self, video_path: Path, relative_path: Path, base_output_dir: Path,
                     overlay_dir: Path, pkl_dir: Path, json_dir: Path,
                     final_stats: dict, classification_results: list,
                     frame_poses_results: list, rtmo_poses_results: list):
        """결과 파일 저장 - 공용 ResultSaver 사용"""
        from .result_saver import ResultSaver
        
        video_name = video_path.stem
        
        # 결과 데이터 구성
        result_dict = {
            'classifications': classification_results,
            'total_frames': len(frame_poses_results),
            'raw_pose_results': rtmo_poses_results,
            'processed_frame_poses': frame_poses_results,
            'performance_stats': final_stats,
            'folder_metadata': {
                'input_video': str(video_path),
                'relative_path': str(relative_path),
                'input_folder_name': self.input_dir.name,
                'processing_duration': self.processing_duration
            }
        }
        
        # 공용 ResultSaver로 저장
        ResultSaver.save_analysis_results(str(video_path), str(json_dir), result_dict)
    
    def _create_visualization(self, video_path: Path, overlay_dir: Path,
                            classification_results: list, frame_poses_results: list,
                            rtmo_poses_results: list):
        """시각화 생성"""
        try:
            from visualization.inference_visualizer import create_inference_visualization
            
            video_name = video_path.stem
            vis_output_path = overlay_dir / f"{video_name}_overlay.mp4"
            
            success = create_inference_visualization(
                input_video=str(video_path),
                classification_results=classification_results,
                frame_poses_results=frame_poses_results,
                rtmo_poses_results=rtmo_poses_results,
                output_path=str(vis_output_path)
            )
            
            if success:
                logger.info(f"Visualization saved: {vis_output_path}")
            else:
                logger.warning(f"Visualization failed for: {video_path.name}")
                
        except Exception as e:
            logger.error(f"Visualization error for {video_path.name}: {e}")
    
    def _calculate_processing_duration(self, video_path: Path) -> float:
        """비디오 길이 기반 동적 처리 시간 계산"""
        from .video_utils import get_video_duration
        
        try:
            video_duration = get_video_duration(str(video_path))
            if video_duration <= 0:
                return self.processing_duration
            
            # 처리 시간 계산: 비디오 길이 * 1.2 + 3초 버퍼
            calculated_duration = video_duration * 1.2 + 3.0
            final_duration = max(10.0, min(calculated_duration, 30.0))
            
            logger.debug(f"Video duration: {video_duration:.1f}s, Processing: {final_duration:.1f}s")
            return final_duration
            
        except Exception as e:
            logger.warning(f"Error calculating processing duration for {video_path}: {e}")
            return self.processing_duration
    
    def _wait_for_processing_completion(self, pipeline, max_duration: float) -> float:
        """처리 완료까지 대기 (윈도우 완료 조건 포함)"""
        start_time = time.time()
        last_window_count = 0
        stable_count = 0
        
        logger.debug(f"Waiting for processing completion (max {max_duration:.1f}s)")
        
        while True:
            elapsed = time.time() - start_time
            
            # 최대 시간 초과 확인
            if elapsed >= max_duration:
                logger.info(f"Processing completed by timeout after {elapsed:.1f}s")
                break
            
            # 현재 처리 상태 확인
            try:
                stats = pipeline.get_performance_stats()
                classification_results = pipeline.get_classification_results()
                current_window_count = len(classification_results)
                
                # 윈도우 처리 진행 상태 모니터링
                if current_window_count > last_window_count:
                    logger.debug(f"Processing progress: {current_window_count} windows completed ({elapsed:.1f}s)")
                    last_window_count = current_window_count
                    stable_count = 0  # 진행중이므로 리셋
                else:
                    stable_count += 1
                
                # 일정 시간동안 새로운 윈도우가 없으면 완료로 간주
                # 단, 최소 처리 시간은 비디오 길이의 50% 이상 보장
                min_processing_time = max(10.0, max_duration * 0.5)
                if elapsed > min_processing_time and stable_count >= 25:  # 5초간 (0.2초 * 25) 안정
                    logger.info(f"Processing completed by stability condition after {elapsed:.1f}s ({current_window_count} windows)")
                    break
                    
                # 처리 진행률 체크 (매 5초마다)
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    frames_processed = stats.get('total_processed', 0)
                    if frames_processed > 0:
                        logger.debug(f"Progress check: {frames_processed} frames, {current_window_count} windows in {elapsed:.1f}s")
                    
            except Exception as e:
                logger.warning(f"Error checking processing status: {e}")
            
            time.sleep(0.2)  # 200ms 간격으로 체크
        
        final_duration = time.time() - start_time
        return final_duration
    
    def process_all_videos(self, pipeline) -> dict:
        """모든 비디오 처리 실행"""
        if not self.input_dir:
            logger.error("No input_dir specified")
            return {}
        
        logger.info("=== FOLDER PROCESSING START ===")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # 모든 비디오 파일 찾기
        video_files = self.find_all_videos()
        self.total_videos = len(video_files)
        
        if self.total_videos == 0:
            logger.warning("No video files found")
            return {'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0}
        
        # 처리 시작
        start_time = time.time()
        
        for i, (video_path, relative_path) in enumerate(video_files, 1):
            logger.info(f"=== Processing {i}/{self.total_videos}: {relative_path} ===")
            
            success = self.process_single_video(video_path, relative_path, pipeline)
            
            # 진행률 출력
            progress = (i / self.total_videos) * 100
            logger.info(f"Progress: {progress:.1f}% ({i}/{self.total_videos})")
        
        # 최종 통계
        elapsed_time = time.time() - start_time
        
        logger.info("=== FOLDER PROCESSING COMPLETE ===")
        logger.info(f"Total time: {elapsed_time:.1f}s")
        logger.info(f"Total videos: {self.total_videos}")
        logger.info(f"Processed: {self.processed_videos}")
        logger.info(f"Failed: {self.failed_videos}")
        logger.info(f"Skipped: {self.skipped_videos}")
        
        return {
            'total': self.total_videos,
            'processed': self.processed_videos,
            'failed': self.failed_videos,
            'skipped': self.skipped_videos,
            'elapsed_time': elapsed_time,
            'success_rate': (self.processed_videos / self.total_videos * 100) if self.total_videos > 0 else 0
        }


def should_use_folder_processing(config) -> bool:
    """폴더 처리 모드를 사용해야 하는지 확인"""
    return hasattr(config, 'input_dir') and config.input_dir and Path(config.input_dir).exists()