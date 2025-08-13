#!/usr/bin/env python3
"""
Enhanced Parallel Inference Pipeline (Config-based)
Pose Estimation -> Tracking -> MMAction2 Inference -> Consecutive Event Processing -> Performance Evaluation

병렬 처리 기능:
  1. GPU-레벨 병렬화: 여러 GPU에 비디오를 분산하여 처리
  2. 비디오-레벨 병렬화: CPU 멀티스레딩으로 비디오 처리 속도 향상  
  3. 윈도우-레벨 병렬화: 각 비디오의 윈도우들을 병렬로 처리

기본 사용법:
  # 기본 실행 (resume 모드, 모든 병렬 처리 활성화)
  python inference_pipeline.py --config configs/inference_config.py
  
  # 모든 비디오 강제 재처리
  python inference_pipeline.py --config configs/inference_config.py --force
  
  # Resume 모드 비활성화 (모든 비디오 처리)
  python inference_pipeline.py --config configs/inference_config.py --no-resume

병렬 처리 제어:
  # 비디오 레벨 병렬 처리 비활성화
  python inference_pipeline.py --config configs/inference_config.py --disable-video-parallel
  
  # 윈도우 레벨 병렬 처리 비활성화  
  python inference_pipeline.py --config configs/inference_config.py --disable-window-parallel
  
  # 워커 수 직접 지정
  python inference_pipeline.py --config configs/inference_config.py --video-workers 2 --window-workers 4
  
  # 모든 병렬 처리 비활성화 (순차 처리)
  python inference_pipeline.py --config configs/inference_config.py --disable-video-parallel --disable-window-parallel

멀티 GPU 사용법:
  # 멀티 GPU + 병렬 처리
  python inference_pipeline.py --config configs/inference_config.py gpu=0,1,2,3
  
  # GPU별 워커 수 조정
  python inference_pipeline.py --config configs/inference_config.py gpu=0,1 --video-workers 1 --window-workers 6

Config 오버라이드:
  # 설정 오버라이드 
  python inference_pipeline.py --config configs/inference_config.py gpu=1 debug_mode=True enable_video_parallel=False
  
  # 병렬 처리와 설정 조합
  python inference_pipeline.py --force gpu=0,1 classification_threshold=0.7 --video-workers 2

성능 최적화 팁:
  - CPU 코어 수가 많을 때: --video-workers와 --window-workers 증가
  - 메모리 제약이 있을 때: 워커 수를 줄이거나 순차 처리 사용
  - GPU 메모리 부족시: 비디오 레벨 병렬 처리만 사용 (--disable-window-parallel)
  - I/O 병목이 있을 때: 윈도우 레벨 병렬 처리만 사용 (--disable-video-parallel)
"""

import os
import sys
import json
import pickle
import glob
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
import signal
import time
import atexit
from functools import partial
import fcntl
from datetime import datetime

try:
    import torch
    from mmaction.apis import init_recognizer, inference_recognizer
    from mmaction.utils import register_all_modules
    register_all_modules()
except ImportError as e:
    print(f"MMAction2 is not available: {e}")
    sys.exit("Please install MMAction2 or check the installation.")

# 상위 디렉토리를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from configs import load_config
from core.pose_extractor import EnhancedRTMOPoseExtractor
from error_logging.error_logger import ProcessingErrorLogger, capture_exception_info

# 병렬 처리를 위한 전역 변수들
_video_processor_pool = None
_window_processor_pool = None

class RealtimeResultsManager:
    """실시간 결과 업데이트 및 관리 클래스"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.window_results_file = os.path.join(results_dir, 'window_results.json')
        self.video_results_file = os.path.join(results_dir, 'video_results.json')
        self.progress_file = os.path.join(results_dir, 'progress.json')
        self.performance_file = os.path.join(results_dir, 'performance_metrics.json')
        
        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()
        
        # 초기화
        self._initialize_files()
        
    def _initialize_files(self):
        """결과 파일들을 초기화"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 기존 파일이 없으면 빈 구조로 초기화
        if not os.path.exists(self.window_results_file):
            self._write_json_safely(self.window_results_file, [])
            
        if not os.path.exists(self.video_results_file):
            self._write_json_safely(self.video_results_file, [])
            
        if not os.path.exists(self.progress_file):
            self._write_json_safely(self.progress_file, {
                'total_videos': 0,
                'processed_videos': 0,
                'current_video': '',
                'start_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'status': 'initializing'
            })
    
    def _write_json_safely(self, file_path: str, data: Any):
        """스레드 안전한 JSON 파일 쓰기"""
        temp_file = file_path + '.tmp'
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                # 파일 락을 사용하여 동시 접근 방지
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(self._convert_numpy_to_serializable(data), f, indent=2, ensure_ascii=False)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # 원자적 파일 교체
            os.replace(temp_file, file_path)
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _read_json_safely(self, file_path: str) -> Any:
        """스레드 안전한 JSON 파일 읽기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data
        except Exception as e:
            print(f"Error reading from {file_path}: {e}")
            return None
    
    def update_progress(self, total_videos: int, processed_videos: int, current_video: str = '', status: str = 'processing'):
        """진행률 업데이트"""
        with self._lock:
            progress_data = {
                'total_videos': total_videos,
                'processed_videos': processed_videos,
                'current_video': current_video,
                'progress_percentage': (processed_videos / total_videos * 100) if total_videos > 0 else 0,
                'last_update': datetime.now().isoformat(),
                'status': status
            }
            
            # 기존 데이터에서 start_time 보존
            existing_data = self._read_json_safely(self.progress_file)
            if existing_data and 'start_time' in existing_data:
                progress_data['start_time'] = existing_data['start_time']
            else:
                progress_data['start_time'] = datetime.now().isoformat()
            
            self._write_json_safely(self.progress_file, progress_data)
    
    def add_video_result(self, video_result: Dict[str, Any]):
        """비디오 결과 추가"""
        with self._lock:
            # 기존 결과 읽기
            existing_results = self._read_json_safely(self.video_results_file) or []
            
            # 중복 제거 (동일한 비디오명의 결과가 있으면 교체)
            video_name = video_result['video_name']
            existing_results = [r for r in existing_results if r.get('video_name') != video_name]
            
            # 새 결과 추가
            existing_results.append(video_result)
            
            # 저장
            self._write_json_safely(self.video_results_file, existing_results)
            
            print(f"Updated results: {video_name} (Total: {len(existing_results)} videos)")
    
    def add_window_results(self, window_results: List[Dict[str, Any]]):
        """윈도우 결과들 추가"""
        if not window_results:
            return
            
        with self._lock:
            # 기존 결과 읽기
            existing_results = self._read_json_safely(self.window_results_file) or []
            
            # 중복 제거 (동일한 비디오의 동일한 윈도우 인덱스 결과가 있으면 제거)
            video_name = window_results[0]['video_name']
            new_window_indices = {w.get('window_idx') for w in window_results}
            existing_results = [r for r in existing_results 
                               if not (r.get('video_name') == video_name and 
                                      r.get('window_idx') in new_window_indices)]
            
            # 새 결과들 추가
            existing_results.extend(window_results)
            
            # 저장
            self._write_json_safely(self.window_results_file, existing_results)
    
    def update_performance_metrics(self):
        """성능 지표 업데이트"""
        with self._lock:
            # 현재 비디오 결과들을 기반으로 성능 계산
            video_results = self._read_json_safely(self.video_results_file) or []
            
            if not video_results:
                return
            
            performance = self._calculate_performance_metrics(video_results)
            performance['last_update'] = datetime.now().isoformat()
            performance['total_videos'] = len(video_results)
            
            self._write_json_safely(self.performance_file, performance)
    
    def _calculate_performance_metrics(self, video_results: List[Dict]) -> Dict[str, Any]:
        """성능 지표 계산"""
        if not video_results:
            return {'confusion_matrix': {}, 'metrics': {}, 'total_videos': 0}
            
        true_labels = [r['true_label'] for r in video_results]
        pred_labels = [r['video_prediction'] for r in video_results]
        
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
        
        return {
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'metrics': {'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': acc},
            'total_videos': len(video_results)
        }
    
    def get_current_progress(self) -> Dict[str, Any]:
        """현재 진행률 반환"""
        return self._read_json_safely(self.progress_file) or {}
    
    def get_all_results(self) -> Tuple[List[Dict], List[Dict]]:
        """모든 결과 반환 (window_results, video_results)"""
        window_results = self._read_json_safely(self.window_results_file) or []
        video_results = self._read_json_safely(self.video_results_file) or []
        return window_results, video_results
    
    def finalize_results(self):
        """처리 완료 시 최종 결과 정리"""
        with self._lock:
            # 최종 성능 지표 업데이트
            self.update_performance_metrics()
            
            # 진행률을 완료로 설정
            progress_data = self.get_current_progress()
            progress_data.update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat()
            })
            self._write_json_safely(self.progress_file, progress_data)
            
            print(f"Results finalized. Total videos processed: {progress_data.get('processed_videos', 0)}")
    
    def _convert_numpy_to_serializable(self, obj: Any) -> Any:
        """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_numpy_to_serializable(i) for i in obj]
        return obj


class FightInferenceProcessor:
    """Fight Detection Inference Processor"""

    def __init__(self, config: Any):
        self.config = config
        self.device, self.gpu_ids = self._parse_gpu_config(config.gpu)
        self.pose_extractor, self.action_model = self._initialize_models()
        self.error_logger = ProcessingErrorLogger(config.output_dir, "")
        
        # 병렬 처리 설정
        self.enable_video_parallel = getattr(config, 'enable_video_parallel', True)
        self.enable_window_parallel = getattr(config, 'enable_window_parallel', True)
        self.video_workers = getattr(config, 'video_workers', min(mp.cpu_count() // 2, 4))
        self.window_workers = getattr(config, 'window_workers', min(mp.cpu_count(), 8))
        self.max_queue_size = getattr(config, 'max_queue_size', 100)
        
        # 실시간 결과 관리자 초기화
        self.results_manager = None

    @staticmethod
    def _parse_gpu_config(gpu_config: Any) -> (str, List[int]):
        """Parses GPU configuration to return device and gpu_ids."""
        if not gpu_config or str(gpu_config).lower() == 'cpu':
            return 'cpu', []
        
        if isinstance(gpu_config, str):
            ids_str = gpu_config.split(',')
            gpu_ids = [int(x.strip()) for x in ids_str]
        elif isinstance(gpu_config, (list, tuple)):
            gpu_ids = list(gpu_config)
        else:
            gpu_ids = [int(gpu_config)]
        
        device = f'cuda:{gpu_ids[0]}'
        return device, gpu_ids

    def _initialize_models(self) -> (EnhancedRTMOPoseExtractor, Any):
        """Initializes and returns the pose and action models."""
        pose_extractor = EnhancedRTMOPoseExtractor(
            config_file=self.config.detector_config,
            checkpoint_file=self.config.detector_checkpoint,
            device=self.device,
            gpu_ids=self.gpu_ids,
            score_thr=self.config.score_thr,
            nms_thr=self.config.nms_thr,
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            track_max_disappeared=self.config.track_max_disappeared,
            track_min_hits=self.config.track_min_hits,
            quality_threshold=self.config.quality_threshold,
            min_track_length=self.config.min_track_length,
            weights=self.config.get_weights()
        )
        action_model = init_recognizer(self.config.action_config, self.config.action_checkpoint, device=self.device)
        return pose_extractor, action_model

    def run_inference(self, resume_mode=True, force_reprocess=False):
        """Executes the full inference pipeline with resume capability."""
        print("=" * 70 + " Fight Detection Inference Pipeline" + "=" * 70)
        output_dir, windows_dir, results_dir = self._setup_directories()
        
        # 실시간 결과 관리자 초기화
        self.results_manager = RealtimeResultsManager(results_dir)
        
        # Check if final results already exist
        if not force_reprocess and self._check_existing_results(results_dir):
            print("Final results already exist:")
            print(f"  - window_results.json")  
            print(f"  - video_results.json")
            print(f"  - performance_metrics.json")
            print("Use --force to reprocess all videos")
            return
        
        video_files = self._collect_video_files(self.config.input_dir)
        
        # 총 비디오 수로 진행률 초기화
        self.results_manager.update_progress(
            total_videos=len(video_files),
            processed_videos=0,
            status='starting'
        )
        
        if force_reprocess:
            remaining_videos = video_files
            print("Force mode: Reprocessing all videos")
        elif resume_mode:
            processed_videos = self._get_processed_videos(windows_dir)
            remaining_videos = [v for v in video_files if self._get_video_key(v) not in processed_videos]
            print(f"Total videos: {len(video_files)}")
            print(f"Already processed: {len(processed_videos)}")
            print(f"Remaining videos: {len(remaining_videos)}")
        else:
            remaining_videos = video_files
            print(f"Processing all {len(video_files)} videos")
        
        try:
            if not remaining_videos:
                print("All videos already processed")
                # Load existing results and update results manager
                all_window_results, all_video_results = self._load_existing_results(windows_dir)
                self._sync_existing_results_to_manager(all_window_results, all_video_results)
            else:
                # Process remaining videos
                if len(self.gpu_ids) > 1:
                    print(f"Using multi-GPU processing with GPUs: {self.gpu_ids}")
                    all_window_results, all_video_results = self._run_multi_gpu_inference(remaining_videos, windows_dir)
                else:
                    print(f"Using single GPU processing with GPU: {self.device}")
                    if self.enable_video_parallel:
                        print(f"Video-level parallel processing enabled with {self.video_workers} workers")
                    if self.enable_window_parallel:
                        print(f"Window-level parallel processing enabled with {self.window_workers} workers")
                    all_window_results, all_video_results = self._run_single_gpu_inference(remaining_videos, windows_dir)
                
                # If resume mode, combine with existing results (but avoid duplicates)
                if resume_mode and not force_reprocess:
                    existing_window_results, existing_video_results = self._load_existing_results(windows_dir)
                    
                    # Get already processed video names to avoid duplicates
                    processed_video_names = {result['video_name'] for result in all_video_results}
                    
                    # Add only non-duplicate results
                    for existing_result in existing_video_results:
                        if existing_result['video_name'] not in processed_video_names:
                            all_video_results.append(existing_result)
                            
                    # Add window results (they shouldn't have duplicates since they're file-based)
                    all_window_results.extend(existing_window_results)

            # 최종 결과 저장 (호환성을 위해 기존 방식도 유지)
            self._save_results_legacy(all_window_results, all_video_results, results_dir)
            
            # 실시간 결과 관리자 완료 처리
            if self.results_manager:
                self.results_manager.finalize_results()
            
            print("Inference pipeline completed.")
            
        except Exception as e:
            # 오류 발생 시에도 현재까지의 결과는 보존
            if self.results_manager:
                progress = self.results_manager.get_current_progress()
                progress.update({
                    'status': 'error',
                    'error_message': str(e),
                    'last_update': datetime.now().isoformat()
                })
                self.results_manager._write_json_safely(self.results_manager.progress_file, progress)
            raise

    def _run_single_gpu_inference(self, video_files: List[str], windows_dir: str):
        """Single GPU inference processing with parallel video processing option."""
        all_window_results, all_video_results = [], []
        
        # 디버그 모드에서는 첫 번째 비디오만 처리
        if hasattr(self.config, 'debug_single_video') and self.config.debug_single_video:
            video_files = video_files[:1]
            print(f"[DEBUG] Processing only first video: {video_files[0]}")
        
        if self.enable_video_parallel and len(video_files) > 1:
            # 비디오 레벨 병렬 처리
            return self._run_parallel_video_inference(video_files, windows_dir)
        else:
            # 기존 순차 처리 방식
            for video_idx, video_path in enumerate(video_files):
                try:
                    print(f"Processing video {video_idx+1}/{len(video_files)}: {Path(video_path).name}")
                    video_result = self._process_video(video_path, windows_dir, is_first_video=(video_idx == 0))
                    if video_result:
                        all_window_results.extend(video_result['window_results'])
                        all_video_results.append(video_result['video_summary'])
                    else:
                        print(f"[WARNING] No result returned for {Path(video_path).name}")
                except Exception as e:
                    self.error_logger.log_general_error("video_processing_error", str(e), capture_exception_info())
                    print(f"Error processing {video_path}: {e}")
                    import traceback
                    traceback.print_exc()
            return all_window_results, all_video_results

    def _run_parallel_video_inference(self, video_files: List[str], windows_dir: str):
        """비디오 레벨 병렬 처리를 위한 새로운 메서드"""
        all_window_results, all_video_results = [], []
        
        try:
            with ThreadPoolExecutor(max_workers=self.video_workers) as executor:
                print(f"Starting parallel video processing with {self.video_workers} workers...")
                
                # 각 비디오를 별도 스레드에서 처리
                future_to_video = {}
                for video_idx, video_path in enumerate(video_files):
                    future = executor.submit(
                        self._process_video_thread_safe, 
                        video_path, 
                        windows_dir, 
                        is_first_video=(video_idx == 0)
                    )
                    future_to_video[future] = video_path
                
                # 완료된 작업들을 수집
                for future in as_completed(future_to_video, timeout=3600):
                    video_path = future_to_video[future]
                    try:
                        video_result = future.result(timeout=300)
                        if video_result:
                            all_window_results.extend(video_result['window_results'])
                            all_video_results.append(video_result['video_summary'])
                            
                            # 실시간 결과 업데이트
                            if self.results_manager:
                                self.results_manager.add_video_result(video_result['video_summary'])
                                self.results_manager.add_window_results(video_result['window_results'])
                                self.results_manager.update_progress(
                                    total_videos=len(video_files),
                                    processed_videos=len(all_video_results),
                                    current_video=video_result['video_summary']['video_name'],
                                    status='processing'
                                )
                                self.results_manager.update_performance_metrics()
                            
                            print(f"Completed processing: {Path(video_path).name}")
                        else:
                            print(f"[WARNING] No result returned for {Path(video_path).name}")
                    except Exception as e:
                        self.error_logger.log_general_error("parallel_video_processing_error", str(e), capture_exception_info())
                        print(f"Error in parallel processing {video_path}: {e}")
        
        except Exception as e:
            print(f"Error in parallel video inference: {e}")
            # 오류 발생시 순차 처리로 폴백
            print("Falling back to sequential processing...")
            return self._run_sequential_video_inference(video_files, windows_dir)
        
        return all_window_results, all_video_results

    def _run_sequential_video_inference(self, video_files: List[str], windows_dir: str):
        """순차 비디오 처리 (폴백용)"""
        all_window_results, all_video_results = [], []
        
        for video_idx, video_path in enumerate(video_files):
            try:
                print(f"Processing video {video_idx+1}/{len(video_files)}: {Path(video_path).name}")
                
                # 현재 비디오 처리 시작 알림
                if self.results_manager:
                    self.results_manager.update_progress(
                        total_videos=len(video_files),
                        processed_videos=len(all_video_results),
                        current_video=Path(video_path).name,
                        status='processing'
                    )
                
                video_result = self._process_video(video_path, windows_dir, is_first_video=(video_idx == 0))
                if video_result:
                    all_window_results.extend(video_result['window_results'])
                    all_video_results.append(video_result['video_summary'])
                    
                    # 실시간 결과 업데이트
                    if self.results_manager:
                        self.results_manager.add_video_result(video_result['video_summary'])
                        self.results_manager.add_window_results(video_result['window_results'])
                        self.results_manager.update_progress(
                            total_videos=len(video_files),
                            processed_videos=len(all_video_results),
                            current_video=video_result['video_summary']['video_name'],
                            status='processing'
                        )
                        self.results_manager.update_performance_metrics()
                else:
                    print(f"[WARNING] No result returned for {Path(video_path).name}")
            except Exception as e:
                self.error_logger.log_general_error("video_processing_error", str(e), capture_exception_info())
                print(f"Error processing {video_path}: {e}")
        
        return all_window_results, all_video_results

    def _process_video_thread_safe(self, video_path: str, windows_dir: str, is_first_video: bool = False) -> Optional[Dict[str, Any]]:
        """스레드 안전한 비디오 처리 메서드"""
        try:
            video_name = Path(video_path).stem
            label_folder = Path(video_path).parent.name
            true_label = 1 if label_folder.lower() == 'fight' else 0

            # 포즈 추출 - 스레드 안전하게 처리
            pose_results = self.pose_extractor.extract_poses_only(video_path)
            
            if not pose_results:
                return None

            pose_results = self._augment_frames(pose_results, self.config.clip_len)
            if not pose_results:
                return None

            if self.enable_window_parallel:
                # 윈도우 레벨 병렬 처리
                window_results = self._process_windows_parallel(
                    pose_results, video_name, true_label, label_folder, is_first_video
                )
            else:
                # 기존 순차 윈도우 처리
                window_results = self._process_windows_sequential(
                    pose_results, video_name, true_label, label_folder, is_first_video
                )

            if not window_results:
                return None
            
            _save_window_results_static(window_results, windows_dir, label_folder, video_name)
            video_prediction = _apply_consecutive_event_rule_static(window_results, self.config.consecutive_event_threshold)
            
            video_summary = {
                'video_name': video_name, 
                'true_label': true_label, 
                'label_folder': label_folder,
                'video_prediction': video_prediction, 
                'window_count': len(window_results),
                'fight_windows': sum(1 for w in window_results if w['predicted_label'] == 1),
                'avg_prediction_score': np.mean([w['prediction'] for w in window_results]) if window_results else 0.0,
                'max_prediction_score': np.max([w['prediction'] for w in window_results]) if window_results else 0.0
            }
            
            return {'window_results': window_results, 'video_summary': video_summary}
            
        except Exception as e:
            print(f"Error in thread-safe video processing {video_path}: {e}")
            return None

    def _process_windows_parallel(self, pose_results: List, video_name: str, true_label: int, 
                                label_folder: str, is_first_video: bool = False) -> List[Dict]:
        """윈도우 레벨 병렬 처리"""
        window_results = []
        
        # 윈도우 인덱스와 파라미터 생성
        window_params = []
        for start_frame in range(0, len(pose_results) - self.config.clip_len + 1, self.config.inference_stride):
            end_frame = start_frame + self.config.clip_len
            window_idx = start_frame // self.config.inference_stride
            debug_enabled = self.config.debug_mode and is_first_video and window_idx == 0
            
            window_params.append({
                'pose_window': pose_results[start_frame:end_frame],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'window_idx': window_idx,
                'video_name': video_name,
                'true_label': true_label,
                'label_folder': label_folder,
                'debug_enabled': debug_enabled
            })
        
        try:
            # 윈도우들을 병렬로 처리
            with ThreadPoolExecutor(max_workers=self.window_workers) as executor:
                future_to_window = {
                    executor.submit(self._process_single_window, params): params['window_idx'] 
                    for params in window_params
                }
                
                # 결과 수집 (순서 보장)
                window_results_dict = {}
                for future in as_completed(future_to_window):
                    window_idx = future_to_window[future]
                    try:
                        result = future.result()
                        if result:
                            window_results_dict[window_idx] = result
                    except Exception as e:
                        print(f"Window {window_idx} processing failed: {e}")
                
                # 윈도우 인덱스 순으로 정렬하여 반환
                window_results = [window_results_dict[idx] for idx in sorted(window_results_dict.keys())]
        
        except Exception as e:
            print(f"Error in parallel window processing: {e}")
            # 오류 발생시 순차 처리로 폴백
            window_results = self._process_windows_sequential(
                pose_results, video_name, true_label, label_folder, is_first_video
            )
        
        return window_results

    def _process_windows_sequential(self, pose_results: List, video_name: str, true_label: int,
                                  label_folder: str, is_first_video: bool = False) -> List[Dict]:
        """윈도우 순차 처리 (기존 방식)"""
        window_results = []
        
        for start_frame in range(0, len(pose_results) - self.config.clip_len + 1, self.config.inference_stride):
            end_frame = start_frame + self.config.clip_len
            window_idx = start_frame // self.config.inference_stride
            try:
                debug_enabled = self.config.debug_mode and is_first_video and window_idx == 0
                
                tracked_window = self.pose_extractor.apply_tracking_to_poses(
                    pose_results[start_frame:end_frame], start_frame, end_frame, window_idx)
                if not tracked_window:
                    continue
                
                prediction = _predict_window_static(
                    tracked_window, self.action_model, self.config.clip_len, 
                    self.config.focus_person, debug=debug_enabled
                )
                
                window_results.append({
                    'video_name': video_name,
                    'true_label': true_label,
                    'label_folder': label_folder,
                    'window_idx': window_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'prediction': prediction,
                    'predicted_label': 1 if prediction > self.config.classification_threshold else 0,
                    'persons_count': len(tracked_window.get('annotation', {}).get('persons', {})),
                    'pose_data': tracked_window
                })
                
                if debug_enabled:
                    print(f"=== END DEBUG INFO ===\n")
                    
            except Exception as e:
                print(f"Window {window_idx} failed: {e}")
        
        return window_results

    def _process_single_window(self, params: Dict) -> Optional[Dict]:
        """단일 윈도우 처리 (병렬 처리용)"""
        try:
            tracked_window = self.pose_extractor.apply_tracking_to_poses(
                params['pose_window'], 
                params['start_frame'], 
                params['end_frame'], 
                params['window_idx']
            )
            
            if not tracked_window:
                return None
            
            prediction = _predict_window_static(
                tracked_window, 
                self.action_model, 
                self.config.clip_len,
                self.config.focus_person, 
                debug=params['debug_enabled']
            )
            
            return {
                'video_name': params['video_name'],
                'true_label': params['true_label'],
                'label_folder': params['label_folder'],
                'window_idx': params['window_idx'],
                'start_frame': params['start_frame'],
                'end_frame': params['end_frame'],
                'prediction': prediction,
                'predicted_label': 1 if prediction > self.config.classification_threshold else 0,
                'persons_count': len(tracked_window.get('annotation', {}).get('persons', {})),
                'pose_data': tracked_window
            }
            
        except Exception as e:
            print(f"Single window processing failed for window {params['window_idx']}: {e}")
            return None

    def _sync_existing_results_to_manager(self, all_window_results: List[Dict], all_video_results: List[Dict]):
        """기존 결과들을 실시간 결과 관리자에 동기화"""
        if not self.results_manager:
            return
            
        # 비디오 결과들 추가
        for video_result in all_video_results:
            self.results_manager.add_video_result(video_result)
        
        # 윈도우 결과들을 비디오별로 그룹핑하여 추가
        video_window_groups = {}
        for window_result in all_window_results:
            video_name = window_result['video_name']
            if video_name not in video_window_groups:
                video_window_groups[video_name] = []
            video_window_groups[video_name].append(window_result)
        
        for video_name, window_results in video_window_groups.items():
            self.results_manager.add_window_results(window_results)
        
        # 진행률 업데이트
        self.results_manager.update_progress(
            total_videos=len(all_video_results),
            processed_videos=len(all_video_results),
            status='loaded_existing'
        )
        
        # 성능 지표 업데이트
        self.results_manager.update_performance_metrics()

    def _save_results_legacy(self, all_window_results: List[Dict], all_video_results: List[Dict], results_dir: str):
        """기존 방식의 결과 저장 (호환성 유지)"""
        # 기존 코드와 동일하게 최종 결과 저장
        # 실시간 결과 관리자와는 별도로 기존 형식도 유지
        pass  # 실시간 결과가 이미 저장되므로 여기서는 생략

    def _run_multi_gpu_inference(self, video_files: List[str], windows_dir: str):
        """Multi-GPU inference processing using process pool with proper cleanup."""
        all_window_results, all_video_results = [], []
        executor = None
        future_to_gpu = {}
        try:
            gpu_count = len(self.gpu_ids)
            video_chunks = [video_files[i::gpu_count] for i in range(gpu_count)]
            print(f"Starting multi-GPU processing on {gpu_count} GPUs...")
            executor = ProcessPoolExecutor(max_workers=gpu_count)
            
            for gpu_idx, gpu_id in enumerate(self.gpu_ids):
                if video_chunks[gpu_idx]:
                    future = executor.submit(process_videos_on_gpu, video_chunks[gpu_idx], windows_dir, self.config, gpu_id)
                    future_to_gpu[future] = gpu_id
                    print(f"Submitted {len(video_chunks[gpu_idx])} videos to GPU {gpu_id}")
            
            if not future_to_gpu: return [], []

            for future in as_completed(future_to_gpu, timeout=3600):
                gpu_id = future_to_gpu[future]
                try:
                    gpu_window_results, gpu_video_results = future.result(timeout=300)
                    all_window_results.extend(gpu_window_results)
                    all_video_results.extend(gpu_video_results)
                    print(f"GPU {gpu_id} completed processing {len(gpu_video_results)} videos.")
                except Exception as e:
                    print(f"GPU {gpu_id} processing failed: {e}")
        finally:
            if executor: executor.shutdown(wait=True)
        return all_window_results, all_video_results

    def _setup_directories(self):
        """Creates necessary output directories."""
        output_dir = self.config.output_dir
        windows_dir = os.path.join(output_dir, 'windows')
        results_dir = os.path.join(output_dir, 'results')
        os.makedirs(windows_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        return output_dir, windows_dir, results_dir

    def _collect_video_files(self, input_dir: str) -> List[str]:
        """Collects video files from the input directory."""
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        return sorted([p for ext in extensions for p in glob.glob(os.path.join(input_dir, '**', ext), recursive=True)])
    
    def _get_processed_videos(self, windows_dir: str) -> set:
        """Get already processed videos by checking window results."""
        processed = set()
        
        # 대소문자 구분없이 fight, normal, nonfight 폴더 모두 처리
        for category in ['fight', 'normal', 'Fight', 'NonFight']:
            category_dir = os.path.join(windows_dir, category)
            if os.path.exists(category_dir):
                for file_name in os.listdir(category_dir):
                    if file_name.endswith('_windows.pkl'):
                        # Extract video name from file name
                        video_name = file_name.replace('_windows.pkl', '')
                        processed.add(f"{category}/{video_name}")
        
        return processed
    
    def _get_video_key(self, video_path: str) -> str:
        """Generate key from video path for comparison."""
        video_name = Path(video_path).stem
        label_folder = Path(video_path).parent.name
        return f"{label_folder}/{video_name}"
    
    def _check_existing_results(self, results_dir: str) -> bool:
        """Check if final results already exist."""
        result_files = [
            'window_results.json',
            'video_results.json', 
            'performance_metrics.json'
        ]
        
        return all(os.path.exists(os.path.join(results_dir, f)) for f in result_files)
    
    def _load_existing_results(self, windows_dir: str) -> tuple:
        """Load existing window results from previous runs."""
        all_window_results = []
        all_video_results = []
        
        # 대소문자 구분없이 fight, normal, nonfight 폴더 모두 처리
        for category in ['fight', 'normal', 'Fight', 'NonFight']:
            category_dir = os.path.join(windows_dir, category)
            if os.path.exists(category_dir):
                for pkl_file in glob.glob(os.path.join(category_dir, '*_windows.pkl')):
                    try:
                        with open(pkl_file, 'rb') as f:
                            window_results = pickle.load(f)
                            
                        if window_results:
                            all_window_results.extend(window_results)
                            
                            # Extract video summary from window results
                            video_name = window_results[0]['video_name']
                            true_label = window_results[0]['true_label']
                            label_folder = window_results[0]['label_folder']
                            
                            # Apply consecutive event rule
                            video_prediction = _apply_consecutive_event_rule_static(
                                window_results, getattr(self.config, 'consecutive_event_threshold', 3)
                            )
                            
                            video_summary = {
                                'video_name': video_name,
                                'true_label': true_label,
                                'label_folder': label_folder,
                                'video_prediction': video_prediction,
                                'window_count': len(window_results),
                                'fight_windows': sum(1 for w in window_results if w['predicted_label'] == 1),
                                'avg_prediction_score': np.mean([w['prediction'] for w in window_results]) if window_results else 0.0,
                                'max_prediction_score': np.max([w['prediction'] for w in window_results]) if window_results else 0.0
                            }
                            all_video_results.append(video_summary)
                            
                    except Exception as e:
                        print(f"Error loading {pkl_file}: {e}")
                        continue
        
        return all_window_results, all_video_results

    def _augment_frames(self, pose_results: List, clip_len: int) -> List:
        """Augments frames if the video is shorter than the clip length."""
        original_length = len(pose_results)
        if original_length >= clip_len: 
            return pose_results
        if original_length == 0: 
            return []
        
        print(f"    Applying padding: {original_length} -> {clip_len} frames")
        augmented_results = list(pose_results)
        while len(augmented_results) < clip_len:
            augmented_results.extend(pose_results)
        return augmented_results[:clip_len]

    def _process_video(self, video_path: str, windows_dir: str, is_first_video: bool = False) -> Dict[str, Any]:
        """Processes a single video file."""
        video_name = Path(video_path).stem
        label_folder = Path(video_path).parent.name
        # 대소문자 구분없이 fight 판단, normal/nonfight를 모두 non-fight로 처리
        true_label = 1 if label_folder.lower() == 'fight' else 0

        pose_results = self.pose_extractor.extract_poses_only(video_path)
        
        if not pose_results: 
            return None

        pose_results = self._augment_frames(pose_results, self.config.clip_len)
        if not pose_results: return None

        window_results = []
        for start_frame in range(0, len(pose_results) - self.config.clip_len + 1, self.config.inference_stride):
            end_frame = start_frame + self.config.clip_len
            window_idx = start_frame // self.config.inference_stride
            try:
                # 첫 번째 비디오의 첫 번째 윈도우에서만 디버그 출력
                debug_enabled = self.config.debug_mode and is_first_video and window_idx == 0
                
                tracked_window = self.pose_extractor.apply_tracking_to_poses(
                    pose_results[start_frame:end_frame], start_frame, end_frame, window_idx)
                if not tracked_window: continue
                
                prediction = _predict_window_static(tracked_window, self.action_model, self.config.clip_len, 
                                                  self.config.focus_person, debug=debug_enabled)
                window_results.append({
                    'video_name': video_name, 'true_label': true_label, 'label_folder': label_folder,
                    'window_idx': window_idx, 'start_frame': start_frame, 'end_frame': end_frame,
                    'prediction': prediction, 'predicted_label': 1 if prediction > self.config.classification_threshold else 0,
                    'persons_count': len(tracked_window.get('annotation', {}).get('persons', {})),
                    'pose_data': tracked_window  # 시각화를 위한 포즈 데이터 추가
                })
                
                if debug_enabled:
                    print(f"=== END DEBUG INFO ===\n")
                    
            except Exception as e:
                print(f"    Window {window_idx} failed: {e}")

        if not window_results: return None
        
        _save_window_results_static(window_results, windows_dir, label_folder, video_name)
        video_prediction = _apply_consecutive_event_rule_static(window_results, self.config.consecutive_event_threshold)
        
        video_summary = {
            'video_name': video_name, 'true_label': true_label, 'label_folder': label_folder,
            'video_prediction': video_prediction, 'window_count': len(window_results),
            'fight_windows': sum(1 for w in window_results if w['predicted_label'] == 1),
            'avg_prediction_score': np.mean([w['prediction'] for w in window_results]) if window_results else 0.0,
            'max_prediction_score': np.max([w['prediction'] for w in window_results]) if window_results else 0.0
        }
        return {'window_results': window_results, 'video_summary': video_summary}

    def _save_results(self, all_window_results: List[Dict], all_video_results: List[Dict], results_dir: str):
        """Saves all results and performance metrics."""
        with open(os.path.join(results_dir, 'window_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_to_serializable(all_window_results), f, indent=2)
        with open(os.path.join(results_dir, 'video_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_to_serializable(all_video_results), f, indent=2)
        
        performance = self._calculate_performance_metrics(all_video_results)
        with open(os.path.join(results_dir, 'performance_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2)
        
        self._print_performance_metrics(performance)
        return {'performance': performance}

    def _calculate_performance_metrics(self, video_results: List[Dict]) -> Dict[str, Any]:
        """Calculates performance metrics."""
        if not video_results: return {'confusion_matrix': {}, 'metrics': {}, 'total_videos': 0}
        true_labels = [r['true_label'] for r in video_results]
        pred_labels = [r['video_prediction'] for r in video_results]
        
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
        
        return {
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'metrics': {'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': acc},
            'total_videos': len(video_results)
        }

    def _print_performance_metrics(self, performance: Dict[str, Any]):
        """Prints performance metrics to the console."""
        print("" + "=" * 70 + " Performance Metrics" + "=" * 70)
        cm = performance['confusion_matrix']
        metrics = performance['metrics']
        print(f"  True Positive: {cm.get('tp', 0)}")
        print(f"  False Positive: {cm.get('fp', 0)}")
        print(f"  True Negative: {cm.get('tn', 0)}")
        print(f"  False Negative: {cm.get('fn', 0)}")
        print("-" * 70)
        print(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1_score', 0.0):.4f}")
        print("=" * 70)
        
        # Total videos processed
        print(f"Total videos processed: {performance['total_videos']}")
        print("=" * 70)

    def _convert_numpy_to_serializable(self, obj: Any) -> Any:
        """Converts numpy types to JSON serializable types."""
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, dict): return {k: self._convert_numpy_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._convert_numpy_to_serializable(i) for i in obj]
        return obj

def process_videos_on_gpu(video_files: List[str], windows_dir: str, config: Any, gpu_id: int):
    """Independent GPU processing function for multiprocessing with parallel processing support."""
    def signal_handler(signum, frame):
        print(f"GPU {gpu_id} - Received signal {signum}, cleaning up...")
        _cleanup_gpu_process(gpu_id)
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        device = f'cuda:{gpu_id}'
        print(f"GPU {gpu_id} - Initializing models...")
        pose_extractor = EnhancedRTMOPoseExtractor(
            config_file=config.detector_config, checkpoint_file=config.detector_checkpoint, device=device, gpu_ids=[gpu_id],
            score_thr=config.score_thr, nms_thr=config.nms_thr, track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh, track_max_disappeared=config.track_max_disappeared,
            track_min_hits=config.track_min_hits, quality_threshold=config.quality_threshold,
            min_track_length=config.min_track_length, weights=config.get_weights()
        )
        action_model = init_recognizer(config.action_config, config.action_checkpoint, device=device)
        error_logger = ProcessingErrorLogger(config.output_dir, f"gpu_{gpu_id}")
        
        # 병렬 처리 설정
        enable_video_parallel = getattr(config, 'enable_video_parallel', True)
        enable_window_parallel = getattr(config, 'enable_window_parallel', True)
        video_workers = getattr(config, 'video_workers', min(mp.cpu_count() // len(getattr(config, 'gpu', [0])) if hasattr(config, 'gpu') else 2, 2))
        
        all_window_results, all_video_results = [], []
        print(f"GPU {gpu_id} - Started processing {len(video_files)} videos")
        
        if enable_video_parallel and len(video_files) > 1:
            print(f"GPU {gpu_id} - Using parallel video processing with {video_workers} workers")
            # 비디오 레벨 병렬 처리
            try:
                with ThreadPoolExecutor(max_workers=video_workers) as executor:
                    future_to_video = {}
                    for i, video_path in enumerate(video_files):
                        future = executor.submit(
                            _process_single_video_parallel, 
                            video_path, windows_dir, config, pose_extractor, action_model, 
                            error_logger, is_first_video=(i == 0), 
                            enable_window_parallel=enable_window_parallel
                        )
                        future_to_video[future] = video_path
                    
                    for future in as_completed(future_to_video):
                        video_path = future_to_video[future]
                        try:
                            video_result = future.result()
                            if video_result:
                                all_window_results.extend(video_result['window_results'])
                                all_video_results.append(video_result['video_summary'])
                                print(f"GPU {gpu_id} - Completed: {Path(video_path).name}")
                        except Exception as e:
                            error_logger.log_general_error("parallel_video_processing_error", str(e), capture_exception_info())
                            print(f"GPU {gpu_id} - Error in parallel processing {video_path}: {e}")
            except Exception as e:
                print(f"GPU {gpu_id} - Error in parallel processing, falling back to sequential: {e}")
                # 순차 처리로 폴백
                for i, video_path in enumerate(video_files):
                    try:
                        video_result = _process_single_video(video_path, windows_dir, config, pose_extractor, action_model, error_logger, is_first_video=(i == 0))
                        if video_result:
                            all_window_results.extend(video_result['window_results'])
                            all_video_results.append(video_result['video_summary'])
                    except Exception as e:
                        error_logger.log_general_error("video_processing_error", str(e), capture_exception_info())
                        print(f"GPU {gpu_id} - Error processing {video_path}: {e}")
        else:
            # 기존 순차 처리
            print(f"GPU {gpu_id} - Using sequential video processing")
            for i, video_path in enumerate(video_files):
                try:
                    print(f"GPU {gpu_id} - Processing video {i+1}/{len(video_files)}: {Path(video_path).name}")
                    video_result = _process_single_video(video_path, windows_dir, config, pose_extractor, action_model, error_logger, is_first_video=(i == 0))
                    if video_result:
                        all_window_results.extend(video_result['window_results'])
                        all_video_results.append(video_result['video_summary'])
                except Exception as e:
                    error_logger.log_general_error("video_processing_error", str(e), capture_exception_info())
                    print(f"GPU {gpu_id} - Error processing {video_path}: {e}")
        
        print(f"GPU {gpu_id} - Completed processing {len(all_video_results)} videos successfully")
        return all_window_results, all_video_results
    finally:
        _cleanup_gpu_process(gpu_id)

def _cleanup_gpu_process(gpu_id: int):
    """GPU process cleanup function."""
    try:
        if 'torch' in sys.modules and torch.cuda.is_available():
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"GPU {gpu_id} - Error during cleanup: {e}")

atexit.register(lambda: _cleanup_gpu_process(os.getpid()))

def _process_single_video_parallel(video_path: str, windows_dir: str, config: Any, 
                                 pose_extractor: Any, action_model: Any, error_logger: Any, 
                                 is_first_video: bool = False, enable_window_parallel: bool = True) -> Optional[Dict[str, Any]]:
    """병렬 처리를 지원하는 단일 비디오 처리 함수"""
    try:
        video_name = Path(video_path).stem
        label_folder = Path(video_path).parent.name
        true_label = 1 if label_folder.lower() == 'fight' else 0

        pose_results = pose_extractor.extract_poses_only(video_path)
        if not pose_results:
            return None

        pose_results = _augment_frames_static(pose_results, config.clip_len)
        if not pose_results:
            return None

        if enable_window_parallel:
            window_results = _process_windows_parallel_static(
                pose_results, video_name, true_label, label_folder, 
                config, pose_extractor, action_model, is_first_video
            )
        else:
            window_results = _process_windows_sequential_static(
                pose_results, video_name, true_label, label_folder,
                config, pose_extractor, action_model, is_first_video
            )

        if not window_results:
            return None
        
        _save_window_results_static(window_results, windows_dir, label_folder, video_name)
        video_prediction = _apply_consecutive_event_rule_static(window_results, config.consecutive_event_threshold)
        
        video_summary = {
            'video_name': video_name, 
            'true_label': true_label, 
            'label_folder': label_folder,
            'video_prediction': video_prediction, 
            'window_count': len(window_results),
            'fight_windows': sum(1 for w in window_results if w['predicted_label'] == 1),
            'avg_prediction_score': np.mean([w['prediction'] for w in window_results]) if window_results else 0.0,
            'max_prediction_score': np.max([w['prediction'] for w in window_results]) if window_results else 0.0
        }
        
        return {'window_results': window_results, 'video_summary': video_summary}
    except Exception as e:
        error_logger.log_general_error(f"parallel_video_processing_error_{video_name}", str(e), capture_exception_info())
        print(f"Error in parallel video processing {video_path}: {e}")
        return None

def _process_windows_parallel_static(pose_results: List, video_name: str, true_label: int, 
                                   label_folder: str, config: Any, pose_extractor: Any, 
                                   action_model: Any, is_first_video: bool = False) -> List[Dict]:
    """정적 함수: 윈도우 레벨 병렬 처리"""
    window_workers = getattr(config, 'window_workers', min(mp.cpu_count(), 8))
    
    # 윈도우 파라미터 생성
    window_params = []
    for start_frame in range(0, len(pose_results) - config.clip_len + 1, config.inference_stride):
        end_frame = start_frame + config.clip_len
        window_idx = start_frame // config.inference_stride
        debug_enabled = config.debug_mode and is_first_video and window_idx == 0
        
        window_params.append({
            'pose_window': pose_results[start_frame:end_frame],
            'start_frame': start_frame,
            'end_frame': end_frame,
            'window_idx': window_idx,
            'video_name': video_name,
            'true_label': true_label,
            'label_folder': label_folder,
            'debug_enabled': debug_enabled
        })
    
    try:
        with ThreadPoolExecutor(max_workers=window_workers) as executor:
            future_to_window = {
                executor.submit(
                    _process_single_window_static, 
                    params, config, pose_extractor, action_model
                ): params['window_idx'] 
                for params in window_params
            }
            
            # 결과 수집 (순서 보장)
            window_results_dict = {}
            for future in as_completed(future_to_window):
                window_idx = future_to_window[future]
                try:
                    result = future.result()
                    if result:
                        window_results_dict[window_idx] = result
                except Exception as e:
                    print(f"Window {window_idx} processing failed: {e}")
            
            # 윈도우 인덱스 순으로 정렬하여 반환
            return [window_results_dict[idx] for idx in sorted(window_results_dict.keys())]
    
    except Exception as e:
        print(f"Error in parallel window processing: {e}")
        # 오류 발생시 순차 처리로 폴백
        return _process_windows_sequential_static(
            pose_results, video_name, true_label, label_folder,
            config, pose_extractor, action_model, is_first_video
        )

def _process_windows_sequential_static(pose_results: List, video_name: str, true_label: int,
                                     label_folder: str, config: Any, pose_extractor: Any,
                                     action_model: Any, is_first_video: bool = False) -> List[Dict]:
    """정적 함수: 윈도우 순차 처리"""
    window_results = []
    
    for start_frame in range(0, len(pose_results) - config.clip_len + 1, config.inference_stride):
        end_frame = start_frame + config.clip_len
        window_idx = start_frame // config.inference_stride
        try:
            debug_enabled = config.debug_mode and is_first_video and window_idx == 0
            
            tracked_window = pose_extractor.apply_tracking_to_poses(
                pose_results[start_frame:end_frame], start_frame, end_frame, window_idx)
            if not tracked_window:
                continue
            
            prediction = _predict_window_static(
                tracked_window, action_model, config.clip_len, 
                config.focus_person, debug=debug_enabled
            )
            
            window_results.append({
                'video_name': video_name,
                'true_label': true_label,
                'label_folder': label_folder,
                'window_idx': window_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'prediction': prediction,
                'predicted_label': 1 if prediction > config.classification_threshold else 0,
                'persons_count': len(tracked_window.get('annotation', {}).get('persons', {})),
                'pose_data': tracked_window
            })
            
            if debug_enabled:
                print(f"=== END DEBUG INFO ===\n")
                
        except Exception as e:
            print(f"Window {window_idx} failed: {e}")
    
    return window_results

def _process_single_window_static(params: Dict, config: Any, pose_extractor: Any, action_model: Any) -> Optional[Dict]:
    """정적 함수: 단일 윈도우 처리 (병렬 처리용)"""
    try:
        tracked_window = pose_extractor.apply_tracking_to_poses(
            params['pose_window'], 
            params['start_frame'], 
            params['end_frame'], 
            params['window_idx']
        )
        
        if not tracked_window:
            return None
        
        prediction = _predict_window_static(
            tracked_window, 
            action_model, 
            config.clip_len,
            config.focus_person, 
            debug=params['debug_enabled']
        )
        
        return {
            'video_name': params['video_name'],
            'true_label': params['true_label'],
            'label_folder': params['label_folder'],
            'window_idx': params['window_idx'],
            'start_frame': params['start_frame'],
            'end_frame': params['end_frame'],
            'prediction': prediction,
            'predicted_label': 1 if prediction > config.classification_threshold else 0,
            'persons_count': len(tracked_window.get('annotation', {}).get('persons', {})),
            'pose_data': tracked_window
        }
        
    except Exception as e:
        print(f"Single window processing failed for window {params['window_idx']}: {e}")
        return None

def _process_single_video(video_path: str, windows_dir: str, config: Any, 
                         pose_extractor: Any, action_model: Any, error_logger: Any, is_first_video: bool = False) -> Dict[str, Any]:
    """Static single video processing function for multiprocessing."""
    video_name = Path(video_path).stem
    label_folder = Path(video_path).parent.name
    # 대소문자 구분없이 fight 판단, normal/nonfight를 모두 non-fight로 처리
    true_label = 1 if label_folder.lower() == 'fight' else 0

    pose_results = pose_extractor.extract_poses_only(video_path)
    if not pose_results: return None

    pose_results = _augment_frames_static(pose_results, config.clip_len)
    if not pose_results: return None

    window_results = []
    for start_frame in range(0, len(pose_results) - config.clip_len + 1, config.inference_stride):
        end_frame = start_frame + config.clip_len
        window_idx = start_frame // config.inference_stride
        try:
            tracked_window = pose_extractor.apply_tracking_to_poses(
                pose_results[start_frame:end_frame], start_frame, end_frame, window_idx)
            if not tracked_window: continue

            # 첫 번째 비디오의 첫 번째 윈도우에서만 디버그 출력
            debug_enabled = config.debug_mode and is_first_video and window_idx == 0
            if debug_enabled:
                print(f"\n=== DEBUG INFO for {video_name} Window {window_idx} ===")

            prediction = _predict_window_static(tracked_window, action_model, config.clip_len, 
                                              config.focus_person, debug=debug_enabled)
            window_results.append({
                'video_name': video_name, 'true_label': true_label, 'label_folder': label_folder,
                'window_idx': window_idx, 'start_frame': start_frame, 'end_frame': end_frame,
                'prediction': prediction, 'predicted_label': 1 if prediction > config.classification_threshold else 0,
                'persons_count': len(tracked_window.get('annotation', {}).get('persons', {})),
                'pose_data': tracked_window  # 시각화를 위한 포즈 데이터 추가
            })
            
            if debug_enabled:
                print(f"=== END DEBUG INFO ===\n")
                
        except Exception as e:
            print(f"    Window {window_idx} failed: {e}")
            error_logger.log_general_error(f"window_processing_error_{video_name}_{window_idx}", str(e), capture_exception_info())

    if not window_results: return None
    
    _save_window_results_static(window_results, windows_dir, label_folder, video_name)
    video_prediction = _apply_consecutive_event_rule_static(window_results, config.consecutive_event_threshold)
    
    video_summary = {
        'video_name': video_name, 'true_label': true_label, 'label_folder': label_folder,
        'video_prediction': video_prediction, 'window_count': len(window_results),
        'fight_windows': sum(1 for w in window_results if w['predicted_label'] == 1),
        'avg_prediction_score': np.mean([w['prediction'] for w in window_results]) if window_results else 0.0,
        'max_prediction_score': np.max([w['prediction'] for w in window_results]) if window_results else 0.0
    }
    return {'window_results': window_results, 'video_summary': video_summary}

def _augment_frames_static(pose_results: List, clip_len: int) -> List:
    """Static function: Augments frames."""
    original_length = len(pose_results)
    if original_length >= clip_len: 
        return pose_results
    if original_length == 0: 
        return []
    
    print(f"    Applying padding: {original_length} -> {clip_len} frames")
    augmented_results = list(pose_results)
    while len(augmented_results) < clip_len:
        augmented_results.extend(pose_results)
    return augmented_results[:clip_len]

def _predict_window_static(tracked_window: dict, action_model: Any, clip_len: int, focus_person: int = 4, debug: bool = False) -> float:
    """Static function: Predicts action for a single window."""
    try:
        if debug:
            print(f"[DEBUG] Starting prediction for window")
            print(f"[DEBUG] tracked_window keys: {list(tracked_window.keys())}")
        
        persons_data = tracked_window.get('annotation', {}).get('persons', {})
        if not persons_data:
            if debug: print(f"[DEBUG] No persons data found")
            return 0.0

        persons_list = sorted(persons_data.values(), key=lambda p: p.get('rank', float('inf')))
        num_persons_to_feed = focus_person
        num_keypoints = 17

        if debug:
            print(f"[DEBUG] Processing {len(persons_list)} persons, feeding {num_persons_to_feed}")
            print(f"[DEBUG] clip_len: {clip_len}, num_keypoints: {num_keypoints}")

        keypoint = np.zeros((num_persons_to_feed, clip_len, num_keypoints, 2), dtype=np.float32)
        keypoint_score = np.zeros((num_persons_to_feed, clip_len, num_keypoints), dtype=np.float32)

        valid_persons = 0
        for i, person in enumerate(persons_list[:num_persons_to_feed]):
            if debug:
                print(f"[DEBUG] Processing person {i}")
                person_keys = list(person.keys()) if isinstance(person, dict) else "not dict"
                print(f"[DEBUG] Person {i} keys: {person_keys}")
            
            # keypoint 데이터는 person 객체 직접 레벨에 있음
            person_kps = person.get('keypoint')
            person_scores = person.get('keypoint_score')
            
            if debug:
                print(f"[DEBUG] Person {i}: Direct keypoint access")

            if debug:
                print(f"[DEBUG] Person {i}: keypoint exists: {person_kps is not None}, score exists: {person_scores is not None}")

            if person_kps is not None and person_scores is not None:
                if debug:
                    print(f"[DEBUG] Person {i}: keypoint shape: {person_kps.shape}, score shape: {person_scores.shape}")
                    print(f"[DEBUG] Person {i}: keypoint type: {type(person_kps)}, score type: {type(person_scores)}")
                    print(f"[DEBUG] Person {i}: keypoint min/max: {np.min(person_kps):.3f}/{np.max(person_kps):.3f}")
                    print(f"[DEBUG] Person {i}: score min/max: {np.min(person_scores):.3f}/{np.max(person_scores):.3f}")
                
                # person_kps 형태 확인 및 처리
                original_shape = person_kps.shape
                if len(person_kps.shape) == 4:  # (1, T, V, C)
                    person_kps = person_kps.squeeze(0)  # (T, V, C)
                if len(person_scores.shape) == 3:  # (1, T, V)
                    person_scores = person_scores.squeeze(0)  # (T, V)
                
                if debug and original_shape != person_kps.shape:
                    print(f"[DEBUG] Person {i}: Shape changed from {original_shape} to {person_kps.shape}")
                
                t_dim = person_kps.shape[0] if len(person_kps.shape) >= 2 else 0
                
                if debug:
                    print(f"[DEBUG] Person {i}: t_dim={t_dim}, clip_len={clip_len}")
                
                if t_dim == clip_len:
                    keypoint[i] = person_kps
                    keypoint_score[i] = person_scores
                    valid_persons += 1
                    if debug:
                        print(f"[DEBUG] Person {i}: Successfully assigned keypoints")
                elif t_dim > 0:
                    # 길이가 다른 경우 처리
                    if t_dim < clip_len:
                        # 짧은 경우: 반복으로 채우기
                        repeat_factor = clip_len // t_dim
                        remainder = clip_len % t_dim
                        repeated_kps = np.tile(person_kps, (repeat_factor, 1, 1))
                        if remainder > 0:
                            repeated_kps = np.concatenate([repeated_kps, person_kps[:remainder]], axis=0)
                        keypoint[i] = repeated_kps
                        
                        repeated_scores = np.tile(person_scores, (repeat_factor, 1))
                        if remainder > 0:
                            repeated_scores = np.concatenate([repeated_scores, person_scores[:remainder]], axis=0)
                        keypoint_score[i] = repeated_scores
                        valid_persons += 1
                        if debug:
                            print(f"[DEBUG] Person {i}: Repeated from {t_dim} to {clip_len} frames")
                    else:
                        # 긴 경우: 잘라내기
                        keypoint[i] = person_kps[:clip_len]
                        keypoint_score[i] = person_scores[:clip_len]
                        valid_persons += 1
                        if debug:
                            print(f"[DEBUG] Person {i}: Truncated from {t_dim} to {clip_len} frames")
                else:
                    if debug:
                        print(f"[DEBUG] Person {i}: Invalid t_dim={t_dim}")
            else:
                if debug:
                    print(f"[DEBUG] Person {i}: No valid keypoint data (kp: {person_kps is not None}, score: {person_scores is not None})")

        if debug:
            print(f"[DEBUG] Valid persons: {valid_persons}/{num_persons_to_feed}")
            print(f"[DEBUG] Final keypoint shape: {keypoint.shape}")
            nonzero_ratio = np.count_nonzero(keypoint) / keypoint.size
            print(f"[DEBUG] Keypoint non-zero ratio: {nonzero_ratio:.4f}")
            if nonzero_ratio > 0:
                print(f"[DEBUG] Keypoint value range: {np.min(keypoint[keypoint!=0]):.3f} ~ {np.max(keypoint):.3f}")

        # 유효한 person이 없으면 0 반환
        if valid_persons == 0:
            if debug: print("[DEBUG] No valid persons found, returning 0.0")
            return 0.0

        # 모든 keypoint가 0이면 0 반환
        if np.all(keypoint == 0):
            if debug: print("[DEBUG] All keypoints are zero, returning 0.0")
            return 0.0

        data_sample = {
            'keypoint': keypoint, # M, T, V, C 형태 유지 (MMAction2가 내부에서 변환)
            'keypoint_score': keypoint_score, # M, T, V 형태 유지
            'total_frames': clip_len,
            'img_shape': tracked_window.get('video_info', {}).get('img_shape', (1080, 1920)),
            'start_index': 0,
            'modality': 'Pose',
            'label': -1
        }

        if debug:
            print(f"[DEBUG] Data sample keypoint shape: {data_sample['keypoint'].shape}")
            print(f"[DEBUG] Data sample keypoint_score shape: {data_sample['keypoint_score'].shape}")
            print(f"[DEBUG] Data sample img_shape: {data_sample['img_shape']}")
            print(f"[DEBUG] Data sample total_frames: {data_sample['total_frames']}")
            print(f"[DEBUG] Starting model inference...")

        with torch.no_grad():
            result = inference_recognizer(action_model, data_sample)
        
        if debug:
            print(f"[DEBUG] Model inference completed")
            print(f"[DEBUG] Result type: {type(result)}")
            if hasattr(result, 'pred_score'):
                print(f"[DEBUG] Result has pred_score attribute")
            else:
                print(f"[DEBUG] Result attributes: {dir(result)}")
        
        scores = result.pred_score.tolist() if hasattr(result, 'pred_score') else [0.0]
        prediction = float(scores[1]) if len(scores) > 1 else float(scores[0])
        
        if debug:
            print(f"[DEBUG] Model prediction: {prediction}, scores: {scores}")
            print(f"[DEBUG] Scores length: {len(scores)}")
        
        return prediction
    except Exception as e:
        print(f"[ERROR] Error in model inference: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def _apply_consecutive_event_rule_static(window_results: List[Dict], consecutive_event_threshold: int) -> int:
    """Static function: Applies the consecutive event rule with enhanced short video handling."""
    if not window_results: return 0
    
    total_windows = len(window_results)
    fight_windows = [w for w in window_results if w['predicted_label'] == 1]
    fight_count = len(fight_windows)
    
    # 짧은 비디오 처리: 윈도우 수가 임계값보다 적을 때
    if total_windows < consecutive_event_threshold:
        # 윈도우가 1개만 있으면 그 결과를 그대로 사용
        if total_windows == 1:
            return window_results[0]['predicted_label']
        
        # 윈도우가 2-3개 있으면, 과반수 이상이 fight여야 fight로 판정
        fight_ratio = fight_count / total_windows
        return 1 if fight_ratio >= 0.5 else 0
    
    # 일반적인 경우: 연속된 fight 윈도우 찾기
    max_consecutive = 0
    consecutive_count = 0
    for window in window_results:
        if window['predicted_label'] == 1:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        else:
            consecutive_count = 0
    
    # 연속 임계값 충족 시 fight로 판정
    if max_consecutive >= consecutive_event_threshold:
        return 1
    
    # 연속 임계값을 충족하지 못하더라도, 전체 fight 비율이 높으면 fight로 판정
    fight_ratio = fight_count / total_windows
    high_fight_ratio_threshold = 0.7  # 70% 이상이 fight면 fight로 판정
    
    return 1 if fight_ratio >= high_fight_ratio_threshold else 0

def _save_window_results_static(window_results: List[Dict], windows_dir: str, label_folder: str, video_name: str):
    """Static function: Saves window-level results."""
    category_dir = os.path.join(windows_dir, label_folder)
    os.makedirs(category_dir, exist_ok=True)
    with open(os.path.join(category_dir, f"{video_name}_windows.pkl"), 'wb') as f:
        pickle.dump(window_results, f)

def main():
    """Main execution function with proper signal handling."""
    # Argument parsing
    parser = argparse.ArgumentParser(description="Fight Detection Inference Pipeline")
    parser.add_argument('--config', type=str, default='configs/inference_config.py',
                       help='Configuration file path')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from last checkpoint (skip already processed videos)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of all videos (ignore existing results)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resume mode (process all videos)')
    
    # 병렬 처리 관련 인수들
    parser.add_argument('--enable-video-parallel', action='store_true', default=True,
                       help='Enable video-level parallel processing')
    parser.add_argument('--disable-video-parallel', action='store_true',
                       help='Disable video-level parallel processing')
    parser.add_argument('--enable-window-parallel', action='store_true', default=True,
                       help='Enable window-level parallel processing')
    parser.add_argument('--disable-window-parallel', action='store_true',
                       help='Disable window-level parallel processing')
    parser.add_argument('--video-workers', type=int, default=None,
                       help='Number of video processing workers (default: auto)')
    parser.add_argument('--window-workers', type=int, default=None,
                       help='Number of window processing workers (default: auto)')
    
    # Support for config overrides (key=value format)
    parser.add_argument('overrides', nargs='*', 
                       help='Config overrides in key=value format (e.g., gpu=0 debug_mode=True)')
    
    args = parser.parse_args()
    
    # Argument validation
    if args.resume and args.force:
        print("Error: --resume and --force options are mutually exclusive")
        print("--resume: Skip already processed videos (default)")
        print("--force: Reprocess all videos")
        sys.exit(1)
    
    # Handle no-resume flag
    resume_mode = args.resume and not args.no_resume and not args.force
    
    # Parse config overrides
    overrides = {}
    for override in args.overrides:
        if '=' in override:
            key, value = override.split('=', 1)
            overrides[key] = value
    
    # 병렬 처리 플래그 처리
    if args.disable_video_parallel:
        overrides['enable_video_parallel'] = 'False'
    elif args.enable_video_parallel:
        overrides['enable_video_parallel'] = 'True'
    
    if args.disable_window_parallel:
        overrides['enable_window_parallel'] = 'False'
    elif args.enable_window_parallel:
        overrides['enable_window_parallel'] = 'True'
    
    if args.video_workers is not None:
        overrides['video_workers'] = str(args.video_workers)
    
    if args.window_workers is not None:
        overrides['window_workers'] = str(args.window_workers)
    
    processor = None
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        if processor and hasattr(processor, 'gpu_ids'):
            for gpu_id in processor.gpu_ids:
                _cleanup_gpu_process(gpu_id)
        print("Shutdown completed.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        config = load_config(args.config, overrides=overrides)
        
        # Print execution mode
        print("=" * 80)
        print("FIGHT DETECTION INFERENCE PIPELINE")
        print("=" * 80)
        print(f"Config file: {args.config}")
        if args.force:
            print("Mode: Force reprocessing all videos")
        elif resume_mode:
            print("Mode: Resume (skip already processed videos)")
        else:
            print("Mode: Process all videos")
        print()
        
        config.print_config()
        processor = FightInferenceProcessor(config)
        processor.run_inference(resume_mode=resume_mode, force_reprocess=args.force)
        print("Inference pipeline completed successfully!")
    except Exception as e:
        print(f"An error occurred in main process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()