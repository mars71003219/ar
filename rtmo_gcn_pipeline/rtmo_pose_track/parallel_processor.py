"""
Parallel Processing Module for Enhanced STGCN++ Annotation Generation
병렬 처리 및 성능 최적화 모듈

주요 기능:
1. 다중 프로세스 비디오 처리
2. 메모리 효율적 배치 처리
3. 스마트 캐싱 시스템
4. 리소스 모니터링
"""

import os
import time
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from tqdm import tqdm

# CUDA 멀티프로세싱 호환성을 위해 spawn 방법 사용 (안전한 방식)
try:
    mp.set_start_method('spawn')
except RuntimeError:
    # 이미 설정된 경우 무시
    pass

from enhanced_rtmo_bytetrack_pose_extraction import (
    process_single_video, 
    FailureLogger,
    find_video_files
)


class ResourceMonitor:
    """시스템 리소스 모니터링"""
    
    def __init__(self, warning_threshold=80, critical_threshold=95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process()
        
    def get_system_status(self):
        """현재 시스템 상태 반환"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU 메모리 체크 (nvidia-ml-py 있을 경우)
            gpu_memory = self._get_gpu_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'gpu_memory_percent': gpu_memory,
                'status': self._determine_status(cpu_percent, memory.percent, gpu_memory)
            }
        except Exception as e:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_available_gb': 0,
                'gpu_memory_percent': 0,
                'status': 'unknown',
                'error': str(e)
            }
    
    def _get_gpu_memory(self):
        """GPU 메모리 사용률 체크"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.used / info.total) * 100
        except:
            return 0  # GPU 정보 없으면 0 반환
    
    def _determine_status(self, cpu_percent, memory_percent, gpu_memory_percent):
        """시스템 상태 결정"""
        max_usage = max(cpu_percent, memory_percent, gpu_memory_percent)
        
        if max_usage >= self.critical_threshold:
            return 'critical'
        elif max_usage >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def should_reduce_workers(self):
        """워커 수를 줄여야 하는지 판단"""
        status = self.get_system_status()
        return status['status'] in ['critical', 'warning']


class SmartCache:
    """스마트 캐싱 시스템"""
    
    def __init__(self, max_cache_size=1000):
        self.max_cache_size = max_cache_size
        self.bbox_distance_cache = {}
        self.movement_cache = {}
        self.access_count = defaultdict(int)
        
    def get_cached_distance(self, bbox1, bbox2):
        """바운딩박스 거리 계산 결과 캐싱"""
        key = self._make_bbox_key(bbox1, bbox2)
        
        if key not in self.bbox_distance_cache:
            if len(self.bbox_distance_cache) >= self.max_cache_size:
                self._cleanup_cache()
            
            distance = self._calculate_bbox_distance(bbox1, bbox2)
            self.bbox_distance_cache[key] = distance
        
        self.access_count[key] += 1
        return self.bbox_distance_cache[key]
    
    def _make_bbox_key(self, bbox1, bbox2):
        """바운딩박스 쌍에 대한 고유 키 생성"""
        # 정렬하여 순서 무관한 키 생성
        if bbox1[0] < bbox2[0]:
            return (*bbox1, *bbox2)
        else:
            return (*bbox2, *bbox1)
    
    def _calculate_bbox_distance(self, bbox1, bbox2):
        """실제 바운딩박스 거리 계산"""
        center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
        center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
        return np.linalg.norm(center1 - center2)
    
    def _cleanup_cache(self):
        """LRU 방식으로 캐시 정리"""
        # 사용 빈도가 낮은 항목들 제거
        sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) // 4  # 25% 제거
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.bbox_distance_cache:
                del self.bbox_distance_cache[key]
            del self.access_count[key]
    
    def get_cache_stats(self):
        """캐시 통계 반환"""
        return {
            'total_items': len(self.bbox_distance_cache),
            'max_size': self.max_cache_size,
            'hit_rate': sum(self.access_count.values()) / len(self.access_count) if self.access_count else 0
        }


class OptimizedBatchProcessor:
    """배치 기반 최적화된 프로세서"""
    
    def __init__(self, batch_size=8, max_tracks=20):
        self.batch_size = batch_size
        self.max_tracks = max_tracks
        self.cache = SmartCache()
        
    def process_video_batch(self, video_paths, args, failure_logger):
        """비디오들을 배치로 처리"""
        results = []
        
        for i in range(0, len(video_paths), self.batch_size):
            batch_videos = video_paths[i:i+self.batch_size]
            
            print(f"Processing batch {i//self.batch_size + 1}: {len(batch_videos)} videos")
            
            batch_results = []
            for video_path in batch_videos:
                result = process_single_video(video_path, args, failure_logger)
                batch_results.append((video_path, result))
            
            results.extend(batch_results)
            
            # 배치 간 메모리 정리
            self._cleanup_batch_memory()
        
        return results
    
    def _cleanup_batch_memory(self):
        """배치 처리 후 메모리 정리"""
        import gc
        gc.collect()
        
        # 캐시 크기 확인 및 정리
        cache_stats = self.cache.get_cache_stats()
        if cache_stats['total_items'] > cache_stats['max_size'] * 0.8:
            self.cache._cleanup_cache()


class ParallelVideoProcessor:
    """병렬 비디오 처리기"""
    
    def __init__(self, max_workers=None, enable_monitoring=True):
        self.max_workers = max_workers or min(mp.cpu_count() - 1, 8)
        self.enable_monitoring = enable_monitoring
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        self.batch_processor = OptimizedBatchProcessor()
        
    def process_videos_parallel(self, video_files, args):
        """병렬로 비디오 처리"""
        # 실패 로그 초기화
        failure_log_path = os.path.join(args.output_root, 'parallel_failed_videos.txt')
        failure_logger = FailureLogger(failure_log_path)
        
        print(f"Starting parallel processing with {self.max_workers} workers")
        print(f"Total videos to process: {len(video_files)}")
        
        if self.enable_monitoring:
            initial_status = self.resource_monitor.get_system_status()
            print(f"Initial system status: {initial_status['status']}")
            print(f"CPU: {initial_status['cpu_percent']:.1f}%, "
                  f"Memory: {initial_status['memory_percent']:.1f}%, "
                  f"GPU: {initial_status['gpu_memory_percent']:.1f}%")
        
        success_count = 0
        failed_videos = []
        
        # CUDA 호환성을 위해 spawn context 사용
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            # 작업 제출
            future_to_video = {
                executor.submit(self._process_single_video_wrapper, video, args, failure_log_path): video 
                for video in video_files
            }
            
            # 진행상황 추적
            with tqdm(total=len(video_files), desc="Processing videos") as pbar:
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_videos.append(video_path)
                        
                        pbar.set_postfix({
                            'Success': success_count,
                            'Failed': len(failed_videos),
                            'Rate': f"{success_count/(success_count + len(failed_videos))*100:.1f}%"
                        })
                        
                        # 리소스 모니터링
                        if self.enable_monitoring and self.resource_monitor.should_reduce_workers():
                            print(f"\nWarning: High resource usage detected")
                            status = self.resource_monitor.get_system_status()
                            print(f"Current status: {status}")
                        
                    except Exception as e:
                        failed_videos.append(video_path)
                        failure_logger.log_failure(video_path, f"Parallel processing error: {str(e)}")
                    
                    pbar.update(1)
        
        return {
            'total_videos': len(video_files),
            'successful': success_count,
            'failed': len(failed_videos),
            'failed_videos': failed_videos,
            'success_rate': success_count / len(video_files) * 100
        }
    
    @staticmethod
    def _process_single_video_wrapper(video_path, args, failure_log_path):
        """단일 비디오 처리 래퍼 (pickle 가능하도록)"""
        try:
            # CUDA 초기화 지연을 위해 torch import를 여기서 수행
            import torch
            # 각 워커에서 개별적으로 CUDA 설정
            if hasattr(args, 'device') and 'cuda' in args.device:
                torch.cuda.set_device(args.device)
            
            failure_logger = FailureLogger(failure_log_path)
            return process_single_video(video_path, args, failure_logger)
        except Exception as e:
            # 실패 로깅
            failure_logger = FailureLogger(failure_log_path)
            failure_logger.log_failure(video_path, f"Wrapper error: {str(e)}")
            return False
    
    def adaptive_worker_count(self, video_files, args):
        """적응적 워커 수 결정"""
        if not self.enable_monitoring:
            base_workers = self.max_workers
        else:
            status = self.resource_monitor.get_system_status()
            
            # 시스템 상태에 따른 워커 수 조정
            if status['status'] == 'critical':
                base_workers = max(1, self.max_workers // 4)
            elif status['status'] == 'warning':
                base_workers = max(2, self.max_workers // 2)
            else:
                base_workers = self.max_workers
        
        # 오버레이 생성시 워커 수 추가 조정 (메모리 집약적)
        if getattr(args, 'save_overlayfile', False):
            overlay_workers = max(1, base_workers // 2)
            print(f"Overlay mode detected: reducing workers from {base_workers} to {overlay_workers}")
            return overlay_workers
        
        return base_workers
    
    def estimate_processing_time(self, video_files, args):
        """처리 시간 추정 (비디오 속성 기반)"""
        if not video_files:
            return 0
        
        # 실제 처리 대신 비디오 속성 기반으로 추정
        sample_video = video_files[0]
        try:
            import cv2
            cap = cv2.VideoCapture(sample_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # 프레임 수와 해상도 기반 추정
            # 기본: 30fps 640x480 기준으로 1초당 0.5초 처리 시간
            base_time_per_second = 0.5
            resolution_factor = (width * height) / (640 * 480)
            
            # 오버레이 생성시 추가 시간
            overlay_factor = 1.5 if getattr(args, 'save_overlayfile', False) else 1.0
            
            video_duration = total_frames / fps if fps > 0 else 30
            single_video_time = video_duration * base_time_per_second * resolution_factor * overlay_factor
            
            print(f"Estimation based on: {total_frames} frames, {width}x{height}, overlay={getattr(args, 'save_overlayfile', False)}")
            
        except Exception as e:
            print(f"Warning: Could not analyze sample video: {e}")
            # 기본 추정값
            single_video_time = 90 if getattr(args, 'save_overlayfile', False) else 60
        
        # 병렬 처리 효율 고려 (오버레이 생성시 메모리 제약으로 효율 감소)
        parallel_efficiency = 0.6 if getattr(args, 'save_overlayfile', False) else 0.75
        estimated_time = (len(video_files) * single_video_time) / (self.max_workers * parallel_efficiency)
        
        return estimated_time


def run_parallel_processing(args):
    """병렬 처리 실행"""
    # 비디오 파일 찾기
    video_files = find_video_files(args.input)
    if not video_files:
        print(f"No video files found in {args.input}")
        return
    
    # 병렬 프로세서 초기화
    processor = ParallelVideoProcessor(
        max_workers=args.num_workers,
        enable_monitoring=True
    )
    
    # 처리 시간 추정
    estimated_time = processor.estimate_processing_time(video_files, args)
    print(f"Estimated processing time: {estimated_time/3600:.1f} hours")
    
    # 적응적 워커 수 결정
    optimal_workers = processor.adaptive_worker_count(video_files, args)
    print(f"Using {optimal_workers} workers (max: {processor.max_workers})")
    
    # 병렬 처리 실행
    start_time = time.time()
    results = processor.process_videos_parallel(video_files, args)
    actual_time = time.time() - start_time
    
    # 결과 출력
    print(f"\n{'='*50}")
    print(f"PARALLEL PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total videos: {results['total_videos']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Actual time: {actual_time/3600:.1f} hours")
    print(f"Time per video: {actual_time/results['total_videos']:.1f} seconds")
    
    if results['failed_videos']:
        print(f"\nFailed videos:")
        for video in results['failed_videos'][:10]:  # 처음 10개만 표시
            print(f"  - {os.path.basename(video)}")
        if len(results['failed_videos']) > 10:
            print(f"  ... and {len(results['failed_videos']) - 10} more")


if __name__ == "__main__":
    from enhanced_rtmo_bytetrack_pose_extraction import parse_args
    
    args = parse_args()
    run_parallel_processing(args)