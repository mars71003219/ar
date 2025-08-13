"""
멀티프로세스 처리 관리자

CPU 멀티프로세스를 활용한 병렬 처리를 제공합니다.
num_workers 파라미터로 프로세스 수를 제어할 수 있습니다.
"""

import multiprocessing as mp
import concurrent.futures
import queue
import logging
import time
import os
import pickle
import traceback
from typing import List, Any, Dict, Callable, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass
class ProcessTask:
    """프로세스 작업 정의"""
    task_id: str
    func_name: str
    args: tuple
    kwargs: dict
    priority: int = 0  # 낮을수록 우선순위 높음


@dataclass
class ProcessResult:
    """프로세스 결과"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0


class WorkerProcess:
    """워커 프로세스 클래스"""
    
    def __init__(self, worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                 shutdown_event: mp.Event):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.shutdown_event = shutdown_event
        self.process = None
    
    def start(self):
        """프로세스 시작"""
        self.process = mp.Process(target=self._worker_loop)
        self.process.start()
        return self.process.pid
    
    def _worker_loop(self):
        """워커 프로세스 메인 루프"""
        logging.info(f"Worker {self.worker_id} started (PID: {os.getpid()})")
        
        while not self.shutdown_event.is_set():
            try:
                # 작업 가져오기 (타임아웃 설정)
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # 종료 신호
                    break
                
                # 작업 처리
                result = self._process_task(task)
                
                # 결과 전송
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {self.worker_id} error: {str(e)}")
                error_result = ProcessResult(
                    task_id="unknown",
                    success=False,
                    error=str(e)
                )
                self.result_queue.put(error_result)
        
        logging.info(f"Worker {self.worker_id} shutdown")
    
    def _process_task(self, task: ProcessTask) -> ProcessResult:
        """작업 처리"""
        start_time = time.time()
        
        try:
            # 함수 이름으로 함수 찾기 (동적 임포트)
            func = self._get_function(task.func_name)
            
            # 함수 실행
            result = func(*task.args, **task.kwargs)
            
            processing_time = time.time() - start_time
            
            return ProcessResult(
                task_id=task.task_id,
                success=True,
                result=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Task {task.task_id} failed: {str(e)}\n{traceback.format_exc()}"
            
            return ProcessResult(
                task_id=task.task_id,
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _get_function(self, func_name: str) -> Callable:
        """함수 이름으로 함수 객체 가져오기"""
        # 전역 함수 등록소
        if hasattr(self, '_function_registry'):
            if func_name in self._function_registry:
                return self._function_registry[func_name]
        
        # 동적 임포트 시도
        if '.' in func_name:
            module_path, func_name = func_name.rsplit('.', 1)
            try:
                import importlib
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Cannot import function {func_name}: {str(e)}")
        
        # 내장 함수 또는 글로벌에서 찾기
        if func_name in globals():
            return globals()[func_name]
        
        raise ValueError(f"Function {func_name} not found")
    
    def terminate(self):
        """프로세스 강제 종료"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()


class MultiprocessManager:
    """멀티프로세스 관리자"""
    
    def __init__(self, num_workers: int = None, max_queue_size: int = 1000):
        """
        Args:
            num_workers: 워커 프로세스 수 (None이면 CPU 코어 수)
            max_queue_size: 최대 큐 크기
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.max_queue_size = max_queue_size
        
        # 멀티프로세싱 큐
        self.task_queue = mp.Queue(maxsize=max_queue_size)
        self.result_queue = mp.Queue()
        self.shutdown_event = mp.Event()
        
        # 워커 프로세스들
        self.workers: List[WorkerProcess] = []
        self.is_running = False
        
        # 결과 수집 스레드
        self.result_collector_thread = None
        self.results: Dict[str, ProcessResult] = {}
        self.result_lock = threading.Lock()
        
        # 통계
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0.0
        }
        
        logging.info(f"MultiprocessManager initialized with {self.num_workers} workers")
    
    def start(self):
        """매니저 시작"""
        if self.is_running:
            logging.warning("MultiprocessManager is already running")
            return
        
        logging.info(f"Starting {self.num_workers} worker processes...")
        
        # 워커 프로세스들 시작
        for i in range(self.num_workers):
            worker = WorkerProcess(i, self.task_queue, self.result_queue, self.shutdown_event)
            pid = worker.start()
            self.workers.append(worker)
            logging.info(f"Started worker {i} (PID: {pid})")
        
        # 결과 수집 스레드 시작
        self.result_collector_thread = threading.Thread(target=self._collect_results)
        self.result_collector_thread.daemon = True
        self.result_collector_thread.start()
        
        self.is_running = True
        logging.info("MultiprocessManager started successfully")
    
    def stop(self, timeout: float = 10.0):
        """매니저 중지"""
        if not self.is_running:
            return
        
        logging.info("Stopping MultiprocessManager...")
        
        # 종료 신호 설정
        self.shutdown_event.set()
        
        # 워커들에게 종료 신호 전송
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # 워커 프로세스들 종료 대기
        for worker in self.workers:
            if worker.process:
                worker.process.join(timeout=timeout/len(self.workers))
                if worker.process.is_alive():
                    logging.warning(f"Worker {worker.worker_id} did not shutdown gracefully, terminating")
                    worker.terminate()
        
        self.is_running = False
        logging.info("MultiprocessManager stopped")
    
    def _collect_results(self):
        """결과 수집 스레드"""
        while not self.shutdown_event.is_set():
            try:
                result = self.result_queue.get(timeout=1.0)
                
                with self.result_lock:
                    self.results[result.task_id] = result
                    
                    # 통계 업데이트
                    self.stats['completed_tasks'] += 1
                    if not result.success:
                        self.stats['failed_tasks'] += 1
                    self.stats['total_processing_time'] += result.processing_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error collecting results: {str(e)}")
    
    def submit_task(self, func_name: str, *args, task_id: str = None, 
                   priority: int = 0, **kwargs) -> str:
        """작업 제출
        
        Args:
            func_name: 실행할 함수 이름 (모듈 경로 포함 가능)
            args: 함수 인자
            task_id: 작업 ID (None이면 자동 생성)
            priority: 우선순위 (낮을수록 우선)
            kwargs: 함수 키워드 인자
            
        Returns:
            작업 ID
        """
        if not self.is_running:
            raise RuntimeError("MultiprocessManager is not running")
        
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}_{self.stats['total_tasks']}"
        
        task = ProcessTask(
            task_id=task_id,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        try:
            self.task_queue.put(task, timeout=5.0)
            self.stats['total_tasks'] += 1
            return task_id
        except queue.Full:
            raise RuntimeError("Task queue is full")
    
    def submit_batch(self, tasks: List[Tuple[str, tuple, dict]]) -> List[str]:
        """배치 작업 제출
        
        Args:
            tasks: (func_name, args, kwargs) 튜플 리스트
            
        Returns:
            작업 ID 리스트
        """
        task_ids = []
        
        for i, (func_name, args, kwargs) in enumerate(tasks):
            task_id = self.submit_task(func_name, *args, **kwargs)
            task_ids.append(task_id)
        
        return task_ids
    
    def get_result(self, task_id: str, timeout: float = None) -> ProcessResult:
        """결과 가져오기
        
        Args:
            task_id: 작업 ID
            timeout: 대기 시간 (None이면 무한 대기)
            
        Returns:
            프로세스 결과
        """
        start_time = time.time()
        
        while True:
            with self.result_lock:
                if task_id in self.results:
                    return self.results.pop(task_id)
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timeout")
            
            time.sleep(0.01)
    
    def get_batch_results(self, task_ids: List[str], timeout: float = None) -> List[ProcessResult]:
        """배치 결과 가져오기
        
        Args:
            task_ids: 작업 ID 리스트
            timeout: 대기 시간
            
        Returns:
            프로세스 결과 리스트
        """
        results = []
        
        for task_id in task_ids:
            result = self.get_result(task_id, timeout)
            results.append(result)
        
        return results
    
    def process_batch_sync(self, func_name: str, data_list: List[Any], 
                          timeout: float = None) -> List[Any]:
        """동기 배치 처리
        
        Args:
            func_name: 실행할 함수 이름
            data_list: 처리할 데이터 리스트
            timeout: 대기 시간
            
        Returns:
            처리 결과 리스트
        """
        # 작업 제출
        task_ids = []
        for data in data_list:
            task_id = self.submit_task(func_name, data)
            task_ids.append(task_id)
        
        # 결과 수집
        results = self.get_batch_results(task_ids, timeout)
        
        # 결과 정렬 및 반환
        processed_results = []
        for result in results:
            if result.success:
                processed_results.append(result.result)
            else:
                logging.error(f"Task failed: {result.error}")
                processed_results.append(None)
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self.result_lock:
            avg_time = (self.stats['total_processing_time'] / self.stats['completed_tasks'] 
                       if self.stats['completed_tasks'] > 0 else 0.0)
            
            return {
                'num_workers': self.num_workers,
                'total_tasks': self.stats['total_tasks'],
                'completed_tasks': self.stats['completed_tasks'],
                'failed_tasks': self.stats['failed_tasks'],
                'pending_tasks': self.stats['total_tasks'] - self.stats['completed_tasks'],
                'average_processing_time': avg_time,
                'total_processing_time': self.stats['total_processing_time'],
                'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else -1
            }
    
    def is_busy(self) -> bool:
        """처리 중인 작업이 있는지 확인"""
        pending = self.stats['total_tasks'] - self.stats['completed_tasks']
        return pending > 0
    
    def wait_completion(self, timeout: float = None):
        """모든 작업 완료 대기"""
        start_time = time.time()
        
        while self.is_busy():
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Tasks completion timeout")
            time.sleep(0.1)
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop()


# 전역 매니저 인스턴스
_global_manager = None


def get_global_multiprocess_manager(num_workers: int = None) -> MultiprocessManager:
    """전역 멀티프로세스 매니저 인스턴스 반환"""
    global _global_manager
    
    if _global_manager is None:
        _global_manager = MultiprocessManager(num_workers=num_workers)
        _global_manager.start()
    
    return _global_manager


def cleanup_global_manager():
    """전역 매니저 정리"""
    global _global_manager
    
    if _global_manager:
        _global_manager.stop()
        _global_manager = None


def parallel_process(func_name: str, data_list: List[Any], num_workers: int = None,
                    timeout: float = None) -> List[Any]:
    """병렬 처리 편의 함수
    
    Args:
        func_name: 실행할 함수 이름
        data_list: 처리할 데이터 리스트
        num_workers: 워커 수
        timeout: 대기 시간
        
    Returns:
        처리 결과 리스트
    """
    with MultiprocessManager(num_workers=num_workers) as manager:
        return manager.process_batch_sync(func_name, data_list, timeout)


# 멀티프로세스용 함수들 (예제)
def process_video_frame(frame_data):
    """비디오 프레임 처리 예제 함수"""
    import numpy as np
    # 예제: 간단한 이미지 처리
    if isinstance(frame_data, np.ndarray):
        # 그레이스케일 변환
        if len(frame_data.shape) == 3:
            gray = np.mean(frame_data, axis=2)
            return gray
    return frame_data


def process_pose_data(pose_data):
    """포즈 데이터 처리 예제 함수"""
    # 예제: 포즈 데이터 정규화
    if isinstance(pose_data, dict) and 'keypoints' in pose_data:
        keypoints = pose_data['keypoints']
        # 간단한 정규화
        normalized = [[x/640, y/640, conf] for x, y, conf in keypoints]
        pose_data['keypoints'] = normalized
    return pose_data