"""
실시간 입력 처리 모듈
RTSP, 웹캠, 비디오 파일 등 다양한 입력 소스 지원
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, Union, Iterator, Tuple, Dict, Any
from pathlib import Path
from queue import Queue, Empty, Full
import re

from .import_utils import setup_logger

logger = setup_logger(__name__)


class RealtimeInputManager:
    """실시간 입력 관리자"""
    
    def __init__(self, 
                 input_source: Union[str, int],
                 buffer_size: int = 10,
                 target_fps: Optional[int] = None,
                 frame_skip: int = 0):
        """
        실시간 입력 관리자 초기화
        
        Args:
            input_source: 입력 소스 (파일 경로, RTSP URL, 웹캠 인덱스)
            buffer_size: 프레임 버퍼 크기
            target_fps: 목표 FPS (None이면 원본 FPS 사용)
            frame_skip: 건너뛸 프레임 수
        """
        self.input_source = input_source
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.frame_skip = frame_skip
        
        # 입력 소스 분석
        self.source_type = self._analyze_source_type(input_source)
        
        # 상태 관리
        self.is_running = False
        self.cap = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.capture_thread = None
        
        # 통계
        self.total_frames = 0
        self.dropped_frames = 0
        self.start_time = None
        
        # 비디오 정보
        self.video_info = {}
        
        logger.info(f"RealtimeInputManager initialized: {input_source} ({self.source_type})")
    
    def _analyze_source_type(self, source: Union[str, int]) -> str:
        """입력 소스 타입 분석"""
        if isinstance(source, int):
            return "webcam"
        elif isinstance(source, str):
            if source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                return "stream"
            elif self._is_video_file(source):
                return "file"
            else:
                # 숫자 문자열인 경우 웹캠으로 처리
                try:
                    int(source)
                    return "webcam"
                except ValueError:
                    return "unknown"
        return "unknown"
    
    def _is_video_file(self, path: str) -> bool:
        """비디오 파일 확장자 확인"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'}
        return Path(path).suffix.lower() in video_extensions
    
    def start(self) -> bool:
        """입력 캡처 시작"""
        if self.is_running:
            logger.warning("Already running")
            return True
        
        # VideoCapture 초기화
        if not self._initialize_capture():
            return False
        
        # 비디오 정보 수집
        self._collect_video_info()
        
        # 캡처 스레드 시작
        self.is_running = True
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Input capture started: {self.video_info}")
        return True
    
    def stop(self):
        """입력 캡처 중지"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 큐 정리
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Input capture stopped")
    
    def _initialize_capture(self) -> bool:
        """VideoCapture 초기화"""
        try:
            if self.source_type == "webcam":
                device_id = int(self.input_source)
                self.cap = cv2.VideoCapture(device_id)
            else:
                self.cap = cv2.VideoCapture(str(self.input_source))
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open input source: {self.input_source}")
                return False
            
            # 버퍼 크기 설정 (스트림의 경우 지연 최소화)
            if self.source_type == "stream":
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize capture: {e}")
            return False
    
    def _collect_video_info(self):
        """비디오 정보 수집"""
        if not self.cap:
            return
        
        self.video_info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'source_type': self.source_type
        }
        
        # 실시간 스트림의 경우 total_frames는 의미없음
        if self.source_type in ["stream", "webcam"]:
            self.video_info['total_frames'] = -1  # 무한
    
    def _capture_loop(self):
        """프레임 캡처 루프"""
        frame_interval = None
        if self.target_fps:
            frame_interval = 1.0 / self.target_fps
        
        last_frame_time = time.time()
        frame_skip_count = 0
        
        while self.is_running:
            try:
                # FPS 제한
                if frame_interval:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()
                
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == "file":
                        logger.info("End of video file reached")
                        break
                    else:
                        logger.warning("Failed to read frame, retrying...")
                        time.sleep(0.1)
                        continue
                
                self.total_frames += 1
                
                # 프레임 건너뛰기
                if self.frame_skip > 0:
                    if frame_skip_count < self.frame_skip:
                        frame_skip_count += 1
                        continue
                    else:
                        frame_skip_count = 0
                
                # 프레임을 큐에 추가
                try:
                    self.frame_queue.put_nowait((frame, self.total_frames, time.time()))
                except Full:
                    # 큐가 가득 찬 경우 오래된 프레임 제거
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except Empty:
                        pass
                    
                    try:
                        self.frame_queue.put_nowait((frame, self.total_frames, time.time()))
                    except Full:
                        self.dropped_frames += 1
                        continue
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        logger.info("Capture loop ended")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        프레임 가져오기
        
        Args:
            timeout: 대기 시간 (초)
        
        Returns:
            (frame, frame_number, timestamp) 또는 None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_nowait(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """즉시 프레임 가져오기 (대기 없음)"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """최신 프레임 가져오기 (큐의 모든 오래된 프레임 제거)"""
        latest_frame = None
        
        # 큐에서 모든 프레임을 가져와서 최신 것만 유지
        while True:
            try:
                frame_data = self.frame_queue.get_nowait()
                if latest_frame is not None:
                    self.dropped_frames += 1
                latest_frame = frame_data
            except Empty:
                break
        
        return latest_frame
    
    def is_alive(self) -> bool:
        """캡처가 활성 상태인지 확인"""
        return self.is_running and (
            self.capture_thread is None or self.capture_thread.is_alive()
        )
    
    def get_queue_size(self) -> int:
        """현재 큐 크기"""
        return self.frame_queue.qsize()
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        capture_fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'queue_size': self.get_queue_size(),
            'capture_fps': capture_fps,
            'elapsed_time': elapsed_time,
            'drop_rate': self.dropped_frames / max(self.total_frames, 1),
            'video_info': self.video_info.copy()
        }
    
    def __enter__(self):
        """Context manager 시작"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()


def detect_input_type(source: Union[str, int]) -> Dict[str, Any]:
    """
    입력 소스 타입 감지 및 정보 반환
    
    Args:
        source: 입력 소스
    
    Returns:
        소스 정보 딕셔너리
    """
    manager = RealtimeInputManager(source)
    source_info = {
        'type': manager.source_type,
        'source': source,
        'is_realtime': manager.source_type in ['stream', 'webcam']
    }
    
    # 간단한 테스트로 유효성 확인
    if manager._initialize_capture():
        manager._collect_video_info()
        source_info.update(manager.video_info)
        manager.cap.release()
        source_info['valid'] = True
    else:
        source_info['valid'] = False
    
    return source_info