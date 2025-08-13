#!/usr/bin/env python3
"""
RTSP Stream Processor - 실시간 RTSP/카메라 스트림 처리기

기존 BaseProcessor를 상속하여 RTSP 스트림과 웹캠을 실시간으로 처리합니다.
기존 모듈의 인터페이스와 완벽히 호환됩니다.
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Optional, Tuple, Iterator, Callable, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from .base_processor import BaseProcessor
from utils.video_utils import get_video_info


class RTSPStreamProcessor(BaseProcessor):
    """
    RTSP 스트림 및 카메라 입력을 실시간으로 처리하는 클래스
    
    기존 비디오 파일 처리와 동일한 인터페이스를 제공하면서
    실시간 스트림 특성에 맞는 버퍼링과 예외 처리를 제공합니다.
    """
    
    def __init__(self, source: str, config: Optional[Dict[str, Any]] = None):
        """
        초기화
        
        Args:
            source: RTSP URL, 웹캠 인덱스(0, 1, 2...), 또는 비디오 파일 경로
                   예: 'rtsp://192.168.1.100:554/stream'
                       0 (웹캠)
                       '/path/to/video.mp4'
            config: 설정 딕셔너리
                   {
                       'buffer_size': 30,        # 프레임 버퍼 크기
                       'reconnect_attempts': 5,  # 재연결 시도 횟수
                       'timeout_seconds': 10,    # 타임아웃
                       'target_fps': 15,         # 타겟 FPS
                       'frame_skip': 2           # 프레임 스킵 (성능 최적화)
                   }
        """
        super().__init__()
        
        self.source = source
        self.config = config or {}
        
        # 설정값 초기화
        self.buffer_size = self.config.get('buffer_size', 30)
        self.reconnect_attempts = self.config.get('reconnect_attempts', 5)
        self.timeout_seconds = self.config.get('timeout_seconds', 10)
        self.target_fps = self.config.get('target_fps', 15)
        self.frame_skip = self.config.get('frame_skip', 2)
        
        # 내부 상태
        self.cap = None
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.capture_thread = None
        self.is_running = False
        self.frame_count = 0
        self.skip_counter = 0
        
        # 성능 모니터링
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # 재연결 관리
        self.connection_failures = 0
        self.last_reconnect_time = 0
        
        # 스레드 안전성
        self.lock = threading.Lock()
        
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop_capture()
    
    def _is_rtsp_source(self) -> bool:
        """RTSP 소스인지 확인"""
        return isinstance(self.source, str) and self.source.startswith('rtsp://')
    
    def _is_camera_source(self) -> bool:
        """카메라 소스인지 확인"""
        return isinstance(self.source, int)
    
    def _create_capture(self) -> cv2.VideoCapture:
        """VideoCapture 객체 생성"""
        if self._is_rtsp_source():
            # RTSP 스트림용 최적화된 설정
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화 (지연 감소)
            cap.set(cv2.CAP_PROP_TIMEOUT, self.timeout_seconds * 1000)
        elif self._is_camera_source():
            # 웹캠용 설정
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)  # 웹캠 FPS 설정
        else:
            # 비디오 파일용 설정
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        return cap
    
    def start_capture(self):
        """캡처 시작"""
        if self.is_running:
            return
        
        try:
            self.cap = self._create_capture()
            self.is_running = True
            
            # 캡처 스레드 시작
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            print(f"Started capture from {self.source}")
            
        except Exception as e:
            print(f"Failed to start capture: {e}")
            self.stop_capture()
            raise
    
    def stop_capture(self):
        """캡처 중지"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 큐 비우기
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        print(f"Stopped capture from {self.source}")
    
    def _capture_frames(self):
        """프레임 캡처 스레드 함수"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame, attempting reconnection...")
                    if self._attempt_reconnect():
                        continue
                    else:
                        print("Max reconnection attempts reached. Stopping capture.")
                        break
                
                # 프레임 스킵 처리 (성능 최적화)
                self.skip_counter += 1
                if self.skip_counter < self.frame_skip:
                    continue
                self.skip_counter = 0
                
                # FPS 제한
                if self.target_fps > 0:
                    time.sleep(1.0 / self.target_fps)
                
                # 프레임을 큐에 추가 (논블로킹)
                try:
                    self.frame_queue.put_nowait((self.frame_count, frame))
                    self.frame_count += 1
                    
                    # FPS 계산
                    self._update_fps_counter()
                    
                    # 연결 실패 카운터 리셋
                    self.connection_failures = 0
                    
                except:
                    # 큐가 가득 찬 경우 가장 오래된 프레임 제거
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((self.frame_count, frame))
                        self.frame_count += 1
                    except:
                        pass
                
            except Exception as e:
                print(f"Error in capture thread: {e}")
                if self._attempt_reconnect():
                    continue
                else:
                    break
    
    def _attempt_reconnect(self) -> bool:
        """재연결 시도"""
        if not self._is_rtsp_source():
            return False  # 파일이나 웹캠은 재연결 불가
        
        self.connection_failures += 1
        current_time = time.time()
        
        # 재연결 시도 제한
        if self.connection_failures > self.reconnect_attempts:
            return False
        
        # 재연결 간격 제한 (최소 5초)
        if current_time - self.last_reconnect_time < 5.0:
            time.sleep(1)
            return True
        
        print(f"Reconnection attempt {self.connection_failures}/{self.reconnect_attempts}")
        
        try:
            if self.cap:
                self.cap.release()
            
            time.sleep(2)  # 재연결 전 대기
            self.cap = self._create_capture()
            self.last_reconnect_time = current_time
            
            print("Reconnection successful")
            return True
            
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False
    
    def _update_fps_counter(self):
        """FPS 카운터 업데이트"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def read_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """프레임 읽기 (논블로킹)"""
        if not self.is_running:
            return None
        
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_iterator(self) -> Iterator[Tuple[int, np.ndarray]]:
        """프레임 반복자 반환 (기존 코드와의 호환성)"""
        while self.is_running:
            frame_data = self.read_frame()
            if frame_data is not None:
                yield frame_data
            else:
                # 타임아웃 발생 시 짧게 대기
                time.sleep(0.1)
    
    def get_stream_info(self) -> Dict[str, Any]:
        """스트림 정보 반환"""
        if not self.cap:
            return {}
        
        return {
            'source': self.source,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'current_fps': self.current_fps,
            'buffer_size': self.buffer_size,
            'frame_count': self.frame_count,
            'is_running': self.is_running,
            'connection_failures': self.connection_failures
        }
    
    def is_healthy(self) -> bool:
        """스트림 상태 확인"""
        return (self.is_running and 
                self.connection_failures < self.reconnect_attempts and
                self.current_fps > 0)
    
    def __iter__(self):
        """반복 가능한 객체로 사용 (기존 코드 호환성)"""
        return self.get_frame_iterator()
    
    def __len__(self):
        """총 프레임 수 (실시간 스트림은 무한대)"""
        if self._is_rtsp_source() or self._is_camera_source():
            return float('inf')  # 실시간 스트림은 무한대
        else:
            # 비디오 파일인 경우
            if self.cap:
                return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return 0


class MultiStreamProcessor:
    """
    다중 RTSP 스트림 동시 처리
    여러 카메라를 동시에 모니터링할 때 사용
    """
    
    def __init__(self, sources: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Args:
            sources: {"camera1": "rtsp://...", "camera2": "rtsp://..."}
            config: 공통 설정
        """
        self.sources = sources
        self.config = config or {}
        self.processors = {}
        self.executor = ThreadPoolExecutor(max_workers=len(sources))
    
    def start_all(self):
        """모든 스트림 시작"""
        for name, source in self.sources.items():
            processor = RTSPStreamProcessor(source, self.config)
            processor.start_capture()
            self.processors[name] = processor
            print(f"Started stream {name}: {source}")
    
    def stop_all(self):
        """모든 스트림 중지"""
        for name, processor in self.processors.items():
            processor.stop_capture()
            print(f"Stopped stream {name}")
        
        self.executor.shutdown(wait=True)
    
    def read_frames(self) -> Dict[str, Optional[Tuple[int, np.ndarray]]]:
        """모든 스트림에서 프레임 읽기"""
        frames = {}
        for name, processor in self.processors.items():
            frames[name] = processor.read_frame(timeout=0.1)
        return frames
    
    def get_healthy_streams(self) -> List[str]:
        """정상 동작하는 스트림 목록 반환"""
        healthy = []
        for name, processor in self.processors.items():
            if processor.is_healthy():
                healthy.append(name)
        return healthy
