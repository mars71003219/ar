#!/usr/bin/env python3
"""
Realtime Window Manager - 실시간 윈도우 관리자

기존 WindowProcessor의 기능을 실시간 스트림에 맞게 확장하여
100프레임 단위의 슬라이딩 윈도우를 실시간으로 관리합니다.
기존 모듈과의 완벽한 호호성을 위해 동일한 데이터 형식을 사용합니다.
"""

import time
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from threading import Lock

from .window_processor import WindowProcessor


@dataclass
class FrameData:
    """프레임 데이터 구조체"""
    frame_idx: int
    timestamp: float
    pose_results: List[Dict[str, Any]]
    track_results: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[int, float]] = None  # track_id -> score


@dataclass
class WindowData:
    """윈도우 데이터 구조체 (기존 WindowProcessor 호환성)"""
    window_idx: int
    start_frame: int
    end_frame: int
    frames: List[FrameData]
    annotation: Dict[str, Any]  # STGCN++ 형식
    is_complete: bool = False
    inference_result: Optional[str] = None
    confidence: Optional[float] = None


class RealtimeWindowManager:
    """
    실시간 슬라이딩 윈도우 관리자
    
    기존 WindowProcessor의 로직을 실시간에 맞게 수정하여
    메모리 효율적인 슬라이딩 윈도우 처리를 제공합니다.
    """
    
    def __init__(self, 
                 clip_len: int = 100,
                 inference_stride: int = 50,
                 max_persons: int = 4,
                 config: Optional[Dict[str, Any]] = None):
        """
        초기화
        
        Args:
            clip_len: 윈도우 크기 (프레임 수)
            inference_stride: 추론 간격 (프래임 수)
            max_persons: 최대 인원 수
            config: 추가 설정
        """
        self.clip_len = clip_len
        self.inference_stride = inference_stride
        self.max_persons = max_persons
        self.config = config or {}
        
        # 내부 상태
        self.frame_buffer = deque(maxlen=clip_len * 2)  # 여유 공간 포함
        self.windows_queue = deque(maxlen=10)  # 완료된 윈도우 큐
        self.current_frame_count = 0
        self.window_count = 0
        
        # 스레드 안전성
        self.lock = Lock()
        
        # 성능 모니터링
        self.last_inference_frame = 0
        self.processing_times = deque(maxlen=100)
        
        # 기존 WindowProcessor 연도 (호환성)
        self.window_processor = None
        if self.config.get('enable_legacy_processor', False):
            self.window_processor = WindowProcessor(
                clip_len=clip_len,
                inference_stride=inference_stride,
                config=config
            )
    
    def add_frame_data(self, frame_idx: int, pose_results: List[Dict[str, Any]], 
                      track_results: Optional[Dict[str, Any]] = None,
                      scores: Optional[Dict[int, float]] = None) -> bool:
        """
        프레임 데이터 추가
        
        Args:
            frame_idx: 프레임 인덱스
            pose_results: 포즈 추출 결과
            track_results: 트랙킹 결과
            scores: 인물별 점수
            
        Returns:
            새로운 윈도우가 준비되었는지 여부
        """
        with self.lock:
            # 프레임 데이터 생성
            frame_data = FrameData(
                frame_idx=frame_idx,
                timestamp=time.time(),
                pose_results=pose_results,
                track_results=track_results,
                scores=scores
            )
            
            # 버퍼에 추가
            self.frame_buffer.append(frame_data)
            self.current_frame_count = frame_idx + 1
            
            # 윈도우 생성 조건 확인
            return self._check_window_ready()
    
    def _check_window_ready(self) -> bool:
        """윈도우 준비 상태 확인"""
        # 최소 프레임 수 확인
        if len(self.frame_buffer) < self.clip_len:
            return False
        
        # 추론 간격 확인
        frames_since_last_inference = self.current_frame_count - self.last_inference_frame
        return frames_since_last_inference >= self.inference_stride
    
    def create_window(self) -> Optional[WindowData]:
        """
        새로운 윈도우 생성
        
        Returns:
            WindowData 또는 None (윈도우가 준비되지 않은 경우)
        """
        with self.lock:
            if not self._check_window_ready():
                return None
            
            start_time = time.time()
            
            # 최신 clip_len 개의 프레임 추출
            window_frames = list(self.frame_buffer)[-self.clip_len:]
            
            if len(window_frames) < self.clip_len:
                return None
            
            # 윈도우 메타데이터
            start_frame = window_frames[0].frame_idx
            end_frame = window_frames[-1].frame_idx
            
            # STGCN++ 호환 어노테이션 생성
            annotation = self._create_stgcn_annotation(window_frames)
            
            # 윈도우 데이터 생성
            window_data = WindowData(
                window_idx=self.window_count,
                start_frame=start_frame,
                end_frame=end_frame,
                frames=window_frames.copy(),  # 딥 커피로 데이터 보존
                annotation=annotation
            )
            
            # 상태 업데이트
            self.window_count += 1
            self.last_inference_frame = self.current_frame_count
            
            # 성능 모니터링
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return window_data
    
    def _create_stgcn_annotation(self, frames: List[FrameData]) -> Dict[str, Any]:
        """
        STGCN++ 모델용 어노테이션 생성
        기존 AnnotationGenerator의 로직을 실시간에 맞게 수정
        """
        annotation = {
            'total_frames': len(frames),
            'persons': {}
        }
        
        # 모든 프레임에서 등장한 track_id 수집
        all_track_ids = set()
        for frame in frames:
            if frame.track_results and 'track_ids' in frame.track_results:
                all_track_ids.update(frame.track_results['track_ids'])
        
        # 점수 기반 track_id 정렬 (상위 max_persons명만 선택)
        selected_track_ids = self._select_top_track_ids(frames, all_track_ids)
        
        # 각 track_id별로 키포인트 시퀀스 생성
        for i, track_id in enumerate(selected_track_ids):
            keypoint_sequence = self._extract_keypoint_sequence(frames, track_id)
            
            if keypoint_sequence is not None:
                annotation['persons'][f'person_{i}'] = {
                    'track_id': track_id,
                    'keypoint': keypoint_sequence,  # (1, T, 17, 2) 형태
                    'score': self._calculate_average_score(frames, track_id)
                }
        
        return annotation
    
    def _select_top_track_ids(self, frames: List[FrameData], 
                             all_track_ids: set) -> List[int]:
        """점수 기반 상위 track_id 선택"""
        if not all_track_ids:
            return []
        
        # 각 track_id의 평균 점수 계산
        track_scores = {}
        for track_id in all_track_ids:
            scores = []
            for frame in frames:
                if frame.scores and track_id in frame.scores:
                    scores.append(frame.scores[track_id])
            
            if scores:
                track_scores[track_id] = np.mean(scores)
            else:
                track_scores[track_id] = 0.0
        
        # 점수순으로 정렬 후 상위 선택
        sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        return [track_id for track_id, score in sorted_tracks[:self.max_persons]]
    
    def _extract_keypoint_sequence(self, frames: List[FrameData], 
                                  track_id: int) -> Optional[np.ndarray]:
        """
        특정 track_id의 키포인트 시퀀스 추출
        
        Returns:
            (1, T, 17, 2) 형태의 numpy 배열 또는 None
        """
        keypoints_list = []
        
        for frame in frames:
            keypoints = None
            
            # pose_results에서 track_id에 해당하는 키포인트 찾기
            if frame.track_results and 'track_ids' in frame.track_results:
                track_ids = frame.track_results['track_ids']
                if track_id in track_ids:
                    track_idx = track_ids.index(track_id)
                    if track_idx < len(frame.pose_results):
                        pose_data = frame.pose_results[track_idx]
                        keypoints = self._extract_keypoints_from_pose(pose_data)
            
            # 키포인트가 없으면 0으로 채우기
            if keypoints is None:
                keypoints = np.zeros((17, 2), dtype=np.float32)
            
            keypoints_list.append(keypoints)
        
        if not keypoints_list:
            return None
        
        # (T, 17, 2) -> (1, T, 17, 2) 형태로 변환
        keypoint_array = np.array(keypoints_list, dtype=np.float32)  # (T, 17, 2)
        return keypoint_array[np.newaxis, ...]  # (1, T, 17, 2)
    
    def _extract_keypoints_from_pose(self, pose_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """포즈 데이터에서 키포인트 추출"""
        try:
            if 'keypoints' in pose_data:
                keypoints = pose_data['keypoints']
                if isinstance(keypoints, (list, np.ndarray)):
                    keypoints = np.array(keypoints, dtype=np.float32)
                    if keypoints.shape == (17, 3):
                        return keypoints[:, :2]  # x, y 좌표만 사용
                    elif keypoints.shape == (17, 2):
                        return keypoints
                    elif keypoints.shape == (51,):
                        # 평면화된 경우 reshape
                        return keypoints.reshape(17, 3)[:, :2]
            
            return None
            
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return None
    
    def _calculate_average_score(self, frames: List[FrameData], track_id: int) -> float:
        """특정 track_id의 평균 점수 계산"""
        scores = []
        for frame in frames:
            if frame.scores and track_id in frame.scores:
                scores.append(frame.scores[track_id])
        
        return np.mean(scores) if scores else 0.0
    
    def add_completed_window(self, window_data: WindowData):
        """완료된 윈도우를 플에 추가"""
        with self.lock:
            window_data.is_complete = True
            self.windows_queue.append(window_data)
    
    def get_completed_windows(self) -> List[WindowData]:
        """완료된 윈도우 목록 반환 및 제거"""
        with self.lock:
            windows = list(self.windows_queue)
            self.windows_queue.clear()
            return windows
    
    def get_statistics(self) -> Dict[str, Any]:
        """현재 상태 통계 반환"""
        with self.lock:
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            return {
                'current_frame_count': self.current_frame_count,
                'window_count': self.window_count,
                'buffer_size': len(self.frame_buffer),
                'completed_windows': len(self.windows_queue),
                'last_inference_frame': self.last_inference_frame,
                'avg_processing_time': avg_processing_time,
                'frames_until_next_inference': max(0, 
                    self.inference_stride - (self.current_frame_count - self.last_inference_frame))
            }
    
    def clear_old_data(self, keep_frames: int = None):
        """오래된 데이터 정리 (메모리 최적화)"""
        with self.lock:
            if keep_frames is None:
                keep_frames = self.clip_len
            
            # 버퍼 크기 조정
            while len(self.frame_buffer) > keep_frames:
                self.frame_buffer.popleft()
    
    def reset(self):
        """상태 초기화"""
        with self.lock:
            self.frame_buffer.clear()
            self.windows_queue.clear()
            self.current_frame_count = 0
            self.window_count = 0
            self.last_inference_frame = 0
            self.processing_times.clear()
            
            print("RealtimeWindowManager reset completed")


class WindowEventCallback:
    """윈도우 이벤트 콜백 인터페이스"""
    
    def on_window_created(self, window_data: WindowData):
        """윈도우 생성 시 호출"""
        pass
    
    def on_window_completed(self, window_data: WindowData):
        """윈도우 처리 완료 시 호출"""
        pass
    
    def on_inference_result(self, window_data: WindowData, result: str, confidence: float):
        """추론 결과 수신 시 호출"""
        pass


class CallbackWindowManager(RealtimeWindowManager):
    """콜백 지원 윈도우 관리자"""
    
    def __init__(self, callback: WindowEventCallback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
    
    def create_window(self) -> Optional[WindowData]:
        """윈도우 생성 및 콜백 호출"""
        window_data = super().create_window()
        if window_data and self.callback:
            self.callback.on_window_created(window_data)
        return window_data
    
    def add_completed_window(self, window_data: WindowData):
        """완료된 윈도우 추가 및 콜백 호출"""
        super().add_completed_window(window_data)
        if self.callback:
            self.callback.on_window_completed(window_data)
    
    def notify_inference_result(self, window_data: WindowData, result: str, confidence: float):
        """추론 결과 알림"""
        window_data.inference_result = result
        window_data.confidence = confidence
        if self.callback:
            self.callback.on_inference_result(window_data, result, confidence)
