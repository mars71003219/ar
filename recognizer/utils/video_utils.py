"""
비디오 처리 유틸리티
비디오 관련 공통 기능들을 통합 관리
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Tuple[int, float, float]:
    """
    비디오 기본 정보 가져오기
    
    Args:
        video_path: 비디오 파일 경로
    
    Returns:
        Tuple[int, float, float]: (frame_count, fps, duration)
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return 0, 0.0, 0.0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        duration = frame_count / fps if fps > 0 else 0.0
        
        return frame_count, fps, duration
        
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {e}")
        return 0, 0.0, 0.0


def get_video_duration(video_path: str) -> float:
    """
    비디오 길이 가져오기
    
    Args:
        video_path: 비디오 파일 경로
    
    Returns:
        float: 비디오 길이 (초)
    """
    _, _, duration = get_video_info(video_path)
    return duration


def get_video_fps(video_path: str) -> float:
    """
    비디오 FPS 가져오기
    
    Args:
        video_path: 비디오 파일 경로
    
    Returns:
        float: 비디오 FPS
    """
    _, fps, _ = get_video_info(video_path)
    return fps


def get_video_frame_count(video_path: str) -> int:
    """
    비디오 프레임 수 가져오기
    
    Args:
        video_path: 비디오 파일 경로
    
    Returns:
        int: 프레임 수
    """
    frame_count, _, _ = get_video_info(video_path)
    return frame_count


def is_valid_video_file(file_path: str, extensions: list = None) -> bool:
    """
    유효한 비디오 파일인지 확인
    
    Args:
        file_path: 확인할 파일 경로
        extensions: 허용되는 확장자 리스트
    
    Returns:
        bool: 유효한 비디오 파일 여부
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    path = Path(file_path)
    
    # 파일 존재 및 확장자 확인
    if not path.exists() or path.suffix.lower() not in extensions:
        return False
    
    # 실제 비디오 파일인지 확인
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except Exception:
        return False


def find_video_files(directory: str, recursive: bool = True, extensions: list = None) -> list:
    """
    디렉토리에서 비디오 파일들 찾기
    
    Args:
        directory: 검색할 디렉토리 경로
        recursive: 하위 디렉토리까지 검색할지 여부
        extensions: 허용되는 확장자 리스트
    
    Returns:
        list: 발견된 비디오 파일 경로 리스트
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    video_files = []
    
    if recursive:
        # 재귀 검색
        for ext in extensions:
            video_files.extend(directory.rglob(f"*{ext}"))
    else:
        # 현재 디렉토리만 검색
        for ext in extensions:
            video_files.extend(directory.glob(f"*{ext}"))
    
    # 실제 비디오 파일만 필터링
    valid_videos = []
    for video_file in video_files:
        if is_valid_video_file(str(video_file), extensions):
            valid_videos.append(video_file)
    
    return valid_videos