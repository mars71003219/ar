#!/usr/bin/env python3
"""
Video handling utilities
"""

import os
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any


def get_video_info(video_path: str) -> Dict[str, Any]:
    """비디오 정보 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def validate_video(video_path: str) -> bool:
    """비디오 파일 유효성 검사"""
    if not os.path.exists(video_path):
        return False
    
    cap = cv2.VideoCapture(video_path)
    valid = cap.isOpened()
    cap.release()
    return valid


def extract_video_name(video_path: str) -> str:
    """비디오 파일명에서 이름 추출"""
    return Path(video_path).stem


def create_segment_video(input_video_path: str, output_path: str, 
                        start_frame: int, end_frame: int) -> bool:
    """비디오 세그먼트 생성"""
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        return True
        
    except Exception as e:
        print(f"Error creating segment video: {e}")
        return False