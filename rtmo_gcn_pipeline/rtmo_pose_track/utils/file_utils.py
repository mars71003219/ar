#!/usr/bin/env python3
"""
File handling utilities
"""

import os
import glob
import pickle
from typing import List, Dict, Any


def collect_video_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """비디오 파일 수집"""
    if extensions is None:
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    video_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)


def get_processed_videos(output_dir: str, dataset_name: str) -> set:
    """처리된 비디오 키 목록 반환"""
    processed_video_keys = set()
    
    # temp 폴더에서 완료된 비디오들 확인
    temp_path = os.path.join(output_dir, dataset_name, 'temp')
    if os.path.exists(temp_path):
        for root, dirs, files in os.walk(temp_path):
            for dir_name in dirs:
                if dir_name not in ['Fight', 'NonFight', 'train', 'val', 'test']:
                    video_dir_path = os.path.join(root, dir_name)
                    pkl_file = os.path.join(video_dir_path, f"{dir_name}_windows.pkl")
                    
                    if os.path.exists(pkl_file):
                        root_parts = root.split(os.sep)
                        if 'Fight' in root_parts:
                            processed_video_keys.add(f"Fight/{dir_name}")
                        elif 'NonFight' in root_parts:
                            processed_video_keys.add(f"NonFight/{dir_name}")
    
    # 최종 폴더에서도 확인
    for split_dir in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, dataset_name, split_dir)
        if os.path.exists(split_path):
            for category in ['Fight', 'NonFight']:
                category_path = os.path.join(split_path, category)
                if os.path.exists(category_path):
                    for video_dir in os.listdir(category_path):
                        pkl_file = os.path.join(category_path, video_dir, f"{video_dir}_windows.pkl")
                        if os.path.exists(pkl_file):
                            processed_video_keys.add(f"{category}/{video_dir}")
    
    return processed_video_keys


def create_output_directories(base_path: str, subdirs: List[str] = None):
    """출력 디렉토리 생성"""
    if subdirs is None:
        subdirs = ['temp', 'train', 'val', 'test']
    
    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        os.makedirs(path, exist_ok=True)


def save_pkl_data(data: Any, file_path: str):
    """PKL 데이터 저장"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl_data(file_path: str) -> Any:
    """PKL 데이터 로드"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)