#!/usr/bin/env python3
"""
Enhanced Visualizer with Inference and Separated Pipeline Overlay
원본 비디오와 추론/파이프라인 결과 PKL을 매칭하여 관절 오버레이와 분류 결과를 표시하는 시각화 시스템

두 가지 오버레이 모드 지원:
1. inference_overlay: inference 결과 pkl 파일 시각화
2. separated_overlay: separated pipeline step2 결과 pkl 파일 시각화
"""

import os
import sys
import json
import cv2
import pickle
import numpy as np
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple




class EnhancedVisualizer:
    """향상된 시각화 클래스 - Inference 및 Separated Pipeline 지원"""
    
    def __init__(self, mode='inference', video_dir=None, pkl_dir=None, 
                 save=False, save_dir=None, num_person=2, config=None):
        if config is None:
            raise ValueError("A config object must be provided.")

        self.config = config
        self.mode = mode
        self.video_dir = video_dir or self.config.default_input_dir
        self.pkl_dir = pkl_dir or self.config.default_output_dir
        self.save = save
        self.save_dir = save_dir or self.config.default_output_dir
        self.output_dir = self.save_dir
        self.num_persons = num_person

        # UI 폰트 설정
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 비디오-PKL 매칭 정보
        self.video_pkl_pairs = []
        
    def find_video_pkl_pairs(self) -> List[Tuple[str, str, str]]:
        """디렉토리에서 비디오-PKL 쌍 찾기"""
        pairs = []
        
        if self.mode == 'inference_overlay':
            # inference 모드: windows 폴더에서 _windows.pkl 파일 찾기
            pkl_files = []
            windows_dir = os.path.join(self.pkl_dir, "windows")
            if os.path.exists(windows_dir):
                for root, dirs, files in os.walk(windows_dir):
                    for file in files:
                        if file.endswith('_windows.pkl'):
                            pkl_path = os.path.join(root, file)
                            pkl_files.append(pkl_path)
        
        elif self.mode == 'separated_overlay':
            # separated 모드: step2 폴더에서 _windows.pkl 파일 찾기
            pkl_files = []
            for root, dirs, files in os.walk(self.pkl_dir):
                for file in files:
                    # _windows.pkl 파일이나 일반 .pkl 파일 모두 허용
                    if file.endswith('_windows.pkl') or (file.endswith('.pkl') and 'window' in file):
                        pkl_path = os.path.join(root, file)
                        pkl_files.append(pkl_path)
        
        else:
            print(f"Unknown mode: {self.mode}")
            return pairs
        
        print(f"Found {len(pkl_files)} PKL files")
        
        # 각 PKL 파일에 대해 매칭되는 비디오 찾기
        for pkl_path in pkl_files:
            video_name = Path(pkl_path).stem.replace('_windows', '')
            
            # video_dir에서 매칭되는 비디오 찾기
            video_extensions = ['mp4', 'avi', 'mov', 'mkv']
            for ext in video_extensions:
                if os.path.isfile(self.video_dir):
                    # 단일 파일인 경우
                    if Path(self.video_dir).stem == video_name:
                        video_path = self.video_dir
                        label_folder = "single_file"
                        pairs.append((video_path, pkl_path, label_folder))
                        break
                else:
                    # 디렉토리인 경우
                    pattern_path = os.path.join(self.video_dir, '**', f"{video_name}.{ext}")
                    matches = glob.glob(pattern_path, recursive=True)
                    if matches:
                        video_path = matches[0]
                        label_folder = Path(video_path).parent.name
                        pairs.append((video_path, pkl_path, label_folder))
                        print(f"Matched: {video_name} -> {label_folder}")
                        break
        
        return pairs
    
    def load_pkl_data(self, pkl_path: str) -> Dict:
        """PKL 파일에서 데이터 로드"""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading PKL file {pkl_path}: {e}")
            return {}
    
    def get_inference_results_from_pkl(self, video_name: str) -> Dict:
        """PKL 파일에서 추론 결과 로드"""
        # windows/{label}/{video_name}_windows.pkl 형태에서 inference 결과 찾기
        inference_dir = os.path.join(self.pkl_dir, "windows")
        
        for label_folder in ['Fight', 'NonFight']:
            inference_pkl = os.path.join(inference_dir, label_folder, f"{video_name}_windows.pkl")
            if os.path.exists(inference_pkl):
                try:
                    with open(inference_pkl, 'rb') as f:
                        inference_data = pickle.load(f)
                    return inference_data
                except Exception as e:
                    print(f"Error loading inference PKL {inference_pkl}: {e}")
        
        return {}
    
    def get_final_video_result(self, video_name: str) -> Dict[str, Any]:
        """video_results.json에서 최종 비디오 분류 결과 로드"""
        try:
            results_file = os.path.join(self.pkl_dir, "results", "video_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    video_results = json.load(f)
                
                # 해당 비디오 찾기
                for result in video_results:
                    if result.get('video_name') == video_name:
                        return {
                            'video_prediction': result.get('video_prediction', 0),
                            'avg_prediction_score': result.get('avg_prediction_score', 0.0),
                            'max_prediction_score': result.get('max_prediction_score', 0.0),
                            'fight_windows': result.get('fight_windows', 0),
                            'window_count': result.get('window_count', 0)
                        }
        except Exception as e:
            print(f"Error loading video results: {e}")
        
        return {'video_prediction': 0, 'avg_prediction_score': 0.0}
    
    def draw_skeleton(self, img: np.ndarray, keypoints: np.ndarray, person_idx: int, 
                     is_overlap: bool = False) -> np.ndarray:
        """스켈레톤 그리기"""
        if keypoints is None or len(keypoints) == 0:
            return img
            
        # 색상 선택
        if is_overlap:
            color = self.config.overlap_color
        else:
            color = self.config.colors[person_idx % len(self.config.colors)]
        
        # keypoints 형태 확인 및 처리
        try:
            # keypoints가 (17, 3) 형태인지 확인
            if keypoints.shape[-1] == 3:
                # (x, y, score) 형태
                coords_scores = keypoints
            elif keypoints.shape[-1] == 2:
                # (x, y) 형태인 경우 기본 score 1.0 추가
                # keypoints 차원에 맞게 ones 배열 생성
                if len(keypoints.shape) == 2:  # (17, 2)
                    ones_shape = (keypoints.shape[0], 1)
                else:  # (T, 17, 2) 등의 경우
                    ones_shape = keypoints.shape[:-1] + (1,)
                coords_scores = np.concatenate([keypoints, np.ones(ones_shape)], axis=-1)
            else:
                print(f"Unexpected keypoints shape: {keypoints.shape}")
                return img
                
            # 관절점 그리기
            for i, (x, y, score) in enumerate(coords_scores):
                if score > self.config.confidence_threshold:
                    cv2.circle(img, (int(x), int(y)), self.config.keypoint_radius, color, -1)
            
            # 스켈레톤 연결선 그리기  
            for connection in self.config.skeleton_connections:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-based to 0-based
                if pt1_idx < len(coords_scores) and pt2_idx < len(coords_scores):
                    x1, y1, score1 = coords_scores[pt1_idx]
                    x2, y2, score2 = coords_scores[pt2_idx]
                    
                    if score1 > self.config.confidence_threshold and score2 > self.config.confidence_threshold:
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, self.config.line_thickness)
        
        except Exception as e:
            print(f"Error drawing skeleton: {e}, keypoints shape: {keypoints.shape}")
            
        return img
    
    def detect_overlap_persons(self, window_data: Dict, frame_idx: int) -> List[int]:
        """겹침 구간의 사람 객체 감지"""
        overlap_persons = []
        
        # 윈도우 경계 확인 (stride로 인한 겹침)
        if 'overlap_info' in window_data:
            overlap_info = window_data['overlap_info']
            if frame_idx in overlap_info.get('overlap_frames', []):
                overlap_persons = overlap_info.get('overlap_persons', [])
        
        return overlap_persons
    
    def get_head_position(self, keypoints: np.ndarray) -> Optional[Tuple[int, int]]:
        """키포인트에서 머리 위치 계산"""
        try:
            # COCO-17 키포인트 인덱스 (0-based)
            nose_idx = 0
            left_eye_idx = 1
            right_eye_idx = 2
            left_ear_idx = 3
            right_ear_idx = 4
            
            # 키포인트가 (V, C) 형태인지 확인
            if len(keypoints.shape) != 2 or keypoints.shape[0] < 5:
                return None
            
            # 머리 부위 키포인트들의 유효성 확인 (confidence가 있는 경우)
            head_points = []
            confidence_threshold = self.config.confidence_threshold
            
            for idx in [nose_idx, left_eye_idx, right_eye_idx, left_ear_idx, right_ear_idx]:
                if keypoints.shape[1] >= 3:  # (x, y, confidence)
                    x, y, conf = keypoints[idx]
                    if conf > confidence_threshold:
                        head_points.append((x, y))
                elif keypoints.shape[1] >= 2:  # (x, y)
                    x, y = keypoints[idx]
                    if x > 0 and y > 0:  # 유효한 좌표인지 확인
                        head_points.append((x, y))
            
            if not head_points:
                return None
            
            # 머리 중심점 계산
            avg_x = sum(point[0] for point in head_points) / len(head_points)
            avg_y = sum(point[1] for point in head_points) / len(head_points)
            
            # 머리 위쪽으로 오프셋 (30픽셀 위)
            head_top_y = avg_y - 30
            
            return (int(avg_x), int(head_top_y))
            
        except Exception as e:
            print(f"Error calculating head position: {e}")
            return None
    
    def draw_person_info_on_head(self, img: np.ndarray, head_pos: Tuple[int, int], 
                                person_idx: int, track_id: str, composite_score: float) -> np.ndarray:
        """사람 머리 위에 정보 표시"""
        try:
            x, y = head_pos
            
            # 텍스트 내용
            text = f"#{person_idx+1} T:{track_id} {composite_score:.3f}"
            
            # 텍스트 크기 측정
            text_size, baseline = cv2.getTextSize(text, self.font, self.config.font_scale, self.config.font_thickness)
            text_width, text_height = text_size
            
            # 텍스트가 화면 밖으로 나가지 않도록 조정
            img_height, img_width = img.shape[:2]
            
            # X 좌표 조정 (텍스트 중앙 정렬)
            text_x = max(5, min(x - text_width // 2, img_width - text_width - 5))
            
            # Y 좌표 조정 (머리 위에 표시하되 화면 위쪽 경계 고려)
            text_y = max(text_height + 10, y - 15)  # 머리 위 15px 위치에, 최소 화면 상단에서 text_height + 10px
            
            # 배경 사각형 계산
            padding = 3
            rect_x1 = text_x - padding
            rect_y1 = text_y - text_height - padding
            rect_x2 = text_x + text_width + padding
            rect_y2 = text_y + baseline + padding
            
            # 배경 사각형 그리기 (완전 검은색)
            cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            
            # 텍스트 그리기 (흰색으로 통일)
            text_color = (255, 255, 255)  # 흰색
            cv2.putText(img, text, (text_x, text_y), 
                       self.font, self.config.font_scale, text_color, self.config.font_thickness)
            
            return img
            
        except Exception as e:
            print(f"Error drawing person info on head: {e}")
            return img
    
    def draw_track_info(self, img: np.ndarray, persons_data: Dict, num_persons: int) -> np.ndarray:
        """trackId와 내림차순 idx 번호 표시 (separated_overlay용)"""
        h, w = img.shape[:2]
        
        # persons_data 구조 검증
        if not isinstance(persons_data, dict):
            print(f"Warning: persons_data is not a dict, type: {type(persons_data)}")
            return img
            
        try:
            # persons_data를 rank 기준으로 정렬
            persons_list = sorted(persons_data.items(), 
                                 key=lambda x: x[1].get('composite_score', 0.0) if isinstance(x[1], dict) else 0.0, reverse=True)
        except Exception as e:
            print(f"Error sorting persons_data: {e}")
            print(f"persons_data structure: {persons_data}")
            return img
        
        # 상위 N명만 표시
        displayed_persons = persons_list[:num_persons]
        
        # 정보 표시 영역 설정 (왼쪽 상단)
        start_x = self.config.box_padding
        start_y = self.config.window_info_y_start
        
        for idx, (person_id, person_info) in enumerate(displayed_persons):
            # person_info 구조 검증
            if not isinstance(person_info, dict):
                print(f"Warning: person_info is not a dict for person {person_id}, type: {type(person_info)}")
                continue
                
            track_id = person_info.get('track_id', person_id)
            composite_score = person_info.get('composite_score', 0.0)
            
            # 텍스트 내용
            text = f"#{idx+1} Track:{track_id} Score:{composite_score:.3f}"
            
            # 텍스트 크기 측정
            text_size, baseline = cv2.getTextSize(text, self.font, self.config.font_scale, self.config.font_thickness)
            text_width, text_height = text_size
            
            # 배경 사각형 계산 (padding 포함)
            padding = 5
            rect_x1 = start_x - padding
            rect_y1 = start_y - text_height - padding
            rect_x2 = start_x + text_width + padding
            rect_y2 = start_y + baseline + padding
            
            # 배경 사각형 그리기
            cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)  # 검은색 배경
            
            # 텍스트 그리기
            cv2.putText(img, text, (start_x, start_y), 
                       self.font, self.config.font_scale, self.config.text_color, self.config.font_thickness)
            
            # 다음 텍스트 위치 계산 (텍스트 높이 + padding 기반)
            start_y += text_height + baseline + padding * 2 + 5  # 추가 간격 5px
        
        return img
    
    def draw_window_results(self, img: np.ndarray, window_results: List[Dict], 
                           current_frame: int, total_frames: int) -> np.ndarray:
        """윈도우별 분류 결과 표시 (이전, 현재, 이후 윈도우 포함)"""
        h, w = img.shape[:2]
        
        # 현재/이전/다음 윈도우 찾기
        current_windows = []
        prev_windows = []
        next_windows = []
        
        for window in window_results:
            start_frame = window.get('start_frame', 0)
            end_frame = window.get('end_frame', 0)
            
            if start_frame <= current_frame < end_frame:
                current_windows.append(window)
            elif end_frame <= current_frame:
                prev_windows.append(window)
            elif start_frame > current_frame:
                next_windows.append(window)
        
        # 왼쪽 상단에 윈도우 정보 표시
        y_offset = self.config.window_info_y_start
        
        # 텍스트 높이 계산 (기본 텍스트로 높이 측정)
        sample_text = "Sample Window 0: Fight (0.000)"
        text_height = cv2.getTextSize(sample_text, self.font, self.config.font_scale, self.config.font_thickness)[0][1]
        line_height = text_height + 35  # 텍스트 높이 + 충분한 여백 (배경 박스 + 간격)
        
        # 이전 윈도우 (최근 1개만)
        if prev_windows:
            recent_prev = prev_windows[-1]  # 가장 최근 이전 윈도우
            window_idx = recent_prev.get('window_idx', 'prev')
            prediction = recent_prev.get('prediction', 0.0)
            predicted_label = recent_prev.get('predicted_label', 0)
            
            text = f"Prev Window {window_idx}: {'Fight' if predicted_label == 1 else 'NonFight'} ({prediction:.3f})"
            text_size = cv2.getTextSize(text, self.font, self.config.font_scale, self.config.font_thickness)[0]
            
            # 회색 배경 (이전 윈도우) - 정확한 박스 높이 계산
            box_padding = 5
            cv2.rectangle(img, (20-box_padding, y_offset-text_height-box_padding), 
                         (20 + text_size[0] + box_padding, y_offset + box_padding), (100, 100, 100), -1)
            cv2.putText(img, text, (20, y_offset), self.font, self.config.font_scale, self.config.text_color, self.config.font_thickness)
            y_offset += line_height
        
        # 현재 윈도우
        for i, window in enumerate(current_windows):
            window_idx = window.get('window_idx', i)
            prediction = window.get('prediction', 0.0)
            predicted_label = window.get('predicted_label', 0)
            
            # 배경 색상 (Fight: 빨간색, NonFight: 파란색)
            bg_color = self.config.fight_color if predicted_label == 1 else self.config.nonfight_color
            
            text = f"Current Window {window_idx}: {'Fight' if predicted_label == 1 else 'NonFight'} ({prediction:.3f})"
            text_size = cv2.getTextSize(text, self.font, self.config.font_scale, self.config.font_thickness)[0]
            
            # 정확한 박스 높이 계산
            box_padding = 5
            cv2.rectangle(img, (20-box_padding, y_offset-text_height-box_padding), 
                         (20 + text_size[0] + box_padding, y_offset + box_padding), bg_color, -1)
            cv2.putText(img, text, (20, y_offset), self.font, self.config.font_scale, self.config.text_color, self.config.font_thickness)
            y_offset += line_height
        
        # 다음 윈도우 (가장 가까운 1개만)
        if next_windows:
            upcoming_next = next_windows[0]  # 가장 가까운 다음 윈도우
            window_idx = upcoming_next.get('window_idx', 'next')
            prediction = upcoming_next.get('prediction', 0.0)
            predicted_label = upcoming_next.get('predicted_label', 0)
            
            text = f"Next Window {window_idx}: {'Fight' if predicted_label == 1 else 'NonFight'} ({prediction:.3f})"
            text_size = cv2.getTextSize(text, self.font, self.config.font_scale, self.config.font_thickness)[0]
            
            # 연한 회색 배경 (다음 윈도우) - 정확한 박스 높이 계산
            box_padding = 5
            cv2.rectangle(img, (20-box_padding, y_offset-text_height-box_padding), 
                         (20 + text_size[0] + box_padding, y_offset + box_padding), (150, 150, 150), -1)
            cv2.putText(img, text, (20, y_offset), self.font, self.config.font_scale, (0, 0, 0), self.config.font_thickness)
        
        return img
    
    def draw_final_result(self, img: np.ndarray, final_prediction: int, confidence: float) -> np.ndarray:
        """최종 판정 결과 표시"""
        h, w = img.shape[:2]
        
        # 최종 결과 텍스트
        final_text = f"Final: {'FIGHT DETECTED' if final_prediction == 1 else 'NO FIGHT'}"
        confidence_text = f"Confidence: {confidence:.3f}"
        
        # 색상 설정
        bg_color = self.config.fight_color if final_prediction == 1 else self.config.nonfight_color
        text_color = self.config.text_color
        
        # 텍스트 크기 계산
        (final_text_width, final_text_height), _ = cv2.getTextSize(
            final_text, self.font, self.config.title_font_scale, self.config.title_font_thickness)
        (conf_text_width, conf_text_height), _ = cv2.getTextSize(
            confidence_text, self.font, self.config.font_scale, self.config.font_thickness)
        
        # 배경 사각형 너비
        rect_w = max(final_text_width, conf_text_width) + self.config.box_padding * 2
        
        # 동적 높이 및 Y 좌표 계산
        top_margin = self.config.box_padding
        line_spacing = self.config.box_padding
        bottom_margin = self.config.box_padding
        
        rect_y = self.config.final_result_margin
        
        # 첫 줄 Y 좌표 (텍스트의 baseline 기준)
        final_text_y = rect_y + top_margin + final_text_height
        # 두 번째 줄 Y 좌표
        conf_text_y = final_text_y + line_spacing + conf_text_height
        
        # 전체 높이
        rect_h = top_margin + final_text_height + line_spacing + conf_text_height + bottom_margin
        
        # 배경 사각형 위치 (우상단)
        rect_x = w - rect_w - self.config.final_result_margin
        
        # 배경 사각형 그리기
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), bg_color, -1)
        
        # 테두리 그리기
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), 2)
        
        # 텍스트 그리기
        cv2.putText(img, final_text, (rect_x + 10, final_text_y), self.font, 
                   self.config.title_font_scale, text_color, self.config.title_font_thickness)
        cv2.putText(img, confidence_text, (rect_x + 10, conf_text_y), self.font, 
                   self.config.font_scale, text_color, self.config.font_thickness)
        
        return img
    
    def visualize_inference_overlay(self, video_path: str, pkl_path: str, 
                                   output_video_path: str = None) -> bool:
        """Inference 결과 오버레이 시각화"""
        return self.visualize_video_with_results(video_path, pkl_path, output_video_path)
    
    def visualize_separated_overlay(self, video_path: str, pkl_path: str, 
                                   output_video_path: str = None) -> bool:
        """Separated pipeline 결과 오버레이 시각화"""
        try:
            # PKL 데이터 로드
            pkl_data = self.load_pkl_data(pkl_path)
            if not pkl_data:
                print(f"Failed to load PKL data from {pkl_path}")
                return False
            
            
            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video {video_path}")
                return False
            
            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video: {video_path}")
            print(f"PKL: {pkl_path}")
            print(f"Frames: {total_frames}, FPS: {fps}")
            
            # 출력 비디오 설정 (옵션)
            out = None
            if output_video_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 현재 프레임의 포즈 데이터 가져오기
                frame_data, relative_frame_idx = self._get_frame_data_separated(pkl_data, frame_idx)
                
                # 관절 오버레이 그리기
                if frame_data and 'annotation' in frame_data:
                    annotation = frame_data['annotation']
                    if 'persons' in annotation:
                        persons_data = annotation['persons']
                        
                        # trackId와 idx 정보는 각 객체 머리 위에 개별 표시 (아래에서 처리)
                        
                        # 상위 N명의 키포인트 그리기
                        persons_list = sorted(persons_data.items(), 
                                            key=lambda x: x[1].get('composite_score', 0.0) if isinstance(x[1], dict) else 0.0, reverse=True)
                        
                        for person_idx, (person_id, person_info) in enumerate(persons_list[:self.num_persons]):
                            try:
                                # person_info 타입 검증
                                if not isinstance(person_info, dict):
                                    print(f"Skipping person {person_id}: person_info is not dict, type: {type(person_info)}")
                                    continue
                                    
                                keypoints = person_info.get('keypoint')
                                if keypoints is not None:
                                    # keypoints 형태 처리 (separated는 보통 (1, T, V, C) 형태)
                                    if len(keypoints.shape) == 4:  # (1, T, V, C)
                                        keypoints_3d = keypoints.squeeze(0)  # (T, V, C)
                                        if relative_frame_idx < keypoints_3d.shape[0]:
                                            current_keypoints = keypoints_3d[relative_frame_idx]
                                        else:
                                            continue
                                    elif len(keypoints.shape) == 3:  # (T, V, C)
                                        if relative_frame_idx < keypoints.shape[0]:
                                            current_keypoints = keypoints[relative_frame_idx]
                                        else:
                                            continue
                                    elif len(keypoints.shape) == 2:  # (V, C)
                                        current_keypoints = keypoints
                                    else:
                                        continue
                                    
                                    # 스켈레톤 그리기 (중복 객체 감지는 separated에서는 적용하지 않음)
                                    frame = self.draw_skeleton(frame, current_keypoints, person_idx, False)
                                    
                                    # 머리 위에 정보 표시
                                    head_pos = self.get_head_position(current_keypoints)
                                    if head_pos:
                                        track_id = person_info.get('track_id', person_id)
                                        composite_score = person_info.get('composite_score', 0.0)
                                        frame = self.draw_person_info_on_head(frame, head_pos, person_idx, 
                                                                             track_id, composite_score)
                                    
                            except Exception as e:
                                print(f"Error processing person {person_idx}: {e}")
                                if keypoints is not None:
                                    print(f"  keypoints shape: {keypoints.shape}")
                                print(f"  relative_frame_idx: {relative_frame_idx}")
                
                # 프레임 정보 표시
                frame_text = f"Frame: {frame_idx}/{total_frames}"
                cv2.putText(frame, frame_text, (self.config.box_padding, height - self.config.frame_info_margin), 
                           self.font, self.config.font_scale, self.config.text_color, self.config.font_thickness)
                
                # 화면 표시
                if not self.save:
                    cv2.imshow('Separated Overlay Visualization', frame)
                
                # 출력 비디오에 저장
                if out:
                    out.write(frame)
                
                # 키보드 입력 처리
                if not self.save:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                
                frame_idx += 1
            
            # 정리
            cap.release()
            if out:
                out.release()
            if not self.save:
                cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            print(f"Error in separated overlay visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_video_with_results(self, video_path: str, pkl_path: str, 
                                   output_video_path: str = None) -> bool:
        """비디오와 결과를 함께 시각화"""
        try:
            # 데이터 로드
            pkl_data = self.load_pkl_data(pkl_path)
            if not pkl_data:
                print(f"Failed to load PKL data from {pkl_path}")
                return False
            
            # 비디오 이름 추출하여 inference 결과 찾기
            video_name = Path(video_path).stem
            inference_results = self.get_inference_results_from_pkl(video_name)
            
            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video {video_path}")
                return False
            
            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.config.verbose:
                print(f"Video: {video_path}")
                print(f"PKL: {pkl_path}")
                print(f"Frames: {total_frames}, FPS: {fps}")
            
            # 출력 비디오 설정 (옵션)
            out = None
            if output_video_path:
                fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc_codec)
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 현재 프레임의 포즈 데이터 가져오기
                frame_data, relative_frame_idx = self._get_frame_data(pkl_data, frame_idx)
                
                # 관절 오버레이 그리기
                if frame_data:
                    # 겹침 구간 사람 감지
                    overlap_persons = self.detect_overlap_persons(frame_data, frame_idx)
                    
                    # 각 사람의 키포인트 그리기
                    persons_data = frame_data.get('annotation', {}).get('persons', {})
                    persons_list = sorted(persons_data.items(), key=lambda x: x[1].get('rank', float('inf')))
                    
                    for person_idx, (person_id, person_info) in enumerate(persons_list[:self.num_persons]):
                        if person_idx >= self.num_persons:
                            break
                            
                        try:
                            # tracked_window 구조에서는 keypoint가 직접 person 레벨에 있음
                            keypoints = person_info.get('keypoint')
                            if keypoints is not None:
                                # print(f"Person {person_idx}: keypoints shape: {keypoints.shape}, relative_frame_idx: {relative_frame_idx}")
                                
                                # keypoints 형태에 따라 처리
                                if len(keypoints.shape) == 4:  # (1, T, V, C) 형태
                                    # batch 차원 제거
                                    keypoints_3d = keypoints.squeeze(0)  # (T, V, C)
                                    if relative_frame_idx < keypoints_3d.shape[0]:
                                        current_keypoints = keypoints_3d[relative_frame_idx]  # (V, C)
                                    else:
                                        print(f"Person {person_idx}: relative_frame_idx {relative_frame_idx} >= keypoints length {keypoints_3d.shape[0]}")
                                        continue
                                elif len(keypoints.shape) == 3:  # (T, V, C) 형태
                                    if relative_frame_idx < keypoints.shape[0]:
                                        current_keypoints = keypoints[relative_frame_idx]  # (V, C)
                                    else:
                                        print(f"Person {person_idx}: relative_frame_idx {relative_frame_idx} >= keypoints length {keypoints.shape[0]}")
                                        continue
                                elif len(keypoints.shape) == 2:  # (V, C) 형태 - 단일 프레임
                                    current_keypoints = keypoints
                                else:
                                    print(f"Person {person_idx}: Unexpected keypoints shape: {keypoints.shape}")
                                    continue
                                
                                # 겹침 구간 확인
                                is_overlap = person_id in overlap_persons
                                
                                # 스켈레톤 그리기
                                frame = self.draw_skeleton(frame, current_keypoints, person_idx, is_overlap)
                            else:
                                print(f"Person {person_idx}: No keypoints data")
                        except Exception as e:
                            print(f"Error processing person {person_idx}: {e}")
                            if keypoints is not None:
                                print(f"  keypoints shape: {keypoints.shape}")
                            print(f"  relative_frame_idx: {relative_frame_idx}")
                
                # 윈도우별 결과 표시
                if inference_results:
                    frame = self.draw_window_results(frame, inference_results, frame_idx, total_frames)
                    
                    # 최종 결과 표시 - video_results.json에서 로드한 결과 사용
                    video_name = Path(video_path).stem
                    final_video_result = self.get_final_video_result(video_name)
                    final_prediction = final_video_result.get('video_prediction', 0)
                    final_confidence = final_video_result.get('avg_prediction_score', 0.0)
                    frame = self.draw_final_result(frame, final_prediction, final_confidence)
                
                # 프레임 정보 표시
                frame_text = f"Frame: {frame_idx}/{total_frames}"
                cv2.putText(frame, frame_text, (self.config.window_info_x, height - self.config.frame_info_margin), 
                           self.font, self.config.font_scale, self.config.text_color, self.config.font_thickness)
                
                # 화면 표시
                cv2.imshow(f'{self.mode} Visualization', frame)
                
                # 출력 비디오에 저장
                if out:
                    out.write(frame)
                
                # 단순 키보드 입력 처리 (q로 종료만)
                key = cv2.waitKey(30) & 0xFF  # 30ms 대기
                if key == ord('q'):
                    break
                
                frame_idx += 1
            
            # 정리
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            return False
    
    def _get_frame_data(self, pkl_data: List[Dict], frame_idx: int) -> Tuple[Dict, int]:
        """특정 프레임의 데이터 추출"""
        # pkl_data는 window_results 리스트
        # 현재 프레임이 속한 윈도우를 찾아서 해당 윈도우의 포즈 데이터에서 프레임 데이터 추출
        for window_result in pkl_data:
            start_frame = window_result.get('start_frame', 0)
            end_frame = window_result.get('end_frame', 0)
            
            # 현재 프레임이 이 윈도우에 속하는지 확인
            if start_frame <= frame_idx < end_frame:
                pose_data = window_result.get('pose_data', {})
                if pose_data:
                    # 윈도우 내에서의 상대적 프레임 인덱스 계산
                    relative_frame_idx = frame_idx - start_frame
                    
                    # pose_data는 tracked_window 구조
                    # annotation.persons에서 해당 프레임의 데이터와 상대적 프레임 인덱스 반환
                    return pose_data, relative_frame_idx
                    
        return {}, 0
    
    def _get_frame_data_separated(self, pkl_data: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Separated pipeline용 프레임 데이터 추출"""
        # separated pipeline pkl 구조: 윈도우 단위의 데이터
        
        # 1. 단일 윈도우 데이터인 경우 (dict with annotation)
        if isinstance(pkl_data, dict) and 'annotation' in pkl_data:
            return pkl_data, frame_idx
        
        # 2. 'windows' 키가 있는 경우 - 분리된 파이프라인의 일반적인 구조
        if isinstance(pkl_data, dict) and 'windows' in pkl_data:
            windows_data = pkl_data['windows']
            if isinstance(windows_data, list):
                for window_result in windows_data:
                    if isinstance(window_result, dict):
                        start_frame = window_result.get('start_frame', 0)
                        end_frame = window_result.get('end_frame', 0)
                        
                        if start_frame <= frame_idx < end_frame:
                            relative_frame_idx = frame_idx - start_frame
                            return window_result, relative_frame_idx
        
        # 3. 여러 윈도우가 있는 경우 (리스트 형태)
        if isinstance(pkl_data, list):
            for window_result in pkl_data:
                if isinstance(window_result, dict):
                    start_frame = window_result.get('start_frame', 0)
                    end_frame = window_result.get('end_frame', 0)
                    
                    if start_frame <= frame_idx < end_frame:
                        relative_frame_idx = frame_idx - start_frame
                        return window_result, relative_frame_idx
        
        # 4. dict인데 다른 구조인 경우 - 윈도우별 키가 있을 수 있음
        if isinstance(pkl_data, dict):
            # 윈도우 키들을 찾아봄 (예: 'window_0', 'window_1', etc.)
            for key, window_data in pkl_data.items():
                if isinstance(window_data, dict) and 'start_frame' in window_data and 'end_frame' in window_data:
                    start_frame = window_data.get('start_frame', 0)
                    end_frame = window_data.get('end_frame', 0)
                    
                    if start_frame <= frame_idx < end_frame:
                        relative_frame_idx = frame_idx - start_frame
                        return window_data, relative_frame_idx
            
            # 키 기반 접근이 안되면 첫 번째 윈도우 데이터 반환 시도
            if pkl_data:
                return pkl_data, frame_idx
        
        return {}, 0
    
    def _get_final_prediction(self, inference_results: List[Dict]) -> int:
        """최종 예측 결과 계산"""
        if not inference_results:
            return 0
        
        # 연속된 Fight 윈도우 개수 계산
        consecutive_fights = 0
        max_consecutive = 0
        
        for result in inference_results:
            if result.get('predicted_label', 0) == 1:
                consecutive_fights += 1
                max_consecutive = max(max_consecutive, consecutive_fights)
            else:
                consecutive_fights = 0
        
        # 설정된 임계값 이상이면 Fight로 판정
        return 1 if max_consecutive >= self.config.consecutive_threshold else 0
    
    def _get_final_confidence(self, inference_results: List[Dict]) -> float:
        """최종 신뢰도 계산"""
        if not inference_results:
            return 0.0
        
        predictions = [r.get('prediction', 0.0) for r in inference_results]
        return float(np.mean(predictions))
    
    def run(self):
        """실행 모드"""
        print(f"=== Enhanced Visualizer ({self.mode}) ===")
        
        # 비디오-PKL 쌍 찾기
        self.video_pkl_pairs = self.find_video_pkl_pairs()
        if not self.video_pkl_pairs:
            print("No matching video-PKL pairs found!")
            return
        
        print(f"Found {len(self.video_pkl_pairs)} video-PKL pairs")
        
        # 모든 쌍에 대해 순차적으로 시각화
        for idx, (video_path, pkl_path, label) in enumerate(self.video_pkl_pairs):
            print(f"\n[{idx + 1}/{len(self.video_pkl_pairs)}]")
            print(f"Video: {Path(video_path).name}")
            print(f"Label: {label}")
            if not self.save:
                print("Press 'q' to close current video and move to next")

            output_video_path = None
            if self.save:
                # OVERLAY_SUB_DIR 설정을 사용하여 저장할 디렉토리 생성
                if self.config.debug_mode:
                    print(f"Debug: self.save_dir = {self.save_dir}")
                    print(f"Debug: SAVE_OVERLAY_VIDEO = {self.config.SAVE_OVERLAY_VIDEO}")
                    print(f"Debug: OVERLAY_SUB_DIR = {self.config.OVERLAY_SUB_DIR}")
                
                if self.config.SAVE_OVERLAY_VIDEO and self.config.OVERLAY_SUB_DIR:
                    actual_save_dir = os.path.join(self.save_dir, self.config.OVERLAY_SUB_DIR)
                else:
                    actual_save_dir = self.save_dir
                
                if self.config.debug_mode:
                    print(f"Debug: actual_save_dir = {actual_save_dir}")
                os.makedirs(actual_save_dir, exist_ok=True)
                if self.config.debug_mode:
                    print(f"Debug: Directory created/exists: {os.path.exists(actual_save_dir)}")
                
                video_filename = Path(video_path).name
                output_video_path = os.path.join(actual_save_dir, f"{self.mode}_{video_filename}")
                print(f"Overlay video will be saved to: {output_video_path}")
            
            # 모드에 따라 적절한 시각화 메서드 호출
            if self.mode == 'inference_overlay':
                success = self.visualize_inference_overlay(video_path, pkl_path, output_video_path)
            elif self.mode == 'separated_overlay':
                success = self.visualize_separated_overlay(video_path, pkl_path, output_video_path)
            else:
                print(f"Unknown mode: {self.mode}")
                continue
            
            if not success:
                print("Failed to visualize current pair")
            else:
                print(f"Successfully processed: {Path(video_path).name}")
                if self.save and output_video_path:
                    if os.path.exists(output_video_path):
                        file_size = os.path.getsize(output_video_path)
                        print(f"Saved overlay video: {output_video_path} ({file_size} bytes)")
                    else:
                        print(f"Warning: Expected output file not found: {output_video_path}")
        
        print("\nAll videos processed!")
    
    def run_single_file(self, video_path: str, pkl_path: str, output_path: str = None):
        """단일 파일 실행"""
        print("=== Single File Visualization ===")
        print(f"Video: {video_path}")
        print(f"PKL: {pkl_path}")
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
            
        if not os.path.exists(pkl_path):
            print(f"PKL file not found: {pkl_path}")
            return False
        
        # 모드에 따라 적절한 메서드 호출
        if self.mode == 'inference_overlay':
            success = self.visualize_video_with_results(video_path, pkl_path, output_path)
        elif self.mode == 'separated_overlay':
            success = self.visualize_separated_overlay(video_path, pkl_path, output_path)
        else:
            print(f"Unknown mode: {self.mode}")
            return False
        
        if success:
            print("Visualization completed successfully!")
        else:
            print("Visualization failed!")
        
        return success


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Enhanced Visualizer with Inference and Separated Pipeline Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference overlay mode
  python visualizer.py inference_overlay --video-dir ./test_data --pkl-dir ./output/windows --num-person 2

  # Separated overlay mode with saving
  python visualizer.py separated_overlay --video-dir ./test_data --pkl-dir ./output/step2 --save --save-dir ./overlay_results

  # Single file processing
  python visualizer.py inference_overlay --video-dir ./video.mp4 --pkl-dir ./results.pkl --save --save-dir ./output
        """
    )
    
    # 위치 인수: 모드 선택
    parser.add_argument('mode', choices=['inference_overlay', 'separated_overlay'], 
                       help='Overlay mode: inference_overlay or separated_overlay')
    
    # 필수 인수
    parser.add_argument('--video-dir', required=True, 
                       help='Input video directory or file path')
    parser.add_argument('--pkl-dir', required=True, 
                       help='PKL files directory path (windows for inference, step2 for separated)')
    
    # 선택적 인수
    parser.add_argument('--save', action='store_true', 
                       help='Save overlay video to file (default: real-time display only)')
    parser.add_argument('--save-dir', default='./overlay_output', 
                       help='Output directory for saved overlay videos (default: ./overlay_output)')
    parser.add_argument('--num-person', type=int, default=2, 
                       help='Number of top persons to display skeleton overlay (default: 2)')
    
    args = parser.parse_args()
    
    try:
        print(f"=== Enhanced Visualizer ({args.mode}) ===")
        print(f"Video Directory: {args.video_dir}")
        print(f"PKL Directory: {args.pkl_dir}")
        print(f"Save Mode: {args.save}")
        if args.save:
            print(f"Save Directory: {args.save_dir}")
        print(f"Number of Persons: {args.num_person}")
        print()
        
        # 경로 확인
        if not os.path.exists(args.video_dir):
            print(f"Error: Video directory/file not found: {args.video_dir}")
            return
            
        if not os.path.exists(args.pkl_dir):
            print(f"Error: PKL directory not found: {args.pkl_dir}")
            return
        
        # config.py에서 설정 로드 시도
        try:
            from ..configs.visualizer_config import config as vis_config
        except ImportError:
            print("Warning: visualizer_config.py not found. Using default settings.")
            # 기본 설정으로 대체할 클래스나 딕셔너리 정의
            class DefaultConfig:
                def __init__(self):
                    self.default_input_dir = './test_data'
                    self.default_output_dir = './output'
                    self.max_persons = 2
                    self.confidence_threshold = 0.3
                    self.verbose = True
                    self.fourcc_codec = 'mp4v'
                    self.colors = [(255,0,0), (0,255,0), (0,0,255)]
                    self.fight_color = (0,0,255)
                    self.nonfight_color = (0,255,0)
                    self.text_color = (255,255,255)
                    self.font_scale = 0.7
                    self.font_thickness = 2
                    self.title_font_scale = 0.8
                    self.title_font_thickness = 2
                    self.box_padding = 10
                    self.line_thickness = 2
                    self.keypoint_radius = 3
                    self.skeleton_connections = []
                    self.SAVE_OVERLAY_VIDEO = True
                    self.OVERLAY_SUB_DIR = 'overlay'
                    self.debug_mode = False
            vis_config = DefaultConfig()

        # 시각화 도구 실행
        visualizer = EnhancedVisualizer(
            mode=args.mode,
            video_dir=args.video_dir,
            pkl_dir=args.pkl_dir,
            save=args.save,
            save_dir=args.save_dir,
            num_person=args.num_person,
            config=vis_config  # 설정 객체 전달
        )
        
        visualizer.run()
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()