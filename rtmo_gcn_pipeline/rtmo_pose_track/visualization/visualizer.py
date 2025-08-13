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

# 분석 로거 import
try:
    from ..utils.pose_analysis_logger import PoseAnalysisLogger
except ImportError:
    try:
        from utils.pose_analysis_logger import PoseAnalysisLogger
    except ImportError:
        print("Warning: Could not import PoseAnalysisLogger. Analysis logging disabled.")
        PoseAnalysisLogger = None




class EnhancedVisualizer:
    """향상된 시각화 클래스 - Inference 및 Separated Pipeline 지원"""
    
    def __init__(self, mode='inference', video_dir=None, pkl_dir=None, 
                 save=False, save_dir=None, num_person=2, config=None, stage='stage1'):
        if config is None:
            raise ValueError("A config object must be provided.")

        self.config = config
        self.mode = mode
        self.stage = stage  # stage1/step1 (poses) or stage2/step2 (tracking)
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
        
        # 분석 로거 초기화
        self.analysis_logger = None
        if PoseAnalysisLogger and mode == 'separated_overlay':
            log_dir = os.path.join(self.save_dir, 'analysis_logs')
            self.analysis_logger = PoseAnalysisLogger(log_dir, 'comparison_analysis')
        
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
            # separated 모드: 모든 .pkl 파일 찾기 (서브폴더 포함)
            pkl_files = []
            for root, dirs, files in os.walk(self.pkl_dir):
                for file in files:
                    if file.endswith('.pkl'):
                        pkl_path = os.path.join(root, file)
                        pkl_files.append(pkl_path)
        
        else:
            print(f"Unknown mode: {self.mode}")
            return pairs
        
        print(f"Found {len(pkl_files)} PKL files")
        
        # 각 PKL 파일에 대해 매칭되는 비디오 찾기
        for pkl_path in pkl_files:
            pkl_filename = Path(pkl_path).stem
            # stage에 따른 접미사 제거하여 비디오명 추출
            if self.stage in ['stage1', 'step1']:
                video_name = pkl_filename.replace('_poses', '').replace('_pose', '').replace('_skeleton', '').replace('_annotation', '')
            elif self.stage in ['stage2', 'step2']:
                video_name = pkl_filename.replace('_windows', '').replace('_tracking', '').replace('_annotation', '')
            else:
                video_name = pkl_filename.replace('_windows', '').replace('_poses', '').replace('_pose', '').replace('_skeleton', '').replace('_annotation', '')
            
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
            print(f"No keypoints to draw for person {person_idx}")
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
                
            # stage별 신뢰도 임계값 조정
            confidence_threshold = self.config.confidence_threshold
            if hasattr(self, 'stage') and self.stage in ['stage2', 'step2']:
                confidence_threshold = 0.1  # tracking 데이터는 더 낮은 임계값 사용
            
            # 관절점 그리기
            drawn_points = 0
            skipped_points = 0
            for i, (x, y, score) in enumerate(coords_scores):
                if score > confidence_threshold and x > 0 and y > 0:
                    cv2.circle(img, (int(x), int(y)), self.config.keypoint_radius, color, -1)
                    drawn_points += 1
                else:
                    skipped_points += 1
            
            if drawn_points == 0 and skipped_points > 0:  # 그려진 점이 없을 때만 로그
                print(f"Person {person_idx}: Drew {drawn_points} points, skipped {skipped_points} points (threshold: {confidence_threshold})")
            
            # 스켈레톤 연결선 그리기  
            for connection in self.config.skeleton_connections:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-based to 0-based
                if pt1_idx < len(coords_scores) and pt2_idx < len(coords_scores):
                    x1, y1, score1 = coords_scores[pt1_idx]
                    x2, y2, score2 = coords_scores[pt2_idx]
                    
                    if score1 > confidence_threshold and score2 > confidence_threshold and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, self.config.line_thickness)
        
        except Exception as e:
            print(f"Error drawing skeleton: {e}, keypoints shape: {keypoints.shape}")
            
        return img
    
    def draw_skeleton_with_custom_color(self, img: np.ndarray, keypoints: np.ndarray, 
                                       color: Tuple[int, int, int], is_overlap: bool = False) -> np.ndarray:
        """커스텀 색상으로 스켈레톤 그리기"""
        if keypoints is None or len(keypoints) == 0:
            return img
            
        # keypoints 형태 확인 및 처리
        try:
            # keypoints가 (17, 3) 형태인지 확인
            if keypoints.shape[-1] == 3:
                # (x, y, score) 형태
                coords_scores = keypoints
            elif keypoints.shape[-1] == 2:
                # (x, y) 형태인 경우 기본 score 1.0 추가
                if len(keypoints.shape) == 2:  # (17, 2)
                    ones_shape = (keypoints.shape[0], 1)
                else:  # (T, 17, 2) 등의 경우
                    ones_shape = keypoints.shape[:-1] + (1,)
                coords_scores = np.concatenate([keypoints, np.ones(ones_shape)], axis=-1)
            else:
                print(f"Unexpected keypoints shape: {keypoints.shape}")
                return img
                
            # stage별 신뢰도 임계값 조정
            confidence_threshold = self.config.confidence_threshold
            if hasattr(self, 'stage') and self.stage in ['stage2', 'step2']:
                confidence_threshold = 0.1  # tracking 데이터는 더 낮은 임계값 사용
            
            # 관절점 그리기
            for i, (x, y, score) in enumerate(coords_scores):
                if score > confidence_threshold and x > 0 and y > 0:
                    cv2.circle(img, (int(x), int(y)), self.config.keypoint_radius, color, -1)
            
            # 스켈레톤 연결선 그리기  
            for connection in self.config.skeleton_connections:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-based to 0-based
                if pt1_idx < len(coords_scores) and pt2_idx < len(coords_scores):
                    x1, y1, score1 = coords_scores[pt1_idx]
                    x2, y2, score2 = coords_scores[pt2_idx]
                    
                    if score1 > confidence_threshold and score2 > confidence_threshold and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, self.config.line_thickness)
        
        except Exception as e:
            print(f"Error drawing skeleton with custom color: {e}, keypoints shape: {keypoints.shape}")
            
        return img
    
    def detect_overlap_persons(self, window_data: Dict, frame_idx: int) -> List[int]:
        """겹침 구간의 사람 객체 감지 - 기능 비활성화"""
        # 중복 색상 기능 제거 - 항상 빈 리스트 반환
        return []
    
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
                if frame_data and 'persons' in frame_data:
                    annotation = frame_data  # 변환된 데이터는 이미 annotation 형태
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
                                    print(f"Person {person_id}: keypoints shape {keypoints.shape}, relative_frame_idx={relative_frame_idx}")
                                    
                                    # keypoints 형태 처리 (separated는 보통 (1, T, V, C) 형태)
                                    if len(keypoints.shape) == 4:  # (1, T, V, C)
                                        keypoints_3d = keypoints.squeeze(0)  # (T, V, C)
                                        print(f"Person {person_id}: After squeeze: {keypoints_3d.shape}")
                                        if relative_frame_idx < keypoints_3d.shape[0]:
                                            current_keypoints = keypoints_3d[relative_frame_idx]
                                            print(f"Person {person_id}: current_keypoints shape: {current_keypoints.shape}")
                                            print(f"Person {person_id}: sample coords: {current_keypoints[:3, :].flatten()}")
                                        else:
                                            print(f"Person {person_id}: relative_frame_idx {relative_frame_idx} >= {keypoints_3d.shape[0]}")
                                            continue
                                    elif len(keypoints.shape) == 3:  # (T, V, C)
                                        if relative_frame_idx < keypoints.shape[0]:
                                            current_keypoints = keypoints[relative_frame_idx]
                                            print(f"Person {person_id}: current_keypoints shape: {current_keypoints.shape}")
                                        else:
                                            print(f"Person {person_id}: relative_frame_idx {relative_frame_idx} >= {keypoints.shape[0]}")
                                            continue
                                    elif len(keypoints.shape) == 2:  # (V, C)
                                        current_keypoints = keypoints
                                        print(f"Person {person_id}: using 2D keypoints: {current_keypoints.shape}")
                                    else:
                                        print(f"Person {person_id}: unexpected keypoints shape: {keypoints.shape}")
                                        continue
                                    
                                    # track_id 기반 색상 할당 (stage2에서만)
                                    if self.stage in ['stage2', 'step2']:
                                        track_id = person_info.get('track_id', person_id)
                                        # track_id를 숫자로 변환하여 색상 인덱스로 사용
                                        try:
                                            color_idx = int(track_id) % len(self.config.colors)
                                        except (ValueError, TypeError):
                                            color_idx = person_idx % len(self.config.colors)
                                    else:
                                        color_idx = person_idx % len(self.config.colors)
                                    
                                    # 스켈레톤 그리기 (중복 객체 감지는 separated에서는 적용하지 않음)
                                    color = self.config.colors[color_idx]
                                    frame = self.draw_skeleton_with_custom_color(frame, current_keypoints, color, False)
                                    print(f"Person {person_id}: skeleton drawn")
                                    
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
        """특정 프레임의 데이터 추출 - 순차적 윈도우 할당"""
        # 윈도우를 start_frame 순으로 정렬
        sorted_windows = sorted(pkl_data, key=lambda w: w.get('start_frame', 0))
        
        # 첫 번째 윈도우는 전체 clip_len 길이만큼 사용
        if len(sorted_windows) > 0:
            first_window = sorted_windows[0]
            first_start = first_window.get('start_frame', 0)
            first_end = first_start + 100  # clip_len
            
            if first_start <= frame_idx < first_end:
                pose_data = first_window.get('pose_data', {})
                if pose_data:
                    relative_frame_idx = frame_idx - first_start
                    return pose_data, relative_frame_idx
        
        # 두 번째 윈도우부터는 스트라이드 길이만큼만 사용
        for i, window_result in enumerate(sorted_windows[1:], 1):
            prev_window = sorted_windows[i-1]
            current_start = window_result.get('start_frame', 0)
            
            # 이전 윈도우의 끝부터 현재 윈도우의 스트라이드 길이만큼
            display_start = prev_window.get('start_frame', 0) + 100  # clip_len
            display_end = current_start + 100  # clip_len
            
            if display_start <= frame_idx < display_end:
                pose_data = window_result.get('pose_data', {})
                if pose_data:
                    relative_frame_idx = frame_idx - current_start
                    return pose_data, relative_frame_idx
                    
        return {}, 0
    
    def _convert_pose_data_sample_to_annotation(self, pose_sample, frame_idx: int) -> Dict:
        """MMPose PoseDataSample을 annotation 형태로 변환"""
        annotation = {
            'persons': {}
        }
        
        try:
            # PoseDataSample에서 키포인트 데이터 추출
            if hasattr(pose_sample, 'pred_instances'):
                pred_instances = pose_sample.pred_instances
                
                # 키포인트와 점수 추출
                if hasattr(pred_instances, 'keypoints'):
                    keypoints = pred_instances.keypoints  # shape: (N, 17, 2) or (N, 17, 3)
                    scores = getattr(pred_instances, 'keypoint_scores', None)  # shape: (N, 17)
                    
                    print(f"Frame {frame_idx}: Found {keypoints.shape[0]} persons, keypoints shape: {keypoints.shape}")
                    
                    # 각 사람별로 처리
                    for person_idx in range(keypoints.shape[0]):
                        person_keypoints = keypoints[person_idx]  # (17, 2) or (17, 3)
                        
                        # 키포인트 좌표 유효성 확인
                        valid_keypoints = np.any(person_keypoints[:, :2] > 0, axis=1)
                        valid_count = np.sum(valid_keypoints)
                        print(f"Person {person_idx}: {valid_count}/17 valid keypoints")
                        
                        # confidence가 있는 경우 추가
                        if scores is not None and scores.shape[0] > person_idx:
                            person_scores = scores[person_idx]  # (17,)
                            # keypoints와 scores를 결합하여 (17, 3) 형태로 만들기
                            if person_keypoints.shape[1] == 2:  # (17, 2)인 경우
                                confidence_keypoints = np.zeros((17, 3))
                                confidence_keypoints[:, :2] = person_keypoints
                                confidence_keypoints[:, 2] = person_scores
                                person_keypoints = confidence_keypoints
                        
                        # annotation 형태로 변환 - (1, 1, 17, 2) 또는 (1, 1, 17, 3)
                        if person_keypoints.shape[1] == 3:  # confidence 포함
                            keypoint_data = person_keypoints[:, :2].reshape(1, 1, 17, 2)
                        else:  # confidence 없음
                            keypoint_data = person_keypoints.reshape(1, 1, 17, 2)
                        
                        annotation['persons'][str(person_idx)] = {
                            'keypoint': keypoint_data
                        }
                        
                        # 실제 키포인트 좌표값 확인
                        sample_coords = keypoint_data[0, 0, :3, :]  # 첫 3개 키포인트만 확인
                        print(f"Person {person_idx} sample coords: {sample_coords.flatten()}")
                
        except Exception as e:
            print(f"Error converting PoseDataSample to annotation: {e}")
            print(f"PoseDataSample type: {type(pose_sample)}")
            if hasattr(pose_sample, '__dict__'):
                print(f"Available attributes: {list(pose_sample.__dict__.keys())}")
        
        return annotation
    
    def _get_frame_data_separated(self, pkl_data: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Separated pipeline용 프레임 데이터 추출"""
        # stage에 따라 다른 처리 방식 사용
        
        # Stage1/Step1: PoseDataSample 구조 처리 (step1_poses.pkl)
        if (self.stage in ['stage1', 'step1']) and isinstance(pkl_data, dict) and 'poses' in pkl_data:
            # MMPose PoseDataSample 리스트를 annotation 형태로 변환
            poses_list = pkl_data['poses']
            if poses_list and frame_idx < len(poses_list):
                pose_sample = poses_list[frame_idx]
                converted_data = self._convert_pose_data_sample_to_annotation(pose_sample, frame_idx)
                return converted_data, 0  # relative frame은 0으로 설정 (이미 프레임별로 처리됨)
        
        # Stage2/Step2: Sequential window 처리 (겹침 구간은 이전 윈도우 우선)
        if (self.stage in ['stage2', 'step2']) and isinstance(pkl_data, dict) and 'windows' in pkl_data:
            windows_list = pkl_data['windows']
            print(f"Processing stage2/step2 data for frame {frame_idx}, found {len(windows_list)} windows")
            
            # 윈도우를 start_frame 순으로 정렬
            sorted_windows = sorted(windows_list, key=lambda x: x.get('start_frame', 0))
            
            # Sequential 처리: 첫 윈도우는 전체, 이후는 stride 구간만
            clip_len = 100  # 설정에서 가져와야 함
            stride = 50     # 설정에서 가져와야 함
            
            for window_idx, window_result in enumerate(sorted_windows):
                if isinstance(window_result, dict):
                    start_frame = window_result.get('start_frame', 0)
                    end_frame = window_result.get('end_frame', 0)
                    
                    # 첫 번째 윈도우: 전체 구간 (0 ~ clip_len)
                    if window_idx == 0:
                        effective_start = start_frame
                        effective_end = start_frame + clip_len
                        print(f"Window {window_idx} (first): effective frames {effective_start}-{effective_end}")
                    else:
                        # 이후 윈도우: stride 구간만 (이전 윈도우 끝 ~ 현재 윈도우 끝)
                        prev_window = sorted_windows[window_idx - 1]
                        prev_end = prev_window.get('start_frame', 0) + clip_len
                        effective_start = prev_end
                        effective_end = end_frame
                        print(f"Window {window_idx}: effective frames {effective_start}-{effective_end} (stride only)")
                    
                    if effective_start <= frame_idx < effective_end:
                        relative_frame_idx = frame_idx - start_frame  # 원본 윈도우 기준으로 relative 계산
                        print(f"Frame {frame_idx} matches window {window_idx}, relative_frame_idx: {relative_frame_idx}")
                        
                        # annotation 데이터가 있는 경우 바로 반환
                        if 'annotation' in window_result:
                            annotation = window_result['annotation']
                            if 'persons' in annotation:
                                print(f"Window {window_idx} annotation has {len(annotation['persons'])} persons")
                            return annotation, relative_frame_idx
                        else:
                            # window_result 자체가 annotation 형태일 수 있음
                            if 'persons' in window_result:
                                print(f"Window {window_idx} has {len(window_result['persons'])} persons directly")
                            return window_result, relative_frame_idx
        
        # Legacy 처리: stage 옵션이 없거나 다른 구조인 경우
        
        # 단일 윈도우 데이터인 경우 (dict with annotation)
        if isinstance(pkl_data, dict) and 'annotation' in pkl_data:
            return pkl_data, frame_idx
        
        # 'windows' 키가 있는 경우 - 분리된 파이프라인의 일반적인 구조
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
        
        # 여러 윈도우가 있는 경우 (리스트 형태)
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
    
    def run_analysis(self, step1_pkl_path: str, step2_pkl_path: str, video_path: str):
        """Step1 vs Step2 비교 분석 실행"""
        if not self.analysis_logger:
            print("Analysis logger not available")
            return False
        
        print(f"Starting step1 vs step2 analysis...")
        print(f"Step1 PKL: {step1_pkl_path}")
        print(f"Step2 PKL: {step2_pkl_path}")
        print(f"Video: {video_path}")
        
        try:
            # 설정 로그
            import configs.separated_pipeline_config as sep_config
            config_dict = {
                'track_high_thresh': getattr(sep_config, 'track_high_thresh', 0.2),
                'track_low_thresh': getattr(sep_config, 'track_low_thresh', 0.1),
                'track_max_disappeared': getattr(sep_config, 'track_max_disappeared', 30),
                'track_min_hits': getattr(sep_config, 'track_min_hits', 2),
                'quality_threshold': getattr(sep_config, 'quality_threshold', 0.2),
                'min_track_length': getattr(sep_config, 'min_track_length', 5),
                'movement_weight': getattr(sep_config, 'movement_weight', 0.40),
                'position_weight': getattr(sep_config, 'position_weight', 0.15),
                'interaction_weight': getattr(sep_config, 'interaction_weight', 0.30),
                'temporal_consistency_weight': getattr(sep_config, 'temporal_consistency_weight', 0.08),
                'persistence_weight': getattr(sep_config, 'persistence_weight', 0.02)
            }
            self.analysis_logger.log_config(config_dict)
            
            # Step1 데이터 로드 및 분석
            with open(step1_pkl_path, 'rb') as f:
                step1_data = pickle.load(f)
            
            # Step2 데이터 로드 및 분석  
            with open(step2_pkl_path, 'rb') as f:
                step2_data = pickle.load(f)
            
            # 비디오 정보
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            print(f"Analyzing {total_frames} frames...")
            
            # 프레임별 분석
            for frame_idx in range(total_frames):
                if frame_idx % 100 == 0:
                    print(f"Analyzing frame {frame_idx}/{total_frames}")
                
                # Step1 데이터 분석
                step1_frame_data, _ = self._get_step1_frame_data(step1_data, frame_idx)
                if step1_frame_data:
                    self.analysis_logger.log_step1_frame(frame_idx, step1_frame_data)
                
                # Step2 데이터 분석
                step2_frame_data, relative_idx, window_idx = self._get_step2_frame_data(step2_data, frame_idx)
                if step2_frame_data:
                    self.analysis_logger.log_step2_frame(frame_idx, window_idx, step2_frame_data, relative_idx)
                
                # 프레임별 비교
                self.analysis_logger.compare_frames(frame_idx)
            
            # 분석 결과 저장
            analysis_file, summary_file = self.analysis_logger.save_analysis()
            
            print(f"\n=== 분석 완료 ===")
            print(f"상세 분석: {analysis_file}")
            print(f"요약 보고서: {summary_file}")
            
            return True
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_step1_frame_data(self, step1_data: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Step1 프레임 데이터 추출"""
        if isinstance(step1_data, dict) and 'poses' in step1_data:
            poses_list = step1_data['poses']
            if poses_list and frame_idx < len(poses_list):
                pose_sample = poses_list[frame_idx]
                converted_data = self._convert_pose_data_sample_to_annotation(pose_sample, frame_idx)
                return converted_data, 0
        return {}, 0
    
    def _get_step2_frame_data(self, step2_data: Dict, frame_idx: int) -> Tuple[Dict, int, int]:
        """Step2 프레임 데이터 추출 (window_idx 포함)"""
        if isinstance(step2_data, dict) and 'windows' in step2_data:
            windows_list = step2_data['windows']
            sorted_windows = sorted(windows_list, key=lambda x: x.get('start_frame', 0))
            
            clip_len = 100
            
            for window_idx, window_result in enumerate(sorted_windows):
                if isinstance(window_result, dict):
                    start_frame = window_result.get('start_frame', 0)
                    end_frame = window_result.get('end_frame', 0)
                    
                    # Sequential 처리 방식 적용
                    if window_idx == 0:
                        effective_start = start_frame
                        effective_end = start_frame + clip_len
                    else:
                        prev_window = sorted_windows[window_idx - 1]
                        prev_end = prev_window.get('start_frame', 0) + clip_len
                        effective_start = prev_end
                        effective_end = end_frame
                    
                    if effective_start <= frame_idx < effective_end:
                        relative_frame_idx = frame_idx - start_frame
                        
                        if 'annotation' in window_result:
                            return window_result['annotation'], relative_frame_idx, window_idx
                        else:
                            return window_result, relative_frame_idx, window_idx
        
        return {}, 0, -1
    
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