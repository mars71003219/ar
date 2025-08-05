#!/usr/bin/env python3
"""
Inference Result Visualizer
원본 비디오와 추론 결과 PKL을 매칭하여 관절 오버레이와 분류 결과를 표시하는 시각화 시스템
"""

import os
import sys
import cv2
import pickle
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 설정 파일에서 모든 설정 로드
try:
    from configs.visualizer_config import *
except ImportError as e:
    print(f"Error: visualizer_config.py 파일을 찾을 수 없습니다!")
    print(f"파일 경로를 확인하세요: configs/visualizer_config.py")
    print(f"상세 오류: {e}")
    sys.exit(1)


class InferenceResultVisualizer:
    """추론 결과 시각화 클래스"""
    
    def __init__(self, input_dir_path=None, output_dir_path=None):
        """
        Args:
            input_dir_path: 입력 비디오 디렉토리 경로
            output_dir_path: 출력 PKL 디렉토리 경로
        """
        # 설정에서 값들 가져오기
        self.input_dir = input_dir_path or input_dir
        self.output_dir = output_dir_path or output_dir
        self.num_persons = num_persons
        
        # COCO 17 keypoints 연결 정보
        self.skeleton = skeleton_connections
        
        # 색상 설정
        self.person_colors = person_colors
        self.overlap_color = overlap_color
        self.fight_color = fight_color
        self.nonfight_color = nonfight_color
        self.final_fight_color = final_fight_color
        self.final_nonfight_color = final_nonfight_color
        
        # UI 폰트 설정
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.title_font_scale = title_font_scale
        self.title_font_thickness = title_font_thickness
        
        # 비디오-PKL 매칭 정보
        self.video_pkl_pairs = []
        self.current_pair_idx = 0
        
    def find_video_pkl_pairs(self) -> List[Tuple[str, str, str]]:
        """원본 비디오와 PKL 파일을 매칭하여 리스트 반환"""
        pairs = []
        
        # 디렉토리 모드: 자동 매칭
        pairs = self._find_pairs_from_directories()
            
        return pairs
    
    def _find_pairs_from_directories(self) -> List[Tuple[str, str, str]]:
        """디렉토리에서 비디오-PKL 쌍 찾기"""
        pairs = []
        
        # output_dir에서 PKL 파일 찾기
        pkl_files = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('_windows.pkl'):
                    pkl_path = os.path.join(root, file)
                    pkl_files.append(pkl_path)
        
        if verbose:
            print(f"Found {len(pkl_files)} PKL files")
        
        # 각 PKL 파일에 대해 매칭되는 비디오 찾기
        for pkl_path in pkl_files:
            video_name = Path(pkl_path).stem.replace('_windows', '')
            
            # input_dir에서 매칭되는 비디오 찾기
            for ext in supported_video_extensions:
                pattern_path = os.path.join(self.input_dir, '**', f"{video_name}.{ext}")
                matches = glob.glob(pattern_path, recursive=True)
                if matches:
                    video_path = matches[0]
                    label_folder = Path(video_path).parent.name
                    pairs.append((video_path, pkl_path, label_folder))
                    if verbose:
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
        inference_dir = os.path.join(self.output_dir, "windows")
        
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
    
    def draw_skeleton(self, img: np.ndarray, keypoints: np.ndarray, person_idx: int, 
                     is_overlap: bool = False) -> np.ndarray:
        """스켈레톤 그리기"""
        if keypoints is None or len(keypoints) == 0:
            return img
            
        # 색상 선택
        if is_overlap:
            color = self.overlap_color
        else:
            color = self.person_colors[person_idx % len(self.person_colors)]
        
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
                if score > confidence_threshold:  # 신뢰도 임계값
                    cv2.circle(img, (int(x), int(y)), 4, color, -1)
            
            # 스켈레톤 연결선 그리기  
            for connection in self.skeleton:
                pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 1-based to 0-based
                if pt1_idx < len(coords_scores) and pt2_idx < len(coords_scores):
                    x1, y1, score1 = coords_scores[pt1_idx]
                    x2, y2, score2 = coords_scores[pt2_idx]
                    
                    if score1 > confidence_threshold and score2 > confidence_threshold:
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
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
    
    def draw_window_results(self, img: np.ndarray, window_results: List[Dict], 
                           current_frame: int, total_frames: int) -> np.ndarray:
        """윈도우별 분류 결과 표시"""
        h, w = img.shape[:2]
        
        # 현재 프레임이 속한 윈도우 찾기
        current_windows = []
        for window in window_results:
            start_frame = window.get('start_frame', 0)
            end_frame = window.get('end_frame', 0)
            if start_frame <= current_frame < end_frame:
                current_windows.append(window)
        
        # 윈도우 정보 표시
        y_offset = window_info_y_start
        for i, window in enumerate(current_windows):
            window_idx = window.get('window_idx', i)
            prediction = window.get('prediction', 0.0)
            predicted_label = window.get('predicted_label', 0)
            
            # 배경 색상 (Fight: 빨간색, NonFight: 파란색)
            bg_color = self.fight_color if predicted_label == 1 else self.nonfight_color
            text_color = (255, 255, 255)
            
            # 텍스트 내용
            text = f"Window {window_idx}: {'Fight' if predicted_label == 1 else 'NonFight'} ({prediction:.3f})"
            
            # 텍스트 크기 계산
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
            
            # 배경 사각형 그리기
            cv2.rectangle(img, (window_info_x, y_offset - 25), 
                         (window_info_x + 10 + text_size[0], y_offset + 5), bg_color, -1)
            
            # 텍스트 그리기
            cv2.putText(img, text, (window_info_x + 5, y_offset), 
                       self.font, self.font_scale, text_color, self.font_thickness)
            
            y_offset += window_info_y_step
        
        return img
    
    def draw_final_result(self, img: np.ndarray, final_prediction: int, confidence: float) -> np.ndarray:
        """최종 판정 결과 표시"""
        h, w = img.shape[:2]
        
        # 최종 결과 텍스트
        final_text = f"Final: {'FIGHT DETECTED' if final_prediction == 1 else 'NO FIGHT'}"
        confidence_text = f"Confidence: {confidence:.3f}"
        
        # 색상 설정
        bg_color = self.final_fight_color if final_prediction == 1 else self.final_nonfight_color
        text_color = (255, 255, 255)
        
        # 텍스트 크기 계산
        (final_text_width, final_text_height), _ = cv2.getTextSize(
            final_text, self.font, self.title_font_scale, self.title_font_thickness)
        (conf_text_width, conf_text_height), _ = cv2.getTextSize(
            confidence_text, self.font, self.font_scale, self.font_thickness)
        
        # 배경 사각형 너비
        rect_w = max(final_text_width, conf_text_width) + 20
        
        # 동적 높이 및 Y 좌표 계산
        top_margin = 15
        line_spacing = 10
        bottom_margin = 15
        
        rect_y = final_result_margin
        
        # 첫 줄 Y 좌표 (텍스트의 baseline 기준)
        final_text_y = rect_y + top_margin + final_text_height
        # 두 번째 줄 Y 좌표
        conf_text_y = final_text_y + line_spacing + conf_text_height
        
        # 전체 높이
        rect_h = top_margin + final_text_height + line_spacing + conf_text_height + bottom_margin
        
        # 배경 사각형 위치 (우상단)
        rect_x = w - rect_w - final_result_margin
        
        # 배경 사각형 그리기
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), bg_color, -1)
        
        # 테두리 그리기
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), 2)
        
        # 텍스트 그리기
        cv2.putText(img, final_text, (rect_x + 10, final_text_y), self.font, 
                   self.title_font_scale, text_color, self.title_font_thickness)
        cv2.putText(img, confidence_text, (rect_x + 10, conf_text_y), self.font, 
                   self.font_scale, text_color, self.font_thickness)
        
        return img
    
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
            
            if verbose:
                print(f"Video: {video_path}")
                print(f"PKL: {pkl_path}")
                print(f"Frames: {total_frames}, FPS: {fps}")
            
            # 출력 비디오 설정 (옵션)
            out = None
            if output_video_path:
                fourcc = cv2.VideoWriter_fourcc(*output_video_codec)
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
                    
                    # 최종 결과 표시
                    final_prediction = self._get_final_prediction(inference_results)
                    final_confidence = self._get_final_confidence(inference_results)
                    frame = self.draw_final_result(frame, final_prediction, final_confidence)
                
                # 프레임 정보 표시
                frame_text = f"Frame: {frame_idx}/{total_frames}"
                cv2.putText(frame, frame_text, (window_info_x, height - frame_info_margin), 
                           self.font, self.font_scale, (255, 255, 255), self.font_thickness)
                
                # 화면 표시
                cv2.imshow('Inference Result Visualization', frame)
                
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
        return 1 if max_consecutive >= consecutive_threshold else 0
    
    def _get_final_confidence(self, inference_results: List[Dict]) -> float:
        """최종 신뢰도 계산"""
        if not inference_results:
            return 0.0
        
        predictions = [r.get('prediction', 0.0) for r in inference_results]
        return float(np.mean(predictions))
    
    def run(self):
        """단순 실행 모드"""
        print("=== Inference Result Visualizer ===")
        
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
            print("Press 'q' to close current video and move to next")

            output_video_path = None
            if SAVE_OVERLAY_VIDEO:
                # 저장할 디렉토리 생성
                save_directory = os.path.join(self.output_dir, OVERLAY_SUB_DIR, label)
                os.makedirs(save_directory, exist_ok=True)

                # 저장할 파일 경로와 이름 지정
                video_filename = Path(video_path).name
                output_video_path = os.path.join(save_directory, video_filename)
                print(f"Overlay video will be saved to: {output_video_path}")
            
            success = self.visualize_video_with_results(video_path, pkl_path, output_video_path)
            
            if not success:
                print("Failed to visualize current pair")
        
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
        
        success = self.visualize_video_with_results(video_path, pkl_path, output_path)
        
        if success:
            print("Visualization completed successfully!")
        else:
            print("Visualization failed!")
        
        return success


def print_usage():
    """사용법 출력"""
    print("=" * 50)
    print(" Simple Inference Result Visualizer")
    print("=" * 50)
    print()
    print("사용법:")
    print("  python visualizer.py [input_dir] [output_dir]")
    print()
    print("예시:")
    print("  # 기본 디렉토리 사용")
    print("  python visualizer.py")
    print()
    print("  # 사용자 지정 디렉토리")
    print("  python visualizer.py /path/to/videos /path/to/output")
    print()
    print("  # 단일 파일 모드")
    print("  python -c \"from visualizer import InferenceResultVisualizer; v = InferenceResultVisualizer(); v.run_single_file('/path/to/video.mp4', '/path/to/result.pkl')\"")
    print()
    print("제어:")
    print("  q: 현재 비디오 종료 후 다음으로 이동")
    print("=" * 50)


def main():
    """메인 실행 함수"""
    try:
        args = sys.argv[1:]
        
        # 도움말 요청 확인
        if '--help' in args or '-h' in args:
            print_usage()
            return
        
        input_dir_arg = None
        output_dir_arg = None
        
        if len(args) >= 1:
            input_dir_arg = args[0]
        if len(args) >= 2:
            output_dir_arg = args[1]
        
        print("Simple Inference Result Visualizer를 시작합니다...")
        print()
        
        # 시각화 도구 실행
        visualizer = InferenceResultVisualizer(input_dir_arg, output_dir_arg)
        
        # 경로 확인
        if not os.path.exists(visualizer.input_dir):
            print(f"Warning: Input directory not found: {visualizer.input_dir}")
            
        if not os.path.exists(visualizer.output_dir):
            print(f"Warning: Output directory not found: {visualizer.output_dir}")
            
        print(f"Input Directory: {visualizer.input_dir}")
        print(f"Output Directory: {visualizer.output_dir}")
        print()
        
        visualizer.run()
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()