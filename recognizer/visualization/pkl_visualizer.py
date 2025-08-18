"""
PKL 기반 시각화 모듈
분석 결과 PKL 파일과 원본 비디오를 이용한 오버레이 생성
"""

import cv2
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class PKLVisualizer:
    """분석 모드 PKL 파일 기반 시각화 클래스 (기존 코드 활용)"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_persons = config.get('models', {}).get('action_classification', {}).get('max_persons', 4)
        
        # 기존 방식 유지 (BaseInferenceVisualizer는 추상 클래스이므로 직접 사용 안함)
        
        # COCO 17 키포인트 연결 구조 (0-based index)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
        
        # 색상 설정
        self.colors = {
            'fight': (0, 0, 255),      # 빨강
            'normal': (0, 255, 0),     # 초록
            'keypoint_face': (255, 255, 0),  # 노란색 (얼굴)
            'keypoint_arm': (0, 255, 0),     # 초록색 (팔)
            'keypoint_leg': (255, 0, 0),     # 파란색 (다리)
            'top_person': (0, 0, 255),       # 빨간색 (max_persons 이내)
            'other_person': (255, 0, 0),     # 파란색 (나머지)
            'text': (255, 255, 255)          # 흰색
        }
        
        # 폰트 크기 기본값
        self.base_font_scale = 1.0
        self.base_thickness = 2
    
    def visualize_single_file(self, video_file: str, results_dir: Path, 
                            save_mode: bool = False, save_dir: str = "overlay_output") -> bool:
        """단일 파일 시각화"""
        video_path = Path(video_file)
        video_name = video_path.stem
        
        # 결과 파일들 찾기
        json_files = list(results_dir.glob(f"**/{video_name}_results.json"))
        frame_pkl_files = list(results_dir.glob(f"**/{video_name}_frame_poses.pkl"))
        rtmo_pkl_files = list(results_dir.glob(f"**/{video_name}_rtmo_poses.pkl"))
        
        if not json_files or not frame_pkl_files:
            logger.error(f"Required result files not found for video: {video_name}")
            return False
        
        # 데이터 로드
        try:
            json_data = self._load_json(json_files[0])
            frame_poses_data = self._load_pkl(frame_pkl_files[0])
            rtmo_poses_data = self._load_pkl(rtmo_pkl_files[0]) if rtmo_pkl_files else []
            
            logger.info(f"Loaded: {len(frame_poses_data)} frame poses, {len(json_data.get('classification_results', []))} classifications")
            
            # 분류 결과 구조 확인 (간단 로그)
            logger.info(f"Classification results: {len(json_data.get('classification_results', []))} windows")
            
        except Exception as e:
            logger.error(f"Failed to load result files: {e}")
            return False
        
        # 시각화 실행
        if save_mode:
            return self._save_overlay_video(video_file, json_data, frame_poses_data, save_dir, video_name)
        else:
            return self._display_realtime_overlay(video_file, json_data, frame_poses_data)
    
    def visualize_folder(self, video_dir: str, results_dir: Path, 
                        save_mode: bool = False, save_dir: str = "overlay_output") -> bool:
        """폴더 시각화"""
        video_path = Path(video_dir)
        if not video_path.exists():
            logger.error(f"Video directory does not exist: {video_dir}")
            return False
        
        # 비디오 파일들 찾기
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(video_files)} video files")
        
        success_count = 0
        for video_file in video_files:
            logger.info(f"Processing: {video_file.name}")
            if self.visualize_single_file(str(video_file), results_dir, save_mode, save_dir):
                success_count += 1
            else:
                logger.warning(f"Failed to visualize: {video_file.name}")
        
        logger.info(f"Visualization complete: {success_count}/{len(video_files)} successful")
        return success_count > 0
    
    def _load_json(self, json_file: Path) -> Dict:
        """JSON 파일 로드"""
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def _load_pkl(self, pkl_file: Path) -> List:
        """PKL 파일 로드"""
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    
    def _display_realtime_overlay(self, video_file: str, json_data: Dict, frame_poses_data: List) -> bool:
        """실시간 오버레이 표시"""
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_file}")
            return False
        
        # 분류 결과를 프레임 인덱스로 매핑 (윈도우 기반)
        classification_by_frame = {}
        for result in json_data.get('classification_results', []):
            # 새로운 윈도우 기반 구조 처리
            start_frame = result.get('window_start_frame', -1)
            end_frame = result.get('window_end_frame', -1)
            
            if start_frame >= 0 and end_frame >= 0:
                # 윈도우 범위의 모든 프레임에 분류 결과 적용
                for frame_idx in range(start_frame, end_frame + 1):
                    classification_by_frame[frame_idx] = result
            else:
                # 이전 구조와의 호환성을 위해 frame_idx도 확인
                frame_idx = result.get('frame_idx', -1)
                if frame_idx >= 0:
                    classification_by_frame[frame_idx] = result
        
        logger.info("Displaying real-time overlay. Press 'q' to quit, 'p' to pause/resume")
        
        frame_idx = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # 현재 프레임 데이터 가져오기
            current_classification = classification_by_frame.get(frame_idx, {})
            current_poses = self._get_frame_poses(frame_poses_data, frame_idx)
            
            # 오버레이 그리기
            display_frame = self._draw_overlay(frame.copy(), current_poses, current_classification, frame_idx)
            
            # 화면에 표시
            cv2.imshow('PKL Overlay Visualization', display_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'}")
            
            if not paused:
                frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time overlay display finished")
        return True
    
    def _save_overlay_video(self, video_file: str, json_data: Dict, frame_poses_data: List, 
                           save_dir: str, video_name: str) -> bool:
        """오버레이 비디오 저장"""
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_file}")
            return False
        
        # 비디오 정보
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 설정
        output_path = Path(save_dir) / f"{video_name}_overlay.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # 분류 결과 매핑 (윈도우 기반)
        classification_by_frame = {}
        for result in json_data.get('classification_results', []):
            # 새로운 윈도우 기반 구조 처리
            start_frame = result.get('window_start_frame', -1)
            end_frame = result.get('window_end_frame', -1)
            
            if start_frame >= 0 and end_frame >= 0:
                # 윈도우 범위의 모든 프레임에 분류 결과 적용
                for frame_idx in range(start_frame, end_frame + 1):
                    classification_by_frame[frame_idx] = result
            else:
                # 이전 구조와의 호환성을 위해 frame_idx도 확인
                frame_idx = result.get('frame_idx', -1)
                if frame_idx >= 0:
                    classification_by_frame[frame_idx] = result
        
        logger.info(f"Video info: {width}x{height}, {fps}FPS, {total_frames} frames")
        logger.info(f"Frame poses mapping: {len(frame_poses_data)} frames")
        logger.info(f"Classification mapping: {len(classification_by_frame)} frames covered")
        
        frame_idx = 0
        progress_interval = max(1, total_frames // 20)  # 5% 간격으로 진행률 표시
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 현재 프레임 데이터
            current_classification = classification_by_frame.get(frame_idx, {})
            current_poses = self._get_frame_poses(frame_poses_data, frame_idx)
            
            # 오버레이 적용
            overlay_frame = self._draw_overlay(frame, current_poses, current_classification, frame_idx)
            out.write(overlay_frame)
            
            # 진행률 표시
            if frame_idx % progress_interval == 0:
                progress = (frame_idx / total_frames) * 100
                logger.info(f"Visualization progress: {progress:.1f}%")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Visualization completed: {output_path}")
        return True
    
    def _get_frame_poses(self, frame_poses_data: List, frame_idx: int):
        """특정 프레임의 포즈 데이터 가져오기"""
        for pose_frame in frame_poses_data:
            if hasattr(pose_frame, 'frame_idx') and pose_frame.frame_idx == frame_idx:
                return pose_frame
        return None
    
    def _get_dynamic_font_params(self, frame_height: int, frame_width: int) -> Tuple[float, int]:
        """비디오 해상도에 따른 동적 폰트 크기 계산"""
        # 기준 해상도: 720p (1280x720)
        base_height = 720
        scale_factor = frame_height / base_height
        
        font_scale = max(0.5, self.base_font_scale * scale_factor)
        thickness = max(1, int(self.base_thickness * scale_factor))
        
        return font_scale, thickness
    
    def _draw_overlay(self, frame: np.ndarray, poses_data, classification_data: Dict, frame_idx: int) -> np.ndarray:
        """프레임에 오버레이 그리기"""
        height, width = frame.shape[:2]
        font_scale, thickness = self._get_dynamic_font_params(height, width)
        
        # 분류 결과 표시
        if classification_data:
            # 새로운 구조에서는 predicted_label 사용
            predicted_class = classification_data.get('predicted_label', 
                             classification_data.get('predicted_class', 'Unknown'))
            confidence = classification_data.get('confidence', 0.0)
            
            color = self.colors['fight'] if predicted_class == 'Fight' else self.colors['normal']
            text = f"{predicted_class} ({confidence:.2f})"
            
            # 동적 폰트 크기 적용
            cv2.putText(frame, text, (int(10 * width/1280), int(30 * height/720)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # 윈도우 정보 표시
            window_id = classification_data.get('window_id', -1)
            frame_range = classification_data.get('frame_range', '')
            if window_id >= 0 and frame_range:
                window_text = f"Window {window_id}: {frame_range}"
                cv2.putText(frame, window_text, (int(10 * width/1280), int(70 * height/720)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, self.colors['text'], thickness)
        
        # 포즈 데이터 표시 (FramePoses 구조 사용)
        if poses_data:
            persons_list = []
            
            # FramePoses 객체인지 확인
            if hasattr(poses_data, 'persons'):
                persons_list = poses_data.persons
            # person_poses 속성이 있는 경우 (이전 구조)
            elif hasattr(poses_data, 'person_poses'):
                persons_list = poses_data.person_poses
            # 딕셔너리 형태인 경우
            elif isinstance(poses_data, dict) and 'persons' in poses_data:
                persons_list = poses_data['persons']
            
            if persons_list:
                # confidence 기준으로 정렬 (내림차순)
                sorted_persons = sorted(persons_list, 
                                      key=lambda p: p.score if hasattr(p, 'score') and p.score else 0.0, 
                                      reverse=True)
                
                for idx, person_pose in enumerate(sorted_persons):
                    # max_persons 이내: 빨간색, 나머지: 파란색
                    is_top_ranked = idx < self.max_persons
                    self._draw_person_pose(frame, person_pose, is_top_ranked, height, width)
        
        # 프레임 번호 표시
        cv2.putText(frame, f"Frame: {frame_idx}", 
                   (int(10 * width/1280), height - int(10 * height/720)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, self.colors['text'], thickness)
        
        return frame
    
    def _draw_person_pose(self, frame: np.ndarray, person_pose, is_top_ranked: bool, height: int, width: int):
        """개별 사람의 포즈 그리기 (기존 PoseVisualizer 스타일)"""
        if not hasattr(person_pose, 'keypoints'):
            logger.warning(f"Person pose missing keypoints attribute: {type(person_pose)}")
            return
        
        # 키포인트 데이터 확인
        keypoints = person_pose.keypoints
        if keypoints is None:
            logger.warning("Keypoints is None")
            return
        
        # 키포인트 데이터 확인 (디버깅 로그 제거)
        
        # 동적 크기 계산
        font_scale, thickness = self._get_dynamic_font_params(height, width)
        keypoint_radius = max(2, int(3 * height / 720))
        bbox_thickness = max(1, int(2 * height / 720))
        skeleton_thickness = max(1, int(2 * height / 720))
        
        # max_persons 이내: 빨간색, 나머지: 파란색
        person_color = self.colors['top_person'] if is_top_ranked else self.colors['other_person']
        
        # 바운딩 박스와 트래킹 ID
        if hasattr(person_pose, 'bbox') and person_pose.bbox:
            x1, y1, x2, y2 = map(int, person_pose.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, bbox_thickness)
            
            # 텍스트 정보
            text_lines = []
            if hasattr(person_pose, 'track_id') and person_pose.track_id is not None:
                text_lines.append(f"ID: {person_pose.track_id}")
            if hasattr(person_pose, 'score') and person_pose.score is not None:
                text_lines.append(f"Conf: {person_pose.score:.2f}")
            
            if text_lines:
                text = " | ".join(text_lines)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, thickness)[0]
                
                # 텍스트 배경
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), 
                             (x1 + text_size[0] + 5, y1), person_color, -1)
                
                # 텍스트
                cv2.putText(frame, text, (x1 + 2, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), thickness)
        
        # 키포인트와 스켈레톤
        self._draw_keypoints_and_skeleton(frame, person_pose.keypoints, person_color, 
                                        keypoint_radius, skeleton_thickness)
    
    def _draw_keypoints_and_skeleton(self, frame: np.ndarray, keypoints, person_color: Tuple[int, int, int],
                                   keypoint_radius: int, skeleton_thickness: int):
        """키포인트와 스켈레톤 그리기 (PoseVisualizer 스타일)"""
        # 키포인트가 numpy array가 아닌 경우 변환
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # 키포인트 그리기 (부위별 색상 구분)
        for i, kpt in enumerate(keypoints):
            if len(kpt) >= 2 and kpt[0] > 0 and kpt[1] > 0:  # 유효한 키포인트만
                # 키포인트별 색상 구분
                if i < 5:  # 얼굴 (코, 눈, 귀)
                    kpt_color = self.colors['keypoint_face']  # 노란색
                elif i < 11:  # 팔 (어깨, 팔꿈치, 손목)
                    kpt_color = self.colors['keypoint_arm']   # 초록색
                else:  # 다리 (엉덩이, 무릎, 발목)
                    kpt_color = self.colors['keypoint_leg']   # 파란색
                
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), keypoint_radius, kpt_color, -1)
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), keypoint_radius + 1, person_color, 1)
        
        # 스켈레톤 연결선 그리기
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx][:2]
                pt2 = keypoints[pt2_idx][:2]
                
                # 두 점이 모두 유효한 경우만 선 그리기
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), person_color, skeleton_thickness)


def create_pkl_visualization(video_file: str, results_dir: str, save_mode: bool = False, 
                           save_dir: str = "overlay_output", config: Dict = None) -> bool:
    """PKL 기반 시각화 생성 (편의 함수)"""
    if config is None:
        config = {}
    
    visualizer = PKLVisualizer(config)
    return visualizer.visualize_single_file(video_file, Path(results_dir), save_mode, save_dir)