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
    
    def __init__(self, config: Dict, target_service: str = None):
        self.config = config
        self.max_persons = config.get('models', {}).get('action_classification', {}).get('max_persons', 4)

        # 시각화 대상 서비스 설정
        self.target_service = target_service

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

        # 서비스별 최신 추론값 저장 (지속적 표시용)
        self.latest_scores = {
            'fight': {'confidence': 0.0, 'label': 'NonFight', 'is_event': False},
            'falldown': {'confidence': 0.0, 'label': 'Normal', 'is_event': False}
        }

        # 활성 서비스 목록
        self.active_services = set()

        # 모든 서비스 결과 (프레임별)
        self.all_service_results = {}
    
    def visualize_single_file(self, video_file: str, results_dir: Path,
                            save_mode: bool = False, save_dir: str = "overlay_output", service_name: str = None) -> bool:
        """단일 파일 시각화 (서비스별 파일 지원)"""
        video_path = Path(video_file)
        video_name = video_path.stem

        # 모든 서비스 파일 수집 (inference 점수 표시를 위해)
        self.all_service_results = self._collect_all_service_results(results_dir, video_name)

        # 서비스별 파일 찾기 (기존 로직)
        if service_name:
            # 서비스명이 지정된 경우 해당 서비스의 파일만 찾기
            json_files = list(results_dir.glob(f"**/{video_name}_{service_name}_results.json"))
            frame_pkl_files = list(results_dir.glob(f"**/{video_name}_{service_name}_frame_poses.pkl"))
            rtmo_pkl_files = list(results_dir.glob(f"**/{video_name}_{service_name}_rtmo_poses.pkl"))

            if not json_files or not frame_pkl_files:
                logger.warning(f"No {service_name} service files found, trying without service name")
                # 서비스명이 없는 파일도 시도
                json_files = list(results_dir.glob(f"**/{video_name}_results.json"))
                frame_pkl_files = list(results_dir.glob(f"**/{video_name}_frame_poses.pkl"))
                rtmo_pkl_files = list(results_dir.glob(f"**/{video_name}_rtmo_poses.pkl"))
        else:
            # 서비스명이 없는 경우 모든 가능한 파일 찾기
            json_files = list(results_dir.glob(f"**/{video_name}_results.json"))
            json_files.extend(list(results_dir.glob(f"**/{video_name}_*_results.json")))

            frame_pkl_files = list(results_dir.glob(f"**/{video_name}_frame_poses.pkl"))
            frame_pkl_files.extend(list(results_dir.glob(f"**/{video_name}_*_frame_poses.pkl")))

            rtmo_pkl_files = list(results_dir.glob(f"**/{video_name}_rtmo_poses.pkl"))
            rtmo_pkl_files.extend(list(results_dir.glob(f"**/{video_name}_*_rtmo_poses.pkl")))

            # 중복 제거
            json_files = list(set(json_files))
            frame_pkl_files = list(set(frame_pkl_files))
            rtmo_pkl_files = list(set(rtmo_pkl_files))
        
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
                        save_mode: bool = False, save_dir: str = "overlay_output", service_name: str = None) -> bool:
        """폴더 시각화 (서비스별 파일 지원)"""
        video_path = Path(video_dir)
        if not video_path.exists():
            logger.error(f"Video directory does not exist: {video_dir}")
            return False

        # 비디오 파일들 찾기
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_path.glob(f"**/*{ext}"))

        service_info = f" for {service_name} service" if service_name else ""
        logger.info(f"Found {len(video_files)} video files{service_info}")

        success_count = 0
        for video_file in video_files:
            logger.info(f"Processing: {video_file.name}")
            if self.visualize_single_file(str(video_file), results_dir, save_mode, save_dir, service_name):
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
        """PKL 파일 로드 (통일된 데이터 구조 지원)"""
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # VisualizationData 객체인 경우 poses_with_tracking 추출
        if hasattr(data, 'poses_with_tracking') and data.poses_with_tracking:
            return data.poses_with_tracking
        elif hasattr(data, 'frame_data') and data.frame_data:
            return data.frame_data
        else:
            # 기존 리스트 형태 데이터 그대로 반환
            return data
    
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

        # 모든 서비스 결과에서 현재 프레임의 데이터 가져오기
        current_frame_results = self.all_service_results.get(frame_idx, {})

        # 모든 서비스 점수 업데이트
        for service in ['fight', 'falldown']:
            if service in current_frame_results:
                service_data = current_frame_results[service]
                self.latest_scores[service]['confidence'] = service_data['confidence']
                self.latest_scores[service]['is_event'] = service_data['is_event']
                self.latest_scores[service]['label'] = (
                    'Fight' if service == 'fight' and service_data['is_event'] else
                    'Falldown' if service == 'falldown' and service_data['is_event'] else
                    'NonFight' if service == 'fight' else 'Normal'
                )

        # 분류 결과 처리 및 최신값 업데이트 (기존 로직 유지)
        if classification_data:
            # 서비스별 분류 결과 처리
            if 'services' in classification_data:
                # 듀얼 서비스 구조
                for service_name, service_result in classification_data['services'].items():
                    predicted_class = service_result.get('predicted_class', 'Unknown')
                    confidence = service_result.get('confidence', 0.0)

                    self.active_services.add(service_name)

                    # 서비스별 라벨 매핑 및 최신값 업데이트
                    if service_name == 'fight':
                        is_event = predicted_class == 'Fight' or predicted_class == 1
                        self.latest_scores['fight']['confidence'] = confidence
                        self.latest_scores['fight']['label'] = 'Fight' if is_event else 'NonFight'
                        self.latest_scores['fight']['is_event'] = is_event

                    elif service_name == 'falldown':
                        is_event = predicted_class == 'Falldown' or predicted_class == 1
                        self.latest_scores['falldown']['confidence'] = confidence
                        self.latest_scores['falldown']['label'] = 'Falldown' if is_event else 'Normal'
                        self.latest_scores['falldown']['is_event'] = is_event

            else:
                # 단일 서비스 구조 (Fight만)
                predicted_class = classification_data.get('predicted_label',
                                 classification_data.get('predicted_class', 'Unknown'))
                confidence = classification_data.get('confidence', 0.0)

                self.active_services.add('fight')

                # Fight 서비스 업데이트
                is_fight = predicted_class == 1 or predicted_class == 'Fight'
                self.latest_scores['fight']['confidence'] = confidence
                self.latest_scores['fight']['label'] = 'Fight' if is_fight else 'NonFight'
                self.latest_scores['fight']['is_event'] = is_fight

        # 새로운 오버레이 표시 형태
        y_start = int(30 * height/720)
        line_height = int(30 * height/720)
        x_pos = int(10 * width/1280)

        # 오버레이 텍스트들 준비 (target_service 기준)
        # 1. Service 라인 - target_service에 따라 표시
        if self.target_service:
            service_text = f"Service: {self.target_service.capitalize()}"
        elif len(self.active_services) >= 2:
            service_text = "Service: Both"
        elif 'fight' in self.active_services:
            service_text = "Service: Fight"
        elif 'falldown' in self.active_services:
            service_text = "Service: Falldown"
        else:
            service_text = "Service: None"

        # 2. Event 라인 - target_service에 따라 해당 서비스만 표시
        y_event = y_start + line_height
        event_parts = []

        if self.target_service == 'fight':
            # Fight 서비스만 표시
            if self.latest_scores['fight']['is_event']:
                event_parts.append("Fight")
        elif self.target_service == 'falldown':
            # Falldown 서비스만 표시
            if self.latest_scores['falldown']['is_event']:
                event_parts.append("Falldown")
        else:
            # 전체 서비스 표시
            if self.latest_scores['fight']['is_event']:
                event_parts.append("Fight")
            if self.latest_scores['falldown']['is_event']:
                event_parts.append("Falldown")

        if event_parts:
            event_text = f"Event: {' | '.join(event_parts)}"
            event_color = (0, 0, 255)  # 빨간색
        else:
            event_text = "Event: Normal"  # None 대신 Normal
            event_color = self.colors['text']  # 흰색

        # 3. Inference 라인 - target_service에 따라 해당 서비스만 표시
        y_inference = y_event + line_height
        if self.target_service == 'fight':
            fight_score = self.latest_scores['fight']['confidence']
            inference_text = f"Inference: Fight({fight_score:.2f})"
        elif self.target_service == 'falldown':
            falldown_score = self.latest_scores['falldown']['confidence']
            inference_text = f"Inference: Falldown({falldown_score:.2f})"
        else:
            # 전체 서비스 표시
            fight_score = self.latest_scores['fight']['confidence']
            falldown_score = self.latest_scores['falldown']['confidence']
            inference_text = f"Inference: Fight({fight_score:.2f}) | Falldown({falldown_score:.2f})"

        # 텍스트 크기 계산
        texts = [service_text, event_text, inference_text]
        max_width = 0
        text_height = 0

        for text in texts:
            (text_width, text_height_single), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness)
            max_width = max(max_width, text_width)
            text_height = text_height_single

        # 배경 사각형 그리기 (검은색)
        padding = int(10 * width/1280)
        bg_x1 = x_pos - padding
        bg_y1 = y_start - text_height - padding
        bg_x2 = x_pos + max_width + padding
        bg_y2 = y_inference + padding

        # 검은색 배경 사각형
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        # 텍스트 표시
        cv2.putText(frame, service_text, (x_pos, y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, self.colors['text'], thickness)

        cv2.putText(frame, event_text, (x_pos, y_event),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, event_color, thickness)

        cv2.putText(frame, inference_text, (x_pos, y_inference),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, self.colors['text'], thickness)

        # 윈도우 정보 표시
        if classification_data:
            window_id = classification_data.get('window_id', -1)
            frame_range = classification_data.get('frame_range', '')
            if window_id >= 0 and frame_range:
                y_window = y_inference + int(30 * height/720)
                window_text = f"Window {window_id}: {frame_range}"
                cv2.putText(frame, window_text, (int(10 * width/1280), y_window),
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

    def _collect_all_service_results(self, results_dir: Path, video_name: str) -> Dict:
        """모든 서비스의 결과를 수집해서 프레임별로 정리"""
        all_results = {}

        # 가능한 서비스들
        services = ['fight', 'falldown']

        for service in services:
            service_files = list(results_dir.glob(f"**/{video_name}_{service}_results.json"))
            if service_files:
                try:
                    with open(service_files[0], 'r') as f:
                        data = json.load(f)

                    if 'classification_results' in data:
                        for result in data['classification_results']:
                            window_start = result.get('window_start_frame', 0)
                            window_end = result.get('window_end_frame', 100)
                            confidence = result.get('confidence', 0.0)
                            predicted_class = result.get('predicted_label', 0)

                            # 각 윈도우의 모든 프레임에 대해 결과 저장
                            for frame_idx in range(window_start, min(window_end + 1, window_start + 100)):
                                if frame_idx not in all_results:
                                    all_results[frame_idx] = {}
                                all_results[frame_idx][service] = {
                                    'confidence': confidence,
                                    'predicted_class': predicted_class,
                                    'is_event': predicted_class == 1
                                }

                except Exception as e:
                    logger.warning(f"Error reading {service} results: {e}")

        return all_results

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
                           save_dir: str = "overlay_output", config: Dict = None, service_name: str = None) -> bool:
    """PKL 기반 시각화 생성 (편의 함수, 서비스별 지원)"""
    if config is None:
        config = {}

    visualizer = PKLVisualizer(config)
    return visualizer.visualize_single_file(video_file, Path(results_dir), save_mode, save_dir, service_name)