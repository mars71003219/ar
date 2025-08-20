"""
실시간 시각화 모듈
실시간 비디오 스트림에서 추론 결과를 즉시 표시하는 기능
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from queue import Queue, Empty
import threading

try:
    from ..utils.import_utils import safe_import_pose_structures, setup_logger
except ImportError:
    try:
        from utils.import_utils import safe_import_pose_structures, setup_logger
    except ImportError:
        def safe_import_pose_structures():
            try:
                from utils.data_structure import PersonPose, FramePoses
                return PersonPose, FramePoses
            except ImportError:
                from ..utils.data_structure import PersonPose, FramePoses
                return PersonPose, FramePoses
        
        def setup_logger(name):
            import logging
            return logging.getLogger(name)

try:
    from .pose_visualizer import PoseVisualizer
except ImportError:
    from visualization.pose_visualizer import PoseVisualizer

PersonPose, FramePoses = safe_import_pose_structures()

logger = setup_logger(__name__)


class RealtimeVisualizer:
    """실시간 시각화 클래스"""
    
    def __init__(self, 
                 window_name: str = "Violence Detection",
                 display_width: int = 1280,
                 display_height: int = 720,
                 fps_limit: int = 30,
                 save_output: bool = False,
                 output_path: Optional[str] = None,
                 max_persons: int = 4,
                 processing_mode: str = "realtime",
                 confidence_threshold: float = 0.4):
        """
        모드별 시각화 초기화
        
        Args:
            window_name: OpenCV 창 이름
            display_width: 표시 창 너비
            display_height: 표시 창 높이
            fps_limit: 표시 FPS 제한
            save_output: 결과를 비디오로 저장할지 여부
            output_path: 저장할 비디오 파일 경로
            max_persons: 최대 인원 수
            processing_mode: 처리 모드 ('realtime' 또는 'analysis')
            confidence_threshold: 분류 결과 표시 임계값
        """
        self.window_name = window_name
        self.display_width = display_width
        self.display_height = display_height
        self.fps_limit = fps_limit
        self.save_output = save_output
        self.output_path = output_path
        self.max_persons = max_persons
        self.confidence_threshold = confidence_threshold
        
        # 처리 모드 설정
        self.processing_mode = processing_mode
        self.display_mode = processing_mode  # 동일하게 설정
        
        # 시각화 도구
        self.pose_visualizer = PoseVisualizer(max_persons=max_persons)
        
        # 상태 관리
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # 비디오 저장 설정
        self.video_writer = None
        if save_output and output_path:
            self.setup_video_writer()
        
        # 통계
        self.fps_history = []
        self.processing_times = []
        
        # 분류 결과 히스토리
        self.classification_history = []  
        
        # 모드별 오버레이 설정
        if self.processing_mode == 'realtime':
            self.realtime_overlay_enabled = True
            self.show_composite_scores = False  # 실시간에서는 비활성화
        else:  # analysis
            self.realtime_overlay_enabled = False
            self.show_composite_scores = True   # 분석에서는 활성화
        
        # 이벤트 히스토리 표시
        self.event_history = []
        self.max_event_history = 10  # 최대 표시할 이벤트 수
        
        # 스케일링 정보 초기화
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.original_size = (display_width, display_height)
        
        logger.info(f"Visualizer initialized: mode={processing_mode}, {display_width}x{display_height}, FPS={fps_limit}")
    
    def setup_video_writer(self):
        """비디오 저장을 위한 VideoWriter 설정"""
        if self.output_path:
            # 출력 디렉토리 생성
            from pathlib import Path
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps_limit,
                (self.display_width, self.display_height)
            )
            
            # VideoWriter 초기화 확인
            if self.video_writer.isOpened():
                logger.info(f"Video writer setup successfully: {self.output_path}")
            else:
                logger.error(f"Failed to initialize video writer: {self.output_path}")
                self.video_writer = None
    
    def start_display(self):
        """디스플레이 창 시작"""
        self.is_running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL로 변경하여 크기 조절 가능
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        logger.info(f"Display window started: {self.window_name} ({self.display_width}x{self.display_height})")
    
    def stop_display(self):
        """디스플레이 창 종료"""
        self.is_running = False
        cv2.destroyWindow(self.window_name)
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"Video saved: {self.output_path}")
        logger.info("Display window stopped")
    
    def show_frame(self, 
                   frame: np.ndarray,
                   poses: Optional[FramePoses] = None,
                   classification: Optional[Dict[str, Any]] = None,
                   additional_info: Optional[Dict[str, Any]] = None,
                   overlay_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        모드별 프레임 표시
        
        Args:
            frame: 원본 프레임
            poses: 포즈 데이터
            classification: 분류 결과 (분석 모드용, 실시간에서는 None)
            additional_info: 추가 정보 (FPS, 처리 시간 등)
            overlay_data: window_processor에서 제공하는 오버레이 데이터 (실시간 모드용)
        
        Returns:
            계속 표시할지 여부 (False면 종료)
        """
        if not self.is_running:
            return False
        
        process_start = time.time()
        
        # FPS 제한 확인
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        target_interval = 1.0 / self.fps_limit
        
        if time_since_last < target_interval:
            time.sleep(target_interval - time_since_last)
            current_time = time.time()
        
        # 프레임 리사이즈
        display_frame = self.resize_frame(frame)
        
        # 포즈 좌표 스케일링
        scaled_poses = self.scale_poses_to_display(poses) if poses else None
        
        # 포즈 시각화
        if scaled_poses and scaled_poses.persons:
            display_frame = self.pose_visualizer.visualize_frame(display_frame, scaled_poses)
        
        # 분류 결과와 오버레이 표시 (원래 방식)
        vis_frame = self.draw_classification_results(display_frame)
        vis_frame = self.add_overlay_info(vis_frame, additional_info)
        
        # 이벤트 상태 표시 (항상)
        vis_frame = self._draw_event_history(vis_frame)
        
        # 화면에 표시
        cv2.imshow(self.window_name, vis_frame)
        
        # 비디오 저장
        if self.video_writer:
            self.video_writer.write(vis_frame)
        
        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' 또는 ESC
            return False
        elif key == ord('s'):  # 스크린샷 저장
            self.save_screenshot(vis_frame)
        elif key == ord('p'):  # 일시정지/재시작
            self.toggle_pause()
        elif key == ord('m'):  # 모드 전환
            self.toggle_mode()
        
        # 통계 업데이트
        self.update_statistics(current_time, time.time() - process_start)
        self.last_frame_time = current_time
        self.frame_count += 1
        
        return True
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임을 디스플레이 크기에 맞게 리사이즈"""
        height, width = frame.shape[:2]
        
        # 스케일 정보 저장 (포즈 좌표 변환용)
        self.original_size = (width, height)
        
        # 설정된 디스플레이 크기와 동일한 경우 그대로 반환
        if width == self.display_width and height == self.display_height:
            self.scale_factor = 1.0
            self.offset_x = 0
            self.offset_y = 0
            return frame
        
        # 종횡비 유지하면서 리사이즈
        scale = min(self.display_width / width, self.display_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        self.scale_factor = scale
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # 설정된 디스플레이 크기에 맞춰 패딩 또는 크롭
        if new_width != self.display_width or new_height != self.display_height:
            # 중앙 정렬을 위한 패딩
            top = max(0, (self.display_height - new_height) // 2)
            bottom = max(0, self.display_height - new_height - top)
            left = max(0, (self.display_width - new_width) // 2)
            right = max(0, self.display_width - new_width - left)
            
            self.offset_x = left
            self.offset_y = top
            
            if top + bottom + left + right > 0:
                resized = cv2.copyMakeBorder(
                    resized, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            
            # 최종적으로 정확한 크기로 조정
            if resized.shape[:2] != (self.display_height, self.display_width):
                resized = cv2.resize(resized, (self.display_width, self.display_height))
        else:
            self.offset_x = 0
            self.offset_y = 0
        
        return resized
    
    def scale_poses_to_display(self, poses: FramePoses) -> FramePoses:
        """포즈 좌표를 디스플레이 크기에 맞게 스케일링"""
        if not poses or not poses.persons:
            return poses
        
        # 스케일 정보가 없으면 그대로 반환
        if not hasattr(self, 'scale_factor'):
            return poses
        
        # 깊은 복사로 원본 데이터 보호
        import copy
        scaled_poses = copy.deepcopy(poses)
        
        for person in scaled_poses.persons:
            # bbox 스케일링
            if person.bbox and len(person.bbox) == 4:
                x1, y1, x2, y2 = person.bbox
                person.bbox = [
                    x1 * self.scale_factor + self.offset_x,
                    y1 * self.scale_factor + self.offset_y,
                    x2 * self.scale_factor + self.offset_x,
                    y2 * self.scale_factor + self.offset_y
                ]
            
            # keypoints 스케일링
            if person.keypoints is not None:
                if hasattr(person.keypoints, 'shape') and len(person.keypoints.shape) == 2:
                    # numpy array 형태 (N, 2) 또는 (N, 3)
                    scaled_keypoints = person.keypoints.copy()
                    scaled_keypoints[:, 0] = scaled_keypoints[:, 0] * self.scale_factor + self.offset_x
                    scaled_keypoints[:, 1] = scaled_keypoints[:, 1] * self.scale_factor + self.offset_y
                    person.keypoints = scaled_keypoints
                elif hasattr(person.keypoints, '__len__'):
                    # 리스트 형태
                    if len(person.keypoints) >= 34:  # 17 keypoints * 2 coordinates
                        scaled_keypoints = list(person.keypoints)
                        for i in range(0, len(scaled_keypoints), 2):
                            if i + 1 < len(scaled_keypoints):
                                scaled_keypoints[i] = scaled_keypoints[i] * self.scale_factor + self.offset_x      # x
                                scaled_keypoints[i + 1] = scaled_keypoints[i + 1] * self.scale_factor + self.offset_y  # y
                        person.keypoints = scaled_keypoints
        
        return scaled_poses
    
    def apply_realtime_visualization(self, 
                                   frame: np.ndarray,
                                   poses: Optional[FramePoses],
                                   overlay_data: Optional[Dict[str, Any]],
                                   additional_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """실시간 모드 시각화 적용
        
        - 항상 관절점과 트래킹 ID 표시
        - classification_delay 후 분류 결과 표시
        - 복합점수 비표시
        - Previous/Current 윈도우 패턴
        """
        vis_frame = frame.copy()
        
        # 포즈 좌표 스케일링
        scaled_poses = self.scale_poses_to_display(poses) if poses else None
        
        # overlay_data가 없으면 기본 포즈만 표시
        if not overlay_data:
            if scaled_poses and scaled_poses.persons:
                vis_frame = self.draw_basic_poses(vis_frame, scaled_poses)
            return vis_frame
        
        # 1. 관절점 및 트래킹 ID 표시 (항상 실시)
        if overlay_data.get('show_keypoints', True) and scaled_poses and scaled_poses.persons:
            vis_frame = self.draw_keypoints_and_tracking(vis_frame, scaled_poses, overlay_data.get('show_tracking_ids', True))
        
        # 2. 분류 결과 표시 (조건부)
        if overlay_data.get('show_classification', False) and overlay_data.get('window_results', []):
            vis_frame = self.draw_realtime_classification(vis_frame, overlay_data['window_results'])
        
        # 3. 추가 정보 (프레임 번호, FPS 등)
        if additional_info:
            vis_frame = self.add_realtime_info_overlay(vis_frame, additional_info, overlay_data)
        
        # 4. 이벤트 상태 표시 (항상)
        vis_frame = self._draw_event_history(vis_frame)
            
        return vis_frame
    
    def apply_analysis_visualization(self, 
                                   frame: np.ndarray,
                                   poses: Optional[FramePoses],
                                   classification: Optional[Dict[str, Any]],
                                   additional_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """분석 모드 시각화 적용 (pkl 기반)
        
        - 복합점수 표시
        - 상세 분류 결과
        - 기존 방식 유지
        """
        vis_frame = frame.copy()
        
        # 포즈 좌표 스케일링
        scaled_poses = self.scale_poses_to_display(poses) if poses else None
        
        # 1. 기본 포즈 표시
        if scaled_poses and scaled_poses.persons:
            vis_frame = self.draw_analysis_poses(vis_frame, scaled_poses)
        
        # 2. 복합점수 표시 (분석 모드에서만)
        if self.show_composite_scores and scaled_poses and scaled_poses.persons:
            vis_frame = self.draw_composite_scores(vis_frame, scaled_poses)
        
        # 3. 분류 결과 표시
        if classification:
            vis_frame = self.draw_analysis_classification(vis_frame, classification)
        
        # 4. 추가 정보
        if additional_info:
            vis_frame = self.add_analysis_info_overlay(vis_frame, additional_info)
            
        return vis_frame
    
    def get_display_mode(self) -> str:
        """현재 표시 모드 반환"""
        return self.display_mode
    
    def set_realtime_overlay_enabled(self, enabled: bool):
        """실시간 오버레이 활성화/비활성화"""
        self.realtime_overlay_enabled = enabled
        logger.info(f"Realtime overlay {'enabled' if enabled else 'disabled'}")
    
    def is_realtime_overlay_enabled(self) -> bool:
        """실시간 오버레이 활성화 상태 확인"""
        return self.realtime_overlay_enabled
    
    def update_event_history(self, event_data: Dict[str, Any]):
        """이벤트 히스토리 업데이트"""
        if not isinstance(event_data, dict):
            return
        
        # 디버깅 로그 추가
        event_type = event_data.get('event_type', 'unknown')
        logging.info(f"[EVENT UPDATE] Received event: {event_type}")
        
        # 타임스탬프 기반으로 정렬하여 삽입
        self.event_history.append({
            'event_type': event_type,
            'timestamp': event_data.get('timestamp', time.time()),
            'window_id': event_data.get('window_id', 0),
            'confidence': event_data.get('confidence', 0.0),
            'duration': event_data.get('duration'),
            'frame_number': self.frame_count
        })
        
        # 최대 개수 제한
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        logging.info(f"[EVENT UPDATE] Event history updated: {event_type}, total events: {len(self.event_history)}")

    def update_classification_history(self, classification: Dict[str, Any]):
        """분류 결과 히스토리 업데이트 - 파이프라인에서 전달받은 윈도우 번호 사용"""
        if not isinstance(classification, dict):
            logging.warning(f"Invalid classification data type: {type(classification)}")
            return
            
        # 파이프라인에서 전달받은 윈도우 번호
        display_id = classification.get('display_id', 0)
        
        # display_id 유효성 검사
        if not isinstance(display_id, (int, float)):
            logging.warning(f"Invalid display_id type: {type(display_id)}, converting to 0")
            display_id = 0
        
        # probabilities 변환 (List[float] -> Dict[str, float])
        raw_probabilities = classification.get('probabilities', [0.0, 0.0])
        if isinstance(raw_probabilities, list) and len(raw_probabilities) >= 2:
            probabilities_dict = {
                'NonFight': raw_probabilities[0],
                'Fight': raw_probabilities[1]
            }
        else:
            probabilities_dict = {'NonFight': 0.0, 'Fight': 0.0}
        
        # 새 윈도우 정보 생성
        new_window_info = {
            'display_id': display_id,
            'predicted_class': classification.get('predicted_class', 'Unknown'),
            'confidence': classification.get('confidence', 0.0),
            'probabilities': probabilities_dict,
            'window_start': classification.get('window_start', 0),
            'window_end': classification.get('window_end', 0),
            'update_frame': self.frame_count
        }
        
        logging.info(f"[VISUALIZER] Updating classification history: Window {display_id}, {new_window_info['predicted_class']} ({new_window_info['confidence']:.3f})")
        
        # 이미 같은 윈도우 번호가 있는지 확인 (중복 방지)
        if self.classification_history:
            last_window = self.classification_history[-1]
            if last_window['display_id'] == display_id:
                # 같은 윈도우 번호면 업데이트하지 않음
                logging.info(f"[VISUALIZER] Skipping duplicate window {display_id}")
                return
        
        # 새 윈도우 추가
        self.classification_history.append(new_window_info)
        
        # 히스토리 크기 제한 (최대 5개 윈도우만 유지)
        if len(self.classification_history) > 5:
            self.classification_history = self.classification_history[-5:]
        
        logging.info(f"[VISUALIZER] Classification history updated. Total windows: {len(self.classification_history)}")
        
        # 현재 히스토리 상태 출력
        for i, window in enumerate(self.classification_history):
            logging.info(f"[VISUALIZER] History[{i}]: Window {window['display_id']}, {window['predicted_class']} ({window['confidence']:.3f})")
        
        # 최대 2개까지만 유지 (Previous, Current)
        if len(self.classification_history) > 2:
            self.classification_history.pop(0)
    
    def draw_classification_results(self, frame: np.ndarray) -> np.ndarray:
        """좌측 상단에 윈도우 분류 결과와 확률 표시"""
        height, width = frame.shape[:2]
        
        # 해상도에 따른 동적 폰트 크기 조정
        base_scale = min(width, height) / 1000.0  # 1000px 기준으로 정규화
        font_scale = max(0.4, base_scale * 0.5)  # 기존보다 작게, 최소 0.4
        thickness = max(1, int(base_scale * 2))
        line_height = max(25, int(base_scale * 30))
        
        # 해상도에 따른 시작 위치 조정
        start_x = max(10, int(width * 0.02))
        start_y = max(20, int(height * 0.05))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        logging.info(f"[VISUALIZER] Drawing classification results. History length: {len(self.classification_history)}")
        
        if not self.classification_history:
            # 분류 결과가 없는 경우
            no_data_text = "Waiting for classification..."
            text_size = cv2.getTextSize(no_data_text, font, font_scale, thickness)[0]
            
            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y - 25), 
                         (start_x + text_size[0] + 10, start_y + 10), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            cv2.putText(frame, no_data_text, (start_x, start_y), 
                       font, font_scale, (255, 255, 255), thickness)
            return frame
        
        y_offset = 0
        
        # 최근 윈도우들을 시간순으로 표시 (가장 최신이 current)
        for i, window_info in enumerate(self.classification_history):
            display_id = window_info['display_id']
            pred_class = window_info['predicted_class']
            confidence = window_info['confidence']
            probabilities = window_info.get('probabilities', {})
            
            # 상태 결정 (최신 윈도우가 current)
            if i == len(self.classification_history) - 1:
                status = "current"
            else:
                status = "previous"
            
            # 윈도우 번호와 결과 표시 - 사용자가 원하는 형식에 맞춤
            # window 1 (current) : Fight(0.921) | NonFight(0.079)
            fight_prob = probabilities.get('Fight', confidence if pred_class == 'Fight' else 1.0 - confidence)
            nonfight_prob = probabilities.get('NonFight', confidence if pred_class == 'NonFight' else 1.0 - confidence)
            
            window_text = f"window {display_id} ({status}) : Fight({fight_prob:.3f}) | NonFight({nonfight_prob:.3f})"
            
            # 색상 결정 - confidence_threshold 고려
            # config에서 설정된 confidence_threshold 가져오기 (기본값 0.4)
            confidence_threshold = getattr(self, 'confidence_threshold', 0.4)
            
            # threshold 기반 색상 결정
            if fight_prob >= confidence_threshold and fight_prob > nonfight_prob:
                bg_color = (0, 0, 200)  # 빨간색 - Fight가 임계값 이상이고 더 높을 때
            elif nonfight_prob >= confidence_threshold and nonfight_prob > fight_prob:
                bg_color = (0, 200, 0)  # 초록색 - NonFight가 임계값 이상이고 더 높을 때
            else:
                bg_color = (128, 128, 128)  # 회색 - 불확실한 경우 (임계값 미달)
            
            text_color = (255, 255, 255)
            
            # 배경 박스와 텍스트
            text_size = cv2.getTextSize(window_text, font, font_scale, thickness)[0]
            
            # 반투명 배경 박스
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y + y_offset - 25), 
                         (start_x + text_size[0] + 10, start_y + y_offset + 8), bg_color, -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # 텍스트 표시
            cv2.putText(frame, window_text, (start_x, start_y + y_offset), 
                       font, font_scale, text_color, thickness)
            
            y_offset += line_height
        
        # FPS 정보 추가 (윈도우 아래에)
        if hasattr(self, '_last_fps_time') and hasattr(self, '_frame_count_for_fps'):
            current_time = time.time()
            if current_time - self._last_fps_time > 0:
                fps = self._frame_count_for_fps / (current_time - self._last_fps_time)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (start_x, start_y + y_offset), 
                           font, font_scale, (255, 255, 255), thickness)
        else:
            # 초기화
            self._last_fps_time = time.time()
            self._frame_count_for_fps = 0
        
        # 확률 그래프 삭제 - 의미없음 (Fight + NonFight = 1.0이라 반대 곡선만 나옴)
        # if len(self.classification_history) > 0:
        #     y_offset += 10
        #     self.draw_probability_graph(frame, start_x, start_y + y_offset)
        
        return frame
    
    def draw_probability_graph(self, frame: np.ndarray, start_x: int, start_y: int):
        """확률 그래프 그리기"""
        if not self.classification_history:
            return
        
        # 그래프 설정
        graph_width = 250
        graph_height = 60
        
        # 그래프 배경
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + graph_width, start_y + graph_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (start_x, start_y), 
                     (start_x + graph_width, start_y + graph_height), (255, 255, 255), 1)
        
        # 제목
        cv2.putText(frame, "Probability History", (start_x + 5, start_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 데이터 가져오기
        history_len = min(len(self.classification_history), 10)
        if history_len < 2:
            return
        
        recent_history = self.classification_history[-history_len:]
        
        # 그래프 그리기
        step_x = graph_width / (history_len - 1) if history_len > 1 else graph_width
        
        fight_points = []
        nonfight_points = []
        
        for i, window_info in enumerate(recent_history):
            probabilities = window_info.get('probabilities', {})
            fight_prob = probabilities.get('Fight', 0.0)
            nonfight_prob = probabilities.get('NonFight', 0.0)
            
            x = start_x + int(i * step_x)
            fight_y = start_y + graph_height - int(fight_prob * graph_height * 0.8)
            nonfight_y = start_y + graph_height - int(nonfight_prob * graph_height * 0.8)
            
            fight_points.append((x, fight_y))
            nonfight_points.append((x, nonfight_y))
        
        # Fight 확률 선 그리기 (빨간색)
        for i in range(len(fight_points) - 1):
            cv2.line(frame, fight_points[i], fight_points[i + 1], (0, 0, 255), 2)
        
        # NonFight 확률 선 그리기 (초록색)
        for i in range(len(nonfight_points) - 1):
            cv2.line(frame, nonfight_points[i], nonfight_points[i + 1], (0, 255, 0), 2)
        
        # 점 그리기
        for point in fight_points:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)
        for point in nonfight_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # 범례
        legend_y = start_y + graph_height + 15
        cv2.putText(frame, "Fight", (start_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(frame, "NonFight", (start_x + 50, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    def add_overlay_info(self, frame: np.ndarray, additional_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """오버레이 정보 추가 - 순수 처리 FPS 및 단계별 FPS 표시"""
        if additional_info is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # 해상도에 따른 동적 폰트 크기 조정
        base_scale = min(width, height) / 1000.0
        font_scale = max(0.4, base_scale * 0.5)
        thickness = max(1, int(base_scale * 2))
        
        # 오버레이 정보 표시 영역 (좌측 상단)
        overlay_x = 10
        overlay_y = 30
        line_height = int(25 * font_scale)
        
        # 배경 박스 (단계별 FPS 포함하여 높이 증가)
        box_width = 320
        box_height = line_height * 8 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (overlay_x - 5, overlay_y - 20), 
                     (overlay_x + box_width, overlay_y + box_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 순수 처리 FPS (시각화 제외)
        processing_fps = additional_info.get('processing_fps', 0)
        cv2.putText(frame, f"Processing FPS: {processing_fps:.1f}", 
                   (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 255, 0), thickness)
        
        # 처리 시간
        processing_time = additional_info.get('pure_processing_time', 0)
        cv2.putText(frame, f"Processing Time: {processing_time*1000:.1f}ms", 
                   (overlay_x, overlay_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 0), thickness)
        
        # 단계별 FPS 정보 표시
        pose_fps = additional_info.get('pose_estimation_fps', 0)
        cv2.putText(frame, f"Pose FPS: {pose_fps:.1f}", 
                   (overlay_x, overlay_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 100, 100), thickness)
        
        tracking_fps = additional_info.get('tracking_fps', 0)
        cv2.putText(frame, f"Track FPS: {tracking_fps:.1f}", 
                   (overlay_x, overlay_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (100, 255, 100), thickness)
        
        scoring_fps = additional_info.get('scoring_fps', 0)
        cv2.putText(frame, f"Score FPS: {scoring_fps:.1f}", 
                   (overlay_x, overlay_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (100, 100, 255), thickness)
        
        classification_fps = additional_info.get('classification_fps', 0)
        cv2.putText(frame, f"Class FPS: {classification_fps:.1f}", 
                   (overlay_x, overlay_y + line_height * 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 100), thickness)
        
        # 검출된 사람 수
        total_persons = additional_info.get('total_persons', 0)
        cv2.putText(frame, f"Persons: {total_persons}", 
                   (overlay_x, overlay_y + line_height * 6), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)
        
        # 버퍼 크기
        buffer_size = additional_info.get('buffer_size', 0)
        cv2.putText(frame, f"Buffer: {buffer_size}", 
                   (overlay_x, overlay_y + line_height * 7), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def calculate_current_fps(self) -> float:
        """현재 FPS 계산"""
        if len(self.fps_history) > 0:
            return np.mean(self.fps_history[-10:])  # 최근 10프레임 평균
        return 0.0
    
    def update_statistics(self, frame_time: float, processing_time: float):
        """통계 업데이트"""
        if len(self.fps_history) > 0:
            fps = 1.0 / (frame_time - self.last_frame_time) if frame_time > self.last_frame_time else 0
            self.fps_history.append(fps)
            
            # 최근 30개 프레임만 유지
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
    
    def get_elapsed_time(self) -> str:
        """경과 시간 문자열"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def save_screenshot(self, frame: np.ndarray):
        """스크린샷 저장"""
        timestamp = int(time.time())
        screenshot_path = f"screenshot_{timestamp}.png"
        cv2.imwrite(screenshot_path, frame)
        logger.info(f"Screenshot saved: {screenshot_path}")
    
    def toggle_pause(self):
        """일시정지/재시작 토글"""
        # 실제 구현에서는 파이프라인 일시정지 기능과 연동
        logger.info("Pause/Resume toggled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        return {
            'total_frames': self.frame_count,
            'current_fps': self.calculate_current_fps(),
            'elapsed_time': time.time() - self.start_time,
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0
        }
    
    # === 새로운 모드별 시각화 메서드들 ===
    
    def draw_keypoints_and_tracking(self, frame: np.ndarray, poses: FramePoses, show_tracking: bool = True) -> np.ndarray:
        """관절점과 트래킹 ID 표시 (실시간 모드용)"""
        if not poses or not poses.persons:
            return frame
        
        # PoseVisualizer를 사용하여 기본 포즈 그리기 (이미 트래킹 ID 포함)
        vis_frame = self.pose_visualizer.visualize_frame(frame, poses)
        
        # PoseVisualizer에서 이미 트래킹 ID를 표시하므로 중복 표시 제거
        # 필요시 추가적인 정보만 여기서 표시
        
        return vis_frame
    
    def draw_realtime_classification(self, frame: np.ndarray, window_results: List[Dict]) -> np.ndarray:
        """실시간 분류 결과 표시 (Previous/Current 패턴)"""
        if not window_results:
            return frame
        
        height, width = frame.shape[:2]
        
        # 해상도에 따른 동적 폰트 크기 조정
        base_scale = min(width, height) / 1000.0  # 1000px 기준으로 정규화
        font_scale = max(0.4, base_scale * 0.5)  # 기존보다 작게, 최소 0.4
        thickness = max(1, int(base_scale * 2))
        line_height = max(25, int(base_scale * 30))
        
        # 해상도에 따른 시작 위치 조정
        start_x = max(10, int(width * 0.02))
        start_y = max(20, int(height * 0.05))
        
        # 윈도우 결과 정렬 (윈도우 ID 순)
        sorted_results = sorted(window_results, key=lambda x: x.get('window_id', 0))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Previous/Current 라벨
        window_labels = ["Previous", "Current"]
        
        for i, window_result in enumerate(sorted_results[-2:]):  # 최근 2개만
            window_id = window_result.get('window_id', 0)
            classification = window_result.get('classification', {})
            
            predicted_class = classification.get('predicted_class', 'Unknown')
            confidence = classification.get('confidence', 0.0)
            probabilities = classification.get('probabilities', [0.0, 0.0])
            
            # probabilities를 딕셔너리로 변환 (필요시)
            if isinstance(probabilities, list) and len(probabilities) >= 2:
                fight_prob = probabilities[1]  # Fight 확률
                nonfight_prob = probabilities[0]  # NonFight 확률
            else:
                fight_prob = probabilities.get('Fight', 0.0) if isinstance(probabilities, dict) else 0.0
                nonfight_prob = probabilities.get('NonFight', 0.0) if isinstance(probabilities, dict) else 0.0
            
            # 라벨 결정
            label = window_labels[i] if i < len(window_labels) else f"Window{i}"
            is_previous = (i == 0 and len(sorted_results) > 1)  # 첫 번째이면서 2개 이상일 때 Previous
            
            # 메인 윈도우 텍스트 (클래스 + 전체 확률)
            window_text = f"{label}-{window_id}: {predicted_class} ({confidence:.3f})"
            
            # 세부 확률 텍스트
            prob_text = f"  Fight: {fight_prob:.3f} | NonFight: {nonfight_prob:.3f}"
            
            # 디버깅: 확률 값 로깅 (주기적으로)
            if window_id % 5 == 1:  # 5개 윈도우마다 한 번씩
                logging.info(f"Window {window_id} probabilities - Fight: {fight_prob:.3f}, NonFight: {nonfight_prob:.3f}, predicted: {predicted_class}")
            
            # 색상 결정
            if is_previous:
                # Previous 윈도우는 항상 검은색 배경에 흰색 글씨
                bg_color = (0, 0, 0)  # 검은색 배경
                text_color = (255, 255, 255)  # 흰색 글씨
            else:
                # Current 윈도우는 confidence_threshold를 고려한 색상 결정
                confidence_threshold = getattr(self, 'confidence_threshold', 0.4)
                
                if fight_prob >= confidence_threshold and fight_prob > nonfight_prob:
                    # Fight 임계값을 넘고 더 높으면 빨간색
                    bg_color = (0, 0, 200)  # 빨간색 배경
                    text_color = (255, 255, 255)
                elif nonfight_prob >= confidence_threshold and nonfight_prob > fight_prob:
                    # NonFight 임계값을 넘고 더 높으면 녹색
                    bg_color = (0, 200, 0)  # 녹색 배경
                    text_color = (255, 255, 255)
                else:
                    # 둘 다 임계값 미달이면 회색 (불확실)
                    bg_color = (128, 128, 128)  # 회색 배경
                    text_color = (255, 255, 255)
            
            y_pos = start_y + i * (line_height + 15)  # 라인 간격을 늘림 (확률 표시 공간 확보)
            
            # 메인 윈도우 텍스트 배경 박스
            text_size = cv2.getTextSize(window_text, font, font_scale, thickness)[0]
            padding = max(5, int(base_scale * 8))
            cv2.rectangle(frame, (start_x - padding, y_pos - 15), 
                         (start_x + text_size[0] + padding, y_pos + 5), bg_color, -1)
            
            # 메인 윈도우 텍스트
            cv2.putText(frame, window_text, (start_x, y_pos), 
                       font, font_scale, text_color, thickness)
            
            # 세부 확률 텍스트 (배경 없이 작은 폰트로)
            prob_font_scale = max(0.3, font_scale * 0.7)  # 확률 텍스트는 더 작게
            cv2.putText(frame, prob_text, (start_x, y_pos + 15), 
                       font, prob_font_scale, (255, 255, 255), 1)
        
        return frame
    
    def add_realtime_info_overlay(self, frame: np.ndarray, additional_info: Dict, overlay_data: Dict) -> np.ndarray:
        """실시간 정보 오버레이 추가"""
        # 우측 상단에 기본 정보
        height, width = frame.shape[:2]
        start_x = width - 200
        start_y = 20
        line_height = 20
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_color = (255, 255, 255)
        
        # 프레임 번호
        frame_idx = overlay_data.get('frame_idx', 0)
        cv2.putText(frame, f"Frame: {frame_idx}", (start_x, start_y), 
                   font, font_scale, text_color, thickness)
        
        # FPS
        if additional_info and 'fps' in additional_info:
            fps_text = f"FPS: {additional_info['fps']:.1f}"
            cv2.putText(frame, fps_text, (start_x, start_y + line_height), 
                       font, font_scale, text_color, thickness)
        
        # 모드 표시
        cv2.putText(frame, f"Mode: {self.processing_mode.upper()}", 
                   (start_x, start_y + line_height * 2), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # 이벤트 상태 표시
        current_y = start_y + line_height * 3
        if additional_info:
            # 현재 이벤트 활성 상태
            event_active = additional_info.get('event_active', False)
            if event_active:
                cv2.putText(frame, "ALERT: VIOLENCE!", (start_x, current_y), 
                           font, font_scale, (0, 0, 255), thickness + 1)
                current_y += line_height
                
                # 이벤트 지속 시간
                event_duration = additional_info.get('event_duration')
                if event_duration:
                    cv2.putText(frame, f"Duration: {event_duration:.1f}s", (start_x, current_y), 
                               font, font_scale, (0, 165, 255), thickness)
                    current_y += line_height
            
            # 연속 탐지 횟수
            consecutive_violence = additional_info.get('consecutive_violence', 0)
            consecutive_normal = additional_info.get('consecutive_normal', 0)
            if consecutive_violence > 0:
                cv2.putText(frame, f"Violence: {consecutive_violence}", (start_x, current_y), 
                           font, font_scale, (0, 100, 255), thickness)
                current_y += line_height
            elif consecutive_normal > 0:
                cv2.putText(frame, f"Normal: {consecutive_normal}", (start_x, current_y), 
                           font, font_scale, (0, 255, 0), thickness)
                current_y += line_height
        
        # 이벤트 히스토리는 apply_realtime_visualization에서 처리하므로 여기서는 제거
        
        return frame
    
    def _draw_event_history(self, frame: np.ndarray) -> np.ndarray:
        """이벤트 상태를 우측 상단에 표시"""
        height, width = frame.shape[:2]
        
        # 우측 상단 위치 설정
        box_width = 250
        box_x = width - box_width - 10
        start_y = 30
        
        # 디버깅: 이벤트 히스토리 상태 출력
        if not hasattr(self, '_last_event_debug_time'):
            self._last_event_debug_time = 0
            
        if time.time() - self._last_event_debug_time > 3:  # 3초마다 한번만 출력
            self._last_event_debug_time = time.time()
            logging.info(f"[EVENT DEBUG] _draw_event_history called! History length: {len(self.event_history)}")
            logging.info(f"[EVENT DEBUG] Frame size: {width}x{height}, Box position: x={box_x}, y={start_y}")
            if self.event_history:
                latest = self.event_history[-1]
                logging.info(f"[EVENT DEBUG] Latest event: {latest}")
        line_height = 25
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 현재 이벤트 상태 판단
        current_status = "Normal"  # 기본 상태
        status_color = (0, 255, 0)  # 기본 초록색 (Normal)
        status_details = ""
        
        if self.event_history:
            latest_event = self.event_history[-1]
            event_type = latest_event.get('event_type', '')
            
            if event_type == 'violence_start':
                current_status = "Violence Detected!"
                status_color = (0, 0, 255)  # 빨간색
                confidence = latest_event.get('confidence', 0.0)
                status_details = f"Confidence: {confidence:.3f}"
                
            elif event_type == 'violence_ongoing':
                current_status = "Violence In Progress"
                status_color = (0, 140, 255)  # 주황색
                duration = latest_event.get('duration', 0.0)
                status_details = f"Duration: {duration:.1f}s"
                
            elif event_type == 'violence_end':
                current_status = "Normal"
                status_color = (0, 255, 0)  # 초록색
                duration = latest_event.get('duration', 0.0)
                status_details = f"Last event: {duration:.1f}s"
        
        # 배경 박스 높이 계산
        box_height = line_height * 3 + 20  # 상태, 상세정보, 여백
        
        # 반투명 배경 박스
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x - 10, start_y - 20), 
                     (box_x + box_width, start_y + box_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 테두리 (더 두껍게 해서 잘 보이도록)
        cv2.rectangle(frame, (box_x - 10, start_y - 20), 
                     (box_x + box_width, start_y + box_height), status_color, 3)
        
        # 디버깅용: 작은 빨간색 점 표시 (우측 상단 모서리)
        cv2.circle(frame, (width - 20, 20), 5, (0, 0, 255), -1)
        
        # 제목
        cv2.putText(frame, "Event Status", (box_x, start_y), 
                   font, 0.5, (255, 255, 255), 1)
        
        # 현재 상태 (큰 글씨, 강조)
        cv2.putText(frame, current_status, (box_x, start_y + line_height), 
                   font, font_scale, status_color, thickness)
        
        # 상세 정보
        if status_details:
            cv2.putText(frame, status_details, (box_x, start_y + line_height * 2), 
                       font, 0.4, (255, 255, 255), 1)
        
        # 추가: 이벤트 히스토리 요약 (항상 표시)
        total_events = len([e for e in self.event_history if e.get('event_type') == 'violence_start']) if self.event_history else 0
        summary_text = f"Total Violence Events: {total_events}"
        cv2.putText(frame, summary_text, (box_x, start_y + line_height * 2 + 20), 
                   font, 0.35, (180, 180, 180), 1)
        
        # 디버깅 정보 (임시)
        debug_text = f"History: {len(self.event_history)} events"
        cv2.putText(frame, debug_text, (box_x, start_y + line_height * 2 + 40), 
                   font, 0.3, (100, 100, 100), 1)
        
        return frame
    
    def draw_basic_poses(self, frame: np.ndarray, poses: FramePoses) -> np.ndarray:
        """기본 포즈 표시 (오버레이 데이터 없을 때)"""
        if not poses or not poses.persons:
            return frame
        
        # 이미 스케일링된 poses를 사용
        return self.pose_visualizer.visualize_frame(frame, poses)
    
    def draw_analysis_poses(self, frame: np.ndarray, poses: FramePoses) -> np.ndarray:
        """분석 모드 포즈 표시"""
        return self.draw_basic_poses(frame, poses)
    
    def draw_composite_scores(self, frame: np.ndarray, poses: FramePoses) -> np.ndarray:
        """복합점수 표시 (분석 모드용)"""
        if not poses or not poses.persons:
            return frame
        
        for person in poses.persons:
            if hasattr(person, 'composite_score') and person.composite_score is not None:
                # bbox 위에 복합점수 표시
                if person.bbox:
                    x1, y1, x2, y2 = person.bbox
                    score_x = int(x1)
                    score_y = int(y1 - 5)
                    
                    # 빨간색 박스로 복합점수 표시
                    score_text = f"Score: {person.composite_score:.3f}"
                    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    
                    cv2.rectangle(frame, (score_x, score_y - 15), 
                                 (score_x + text_size[0] + 10, score_y + 5), (0, 0, 255), -1)
                    cv2.putText(frame, score_text, (score_x + 5, score_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_analysis_classification(self, frame: np.ndarray, classification: Dict) -> np.ndarray:
        """분석 모드 분류 결과 표시"""
        if not classification:
            return frame
        
        # 중앙 상단에 분류 결과 표시
        height, width = frame.shape[:2]
        center_x = width // 2
        start_y = 30
        
        predicted_class = classification.get('predicted_class', 'Unknown')
        confidence = classification.get('confidence', 0.0)
        
        result_text = f"Classification: {predicted_class} ({confidence:.3f})"
        
        # 배경색 결정
        if predicted_class == 'Fight':
            bg_color = (0, 0, 255)
        else:
            bg_color = (0, 255, 0)
        
        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(result_text, font, font_scale, thickness)[0]
        
        # 중앙 정렬
        text_x = center_x - text_size[0] // 2
        
        # 배경 박스
        cv2.rectangle(frame, (text_x - 10, start_y - 20), 
                     (text_x + text_size[0] + 10, start_y + 10), bg_color, -1)
        
        # 텍스트
        cv2.putText(frame, result_text, (text_x, start_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def add_analysis_info_overlay(self, frame: np.ndarray, additional_info: Dict) -> np.ndarray:
        """분석 모드 정보 오버레이"""
        # 우측 하단에 분석 정보
        height, width = frame.shape[:2]
        start_x = width - 300
        start_y = height - 60
        line_height = 20
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_color = (255, 255, 255)
        
        # 처리 시간
        if 'processing_time' in additional_info:
            proc_time = additional_info['processing_time']
            cv2.putText(frame, f"Processing: {proc_time:.3f}s", (start_x, start_y), 
                       font, font_scale, text_color, thickness)
        
        # 모드 표시
        cv2.putText(frame, f"Mode: {self.processing_mode.upper()}", 
                   (start_x, start_y + line_height), 
                   font, font_scale, (255, 255, 0), thickness)
        
        return frame
    
    def toggle_mode(self):
        """모드 전환 (키보드 'M' 키)"""
        if self.processing_mode == 'realtime':
            self.processing_mode = 'analysis'
            self.display_mode = 'analysis'
            self.show_composite_scores = True
            self.realtime_overlay_enabled = False
        else:
            self.processing_mode = 'realtime'
            self.display_mode = 'realtime'
            self.show_composite_scores = False
            self.realtime_overlay_enabled = True
        
        logger.info(f"Mode switched to: {self.processing_mode}")
    
    def set_processing_mode(self, mode: str):
        """처리 모드 설정"""
        if mode in ['realtime', 'analysis']:
            self.processing_mode = mode
            self.display_mode = mode
            
            if mode == 'realtime':
                self.show_composite_scores = False
                self.realtime_overlay_enabled = True
            else:
                self.show_composite_scores = True
                self.realtime_overlay_enabled = False
                
            logger.info(f"Processing mode set to: {mode}")
        else:
            logger.warning(f"Invalid processing mode: {mode}")
    
    def get_processing_mode(self) -> str:
        """현재 처리 모드 반환"""
        return self.processing_mode