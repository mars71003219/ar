"""
포즈 시각화 도구

포즈 추정 및 트래킹 결과를 시각화하는 도구입니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import colorsys

try:
    from ..utils.import_utils import safe_import_pose_structures
except ImportError:
    try:
        from utils.import_utils import safe_import_pose_structures
    except ImportError:
        def safe_import_pose_structures():
            try:
                from utils.data_structure import PersonPose, FramePoses
                return PersonPose, FramePoses
            except ImportError:
                from ..utils.data_structure import PersonPose, FramePoses
                return PersonPose, FramePoses
PersonPose, FramePoses = safe_import_pose_structures()


class PoseVisualizer:
    """포즈 시각화 클래스"""
    
    # COCO 17 키포인트 연결 정보
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
        (5, 11), (6, 12), (11, 12),  # 몸통
        (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
    ]
    
    # 키포인트 이름
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, 
                 show_bbox: bool = True,
                 show_keypoints: bool = True,
                 show_skeleton: bool = True,
                 show_track_id: bool = True,
                 show_confidence: bool = True,
                 keypoint_radius: int = 3,
                 skeleton_thickness: int = 2,
                 bbox_thickness: int = 2,
                 max_persons: int = 4):
        """
        Args:
            show_bbox: 바운딩 박스 표시 여부
            show_keypoints: 키포인트 표시 여부
            show_skeleton: 스켈레톤 표시 여부
            show_track_id: 트랙 ID 표시 여부
            show_confidence: 신뢰도 표시 여부
            keypoint_radius: 키포인트 원 반지름
            skeleton_thickness: 스켈레톤 선 두께
            bbox_thickness: 바운딩 박스 선 두께
        """
        self.show_bbox = show_bbox
        self.show_keypoints = show_keypoints
        self.show_skeleton = show_skeleton
        self.show_track_id = show_track_id
        self.show_confidence = show_confidence
        
        self.keypoint_radius = keypoint_radius
        self.skeleton_thickness = skeleton_thickness
        self.bbox_thickness = bbox_thickness
        self.max_persons = max_persons
        
        # 트랙별 색상 캐시
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
        self.color_index = 0
    
    def visualize_frame(self, image: np.ndarray, frame_poses: FramePoses) -> np.ndarray:
        """프레임에 포즈 시각화
        
        Args:
            image: 입력 이미지
            frame_poses: 프레임 포즈 데이터
            
        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()
        
        # person 정렬 (confidence 기준 내림차순)
        persons_sorted = sorted(frame_poses.persons, key=lambda p: p.score if p.score else 0.0, reverse=True)
        
        for idx, person in enumerate(persons_sorted):
            # max_persons 이내의 객체는 빨간색, 나머지는 파란색
            is_top_ranked = idx < self.max_persons
            color = self._get_person_color(person, is_top_ranked)
            
            # 바운딩 박스 그리기
            if self.show_bbox and person.bbox:
                self._draw_bbox(vis_image, person.bbox, color, person)
            
            # 키포인트와 스켈레톤 그리기
            if person.keypoints is not None:
                if self.show_skeleton:
                    self._draw_skeleton(vis_image, person.keypoints, color)
                
                if self.show_keypoints:
                    self._draw_keypoints(vis_image, person.keypoints, color)
        
        # 프레임 정보 추가 (제거 예정)
        # self._draw_frame_info(vis_image, frame_poses)
        
        return vis_image
    
    def _get_person_color(self, person: PersonPose, is_top_ranked: bool = False) -> Tuple[int, int, int]:
        """인물별 색상 반환 (max_persons 이내: 빨간색, 이외: 파란색)"""
        if is_top_ranked:
            # max_persons 이내의 객체는 빨간색
            return (0, 0, 255)  # BGR: 빨간색
        else:
            # 나머지 객체는 파란색
            return (255, 0, 0)  # BGR: 파란색
    
    def _draw_bbox(self, image: np.ndarray, bbox: List[float], color: Tuple[int, int, int], person: PersonPose):
        """바운딩 박스 그리기"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # 텍스트 정보
        text_lines = []
        
        if self.show_track_id and person.track_id is not None:
            text_lines.append(f"ID: {person.track_id}")
        
        if self.show_confidence:
            text_lines.append(f"Conf: {person.score:.2f}")
        
        # 텍스트 그리기
        if text_lines:
            text = " | ".join(text_lines)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # 텍스트 배경
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0] + 5, y1), color, -1)
            
            # 텍스트
            cv2.putText(image, text, (x1 + 2, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]):
        """키포인트 그리기"""
        for i, (x, y) in enumerate(keypoints[:, :2]):
            if x > 0 and y > 0:  # 유효한 키포인트만
                # 키포인트별 색상 구분
                if i < 5:  # 얼굴 (코, 눈, 귀)
                    kpt_color = (255, 255, 0)  # 노란색
                elif i < 11:  # 팔 (어깨, 팔꿈치, 손목)
                    kpt_color = (0, 255, 0)  # 초록색
                else:  # 다리 (엉덩이, 무릎, 발목)
                    kpt_color = (255, 0, 0)  # 파란색
                
                cv2.circle(image, (int(x), int(y)), self.keypoint_radius, kpt_color, -1)
                cv2.circle(image, (int(x), int(y)), self.keypoint_radius + 1, color, 1)
    
    def _draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]):
        """스켈레톤 그리기"""
        for connection in self.SKELETON_CONNECTIONS:
            pt1_idx, pt2_idx = connection
            
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx][:2]
                pt2 = keypoints[pt2_idx][:2]
                
                # 두 점이 모두 유효한 경우만 선 그리기
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(image, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), color, self.skeleton_thickness)
    
    def _draw_frame_info(self, image: np.ndarray, frame_poses: FramePoses):
        """프레임 정보 그리기"""
        info_text = f"Frame: {frame_poses.frame_idx} | Persons: {len(frame_poses.persons)}"
        
        # 텍스트 배경
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 20), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 20), 
                     (255, 255, 255), 1)
        
        # 텍스트
        cv2.putText(image, info_text, (15, text_size[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def visualize_video_poses(self, video_path: str, frame_poses_list: List[FramePoses],
                            output_path: Optional[str] = None, fps: float = 30.0) -> bool:
        """비디오 포즈 시각화
        
        Args:
            video_path: 원본 비디오 경로
            frame_poses_list: 프레임 포즈 데이터 리스트
            output_path: 출력 비디오 경로 (None이면 화면 표시)
            fps: 출력 프레임레이트
            
        Returns:
            성공 여부
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # 비디오 정보
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 설정
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            poses_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 해당 프레임의 포즈 데이터 찾기
                current_poses = None
                if (poses_idx < len(frame_poses_list) and 
                    frame_poses_list[poses_idx].frame_idx == frame_idx):
                    current_poses = frame_poses_list[poses_idx]
                    poses_idx += 1
                
                # 시각화
                if current_poses:
                    vis_frame = self.visualize_frame(frame, current_poses)
                else:
                    vis_frame = frame
                
                # 출력
                if writer:
                    writer.write(vis_frame)
                else:
                    cv2.imshow('Pose Visualization', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
            
            return True
            
        except Exception as e:
            print(f"Error in video visualization: {str(e)}")
            return False
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if not output_path:
                cv2.destroyAllWindows()
    
    def create_pose_summary_image(self, frame_poses_list: List[FramePoses], 
                                output_size: Tuple[int, int] = (1200, 800)) -> np.ndarray:
        """포즈 데이터 요약 이미지 생성
        
        Args:
            frame_poses_list: 프레임 포즈 데이터 리스트
            output_size: 출력 이미지 크기
            
        Returns:
            요약 이미지
        """
        width, height = output_size
        summary_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 통계 계산
        total_frames = len(frame_poses_list)
        total_persons = sum(len(fp.persons) for fp in frame_poses_list)
        unique_tracks = set()
        
        for fp in frame_poses_list:
            for person in fp.persons:
                if person.track_id is not None:
                    unique_tracks.add(person.track_id)
        
        # 텍스트 정보
        stats_text = [
            f"Total Frames: {total_frames}",
            f"Total Person Detections: {total_persons}",
            f"Unique Tracks: {len(unique_tracks)}",
            f"Avg Persons/Frame: {total_persons/total_frames if total_frames > 0 else 0:.2f}"
        ]
        
        # 통계 텍스트 그리기
        y_offset = 50
        for text in stats_text:
            cv2.putText(summary_image, text, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_offset += 50
        
        # 시간별 person 수 그래프 (간단한 버전)
        if total_frames > 1:
            graph_y = height - 200
            graph_height = 150
            
            persons_per_frame = [len(fp.persons) for fp in frame_poses_list]
            max_persons = max(persons_per_frame) if persons_per_frame else 1
            
            for i in range(len(persons_per_frame) - 1):
                x1 = int(50 + (i * (width - 100)) / total_frames)
                x2 = int(50 + ((i + 1) * (width - 100)) / total_frames)
                
                y1 = int(graph_y - (persons_per_frame[i] * graph_height) / max_persons)
                y2 = int(graph_y - (persons_per_frame[i + 1] * graph_height) / max_persons)
                
                cv2.line(summary_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return summary_image