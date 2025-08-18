"""
Separated 파이프라인 결과 시각화 도구

separated 파이프라인 결과를 윈도우별 분할 영상으로 시각화합니다.
- 윈도우별 분할 영상 생성 (video_name_window_0.mp4, video_name_window_1.mp4, ...)
- 포즈 추정 결과 관절 표시
- num_persons 수까지는 빨간색 박스, 나머지는 파란색 박스
- 트래킹 ID 및 정렬 순위 표시
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from .pose_visualizer import PoseVisualizer
    from ..utils.data_structure import PersonPose, FramePoses, WindowAnnotation
except ImportError:
    # tools에서 실행시 절대 import
    from visualization.pose_visualizer import PoseVisualizer
    from utils.data_structure import PersonPose, FramePoses, WindowAnnotation


class SeparatedVisualizer:
    """Separated 파이프라인 결과 시각화 클래스"""
    
    def __init__(self, num_persons: int = 2):
        """
        Args:
            num_persons: ST-GCN++에 사용되는 상위 정렬 person 수
        """
        self.num_persons = num_persons
        self.pose_visualizer = PoseVisualizer(
            show_bbox=True,
            show_keypoints=True,
            show_skeleton=True,
            show_track_id=True,
            show_confidence=True
        )
        
        # 색상 정의
        self.top_ranked_color = (0, 0, 255)  # 빨간색 (상위 정렬 객체)
        self.normal_color = (255, 0, 0)  # 파란색 (일반 객체)
        self.keypoint_colors = {
            'face': (255, 255, 0),     # 노란색 (얼굴)
            'upper': (0, 255, 0),      # 초록색 (상체)
            'lower': (255, 0, 0)       # 파란색 (하체)
        }
    
    def visualize_separated_results(self,
                                  input_video_path: str,
                                  window_annotations: List[WindowAnnotation],
                                  output_dir: str,
                                  window_size: int = 100) -> bool:
        """
        Separated 결과 시각화 - 윈도우별 분할 영상 생성
        
        Args:
            input_video_path: 원본 비디오 경로
            window_annotations: 윈도우 어노테이션 리스트
            output_dir: 출력 디렉토리
            window_size: 윈도우 크기
            
        Returns:
            성공 여부
        """
        try:
            # 출력 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 비디오 정보 가져오기
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {input_video_path}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logging.info(f"Video info: {width}x{height}, {fps:.1f}FPS, {total_frames} frames")
            
            # 모든 프레임 읽기
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            # 비디오 이름 추출
            video_name = Path(input_video_path).stem
            
            # 각 윈도우별로 영상 생성
            success_count = 0
            for window_idx, window_annotation in enumerate(window_annotations):
                output_video_path = output_path / f"{video_name}_window_{window_idx}.mp4"
                
                success = self._create_window_video(
                    frames=frames,
                    window_annotation=window_annotation,
                    output_path=str(output_video_path),
                    fps=fps,
                    window_idx=window_idx
                )
                
                if success:
                    success_count += 1
                    logging.info(f"Created window video {window_idx}: {output_video_path}")
                else:
                    logging.error(f"Failed to create window video {window_idx}")
            
            logging.info(f"Created {success_count}/{len(window_annotations)} window videos")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Separated visualization failed: {str(e)}")
            return False
    
    def _create_window_video(self,
                           frames: List[np.ndarray],
                           window_annotation: WindowAnnotation,
                           output_path: str,
                           fps: float,
                           window_idx: int) -> bool:
        """개별 윈도우 영상 생성"""
        try:
            # 윈도우 프레임 범위
            start_frame = window_annotation.start_frame
            end_frame = window_annotation.end_frame
            
            # 유효한 프레임 범위 확인
            start_frame = max(0, min(start_frame, len(frames) - 1))
            end_frame = max(start_frame, min(end_frame, len(frames) - 1))
            
            if start_frame >= len(frames):
                logging.warning(f"Window {window_idx}: start_frame {start_frame} >= total frames {len(frames)}")
                return False
            
            # 비디오 라이터 초기화
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 윈도우 내 프레임들 처리
            window_frames = frames[start_frame:end_frame + 1]
            
            # Person ranking 정보 생성 (복합 점수 기반)
            person_rankings = self._get_person_rankings(window_annotation)
            
            for frame_idx, frame in enumerate(window_frames):
                global_frame_idx = start_frame + frame_idx
                
                # 해당 프레임의 포즈 데이터 생성 (윈도우 어노테이션에서 추출)
                frame_poses = self._extract_frame_poses_from_window(
                    window_annotation, frame_idx, global_frame_idx, person_rankings
                )
                
                # 프레임 시각화
                vis_frame = self._visualize_window_frame(
                    frame, frame_poses, window_idx, frame_idx, global_frame_idx
                )
                
                writer.write(vis_frame)
            
            writer.release()
            return True
            
        except Exception as e:
            logging.error(f"Failed to create window video {window_idx}: {str(e)}")
            return False
    
    def _get_person_rankings(self, window_annotation: WindowAnnotation) -> List[Tuple[int, float]]:
        """Person 순위 정보 반환 (복합 점수 기반)"""
        if hasattr(window_annotation, 'person_rankings') and window_annotation.person_rankings:
            return window_annotation.person_rankings
        
        # 복합 점수가 있는 경우
        if hasattr(window_annotation, 'composite_scores') and window_annotation.composite_scores:
            # 점수별로 정렬
            sorted_scores = sorted(
                window_annotation.composite_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_scores
        
        # 기본값: person ID 순서
        M, T, V, C = window_annotation.keypoint.shape
        return [(person_id, 1.0) for person_id in range(M)]
    
    def _extract_frame_poses_from_window(self,
                                       window_annotation: WindowAnnotation,
                                       frame_idx: int,
                                       global_frame_idx: int,
                                       person_rankings: List[Tuple[int, float]]) -> FramePoses:
        """윈도우 어노테이션에서 특정 프레임의 포즈 데이터 추출"""
        persons = []
        
        # 윈도우 어노테이션의 keypoint 데이터 (M, T, V, C)
        keypoint_data = window_annotation.keypoint
        keypoint_score_data = getattr(window_annotation, 'keypoint_score', None)
        
        M, T, V, C = keypoint_data.shape
        
        # 프레임 인덱스가 유효한 범위인지 확인
        if frame_idx >= T:
            # 패딩된 프레임인 경우 빈 persons 반환
            return FramePoses(
                frame_idx=global_frame_idx,
                persons=[],
                timestamp=global_frame_idx / 30.0,
                image_shape=(640, 640)
            )
        
        # 각 person에 대해 처리
        for person_idx in range(M):
            # 해당 프레임의 키포인트 데이터
            person_keypoints = keypoint_data[person_idx, frame_idx, :, :]  # (V, C)
            
            # 키포인트 스코어
            if keypoint_score_data is not None:
                person_scores = keypoint_score_data[person_idx, frame_idx, :]  # (V,)
                # (V, 3) 형태로 재구성: (x, y, confidence)
                keypoints_with_score = np.column_stack([
                    person_keypoints,  # (V, 2) - x, y
                    person_scores      # (V,) - confidence
                ])
            else:
                # confidence 기본값 1.0
                keypoints_with_score = np.column_stack([
                    person_keypoints,
                    np.ones(V)
                ])
            
            # 유효한 키포인트가 있는지 확인
            valid_keypoints = np.any(person_keypoints > 0)
            if not valid_keypoints:
                continue
            
            # 바운딩 박스 추정 (키포인트에서)
            valid_points = person_keypoints[person_keypoints[:, 0] > 0]
            if len(valid_points) > 0:
                x_coords = valid_points[:, 0]
                y_coords = valid_points[:, 1]
                bbox = [
                    float(np.min(x_coords) - 10),
                    float(np.min(y_coords) - 10),
                    float(np.max(x_coords) + 10),
                    float(np.max(y_coords) + 10)
                ]
            else:
                bbox = [0, 0, 100, 100]  # 기본 박스
            
            # Person 랭킹에서 해당 person의 정보 찾기
            track_id = person_idx
            score = 1.0
            for rank_idx, (pid, rank_score) in enumerate(person_rankings):
                if pid == person_idx:
                    track_id = pid
                    score = rank_score
                    break
            
            # PersonPose 객체 생성
            person_pose = PersonPose(
                person_id=person_idx,
                bbox=bbox,
                keypoints=keypoints_with_score,
                score=score,
                track_id=track_id,
                timestamp=global_frame_idx / 30.0
            )
            
            persons.append(person_pose)
        
        return FramePoses(
            frame_idx=global_frame_idx,
            persons=persons,
            timestamp=global_frame_idx / 30.0,
            image_shape=(640, 640)
        )
    
    def _visualize_window_frame(self,
                              frame: np.ndarray,
                              frame_poses: FramePoses,
                              window_idx: int,
                              local_frame_idx: int,
                              global_frame_idx: int) -> np.ndarray:
        """윈도우 프레임 시각화"""
        vis_frame = frame.copy()
        
        # Person별 색상 결정 및 그리기
        for rank_idx, person in enumerate(frame_poses.persons):
            # 상위 num_persons까지는 빨간색, 나머지는 파란색
            if rank_idx < self.num_persons:
                color = self.top_ranked_color  # 빨간색
            else:
                color = self.normal_color  # 파란색
            
            # 바운딩 박스 그리기
            self._draw_person_bbox(vis_frame, person, color, rank_idx)
            
            # 키포인트와 스켈레톤 그리기
            if person.keypoints is not None and len(person.keypoints) > 0:
                self._draw_person_keypoints(vis_frame, person.keypoints)
                self._draw_person_skeleton(vis_frame, person.keypoints, color)
        
        # 윈도우 정보 표시
        self._draw_window_info(vis_frame, window_idx, local_frame_idx, global_frame_idx, frame_poses)
        
        return vis_frame
    
    def _draw_person_bbox(self, 
                         image: np.ndarray, 
                         person: PersonPose, 
                         color: Tuple[int, int, int],
                         rank_idx: int):
        """Person 바운딩 박스 그리기"""
        x1, y1, x2, y2 = map(int, person.bbox)
        
        # 바운딩 박스
        thickness = 3 if rank_idx < self.num_persons else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 텍스트 정보
        text_lines = []
        if person.track_id is not None:
            text_lines.append(f"ID: {person.track_id}")
        text_lines.append(f"Rank: {rank_idx + 1}")
        text_lines.append(f"Score: {person.score:.2f}")
        
        # 텍스트 그리기
        if text_lines:
            text = " | ".join(text_lines)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # 텍스트 배경
            bg_color = color
            cv2.rectangle(image, (x1, y1 - text_size[1] - 8), 
                         (x1 + text_size[0] + 8, y1), bg_color, -1)
            
            # 텍스트
            cv2.putText(image, text, (x1 + 4, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_person_keypoints(self, image: np.ndarray, keypoints: np.ndarray):
        """Person 키포인트 그리기 (관절 표시)"""
        for i, (x, y, conf) in enumerate(keypoints):
            if x > 0 and y > 0 and conf > 0.1:  # 유효한 키포인트만
                # 키포인트별 색상
                if i < 5:  # 얼굴 (코, 눈, 귀)
                    kpt_color = self.keypoint_colors['face']
                elif i < 11:  # 상체 (어깨, 팔꿈치, 손목)
                    kpt_color = self.keypoint_colors['upper']
                else:  # 하체 (엉덩이, 무릎, 발목)
                    kpt_color = self.keypoint_colors['lower']
                
                # 키포인트 원
                radius = max(2, int(conf * 5))
                cv2.circle(image, (int(x), int(y)), radius, kpt_color, -1)
                cv2.circle(image, (int(x), int(y)), radius + 1, (255, 255, 255), 1)
    
    def _draw_person_skeleton(self, 
                            image: np.ndarray, 
                            keypoints: np.ndarray, 
                            color: Tuple[int, int, int]):
        """Person 스켈레톤 그리기"""
        # COCO 17 키포인트 연결
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
        
        for connection in connections:
            pt1_idx, pt2_idx = connection
            
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                # 두 점이 모두 유효한 경우만
                if (pt1[0] > 0 and pt1[1] > 0 and pt1[2] > 0.1 and
                    pt2[0] > 0 and pt2[1] > 0 and pt2[2] > 0.1):
                    cv2.line(image, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), color, 2)
    
    def _draw_window_info(self, 
                         image: np.ndarray,
                         window_idx: int,
                         local_frame_idx: int,
                         global_frame_idx: int,
                         frame_poses: FramePoses):
        """윈도우 정보 그리기"""
        info_text = f"Window: {window_idx} | Frame: {local_frame_idx} (Global: {global_frame_idx}) | Persons: {len(frame_poses.persons)}"
        
        # 상위 정렬 객체 수 정보
        top_ranked_count = min(len(frame_poses.persons), self.num_persons)
        ranking_text = f"Top Ranked: {top_ranked_count}/{self.num_persons}"
        
        # 텍스트 배경
        text_size1 = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_size2 = cv2.getTextSize(ranking_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        
        max_width = max(text_size1[0], text_size2[0])
        total_height = text_size1[1] + text_size2[1] + 30
        
        # 배경 박스
        cv2.rectangle(image, (10, 10), (max_width + 30, total_height + 10), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (max_width + 30, total_height + 10), (255, 255, 255), 2)
        
        # 텍스트
        cv2.putText(image, info_text, (20, text_size1[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, ranking_text, (20, text_size1[1] + text_size2[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)


def create_separated_visualization(input_video: str,
                                 window_annotations: List[WindowAnnotation],
                                 output_dir: str,
                                 num_persons: int = 2,
                                 window_size: int = 100) -> bool:
    """
    Separated 결과 시각화 생성 (편의 함수)
    
    Args:
        input_video: 입력 비디오 경로
        window_annotations: 윈도우 어노테이션 리스트
        output_dir: 출력 디렉토리
        num_persons: 상위 정렬 person 수
        window_size: 윈도우 크기
        
    Returns:
        성공 여부
    """
    visualizer = SeparatedVisualizer(num_persons=num_persons)
    return visualizer.visualize_separated_results(
        input_video_path=input_video,
        window_annotations=window_annotations,
        output_dir=output_dir,
        window_size=window_size
    )