"""
Annotation Stage 전용 시각화 클래스
Stage 1, 2의 PKL 데이터를 간단하게 시각화합니다.
"""

import cv2
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    from ..utils.data_structure import PersonPose, FramePoses
except ImportError:
    from utils.data_structure import PersonPose, FramePoses

logger = logging.getLogger(__name__)


class AnnotationStageVisualizer:
    """Annotation Stage 전용 시각화 클래스"""
    
    def __init__(self, max_persons: int = 4):
        """초기화
        
        Args:
            max_persons: 우선순위 정렬에서 상위 N명 (빨간색으로 표시)
        """
        self.max_persons = max_persons
        
        # COCO 17 키포인트 연결 구조
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
        
        # 색상 설정
        self.keypoint_color = (0, 255, 0)      # 초록색
        self.skeleton_color = (255, 0, 0)      # 파란색
        self.top_person_color = (0, 0, 255)    # 빨간색 (상위 max_persons)
        self.other_person_color = (255, 0, 0)  # 파란색 (나머지)
        self.track_id_color = (255, 255, 0)    # 노란색
        self.text_color = (255, 255, 255)      # 흰색
        
    def visualize_stage1_pkl(self, pkl_path: Union[str, Path], 
                            video_path: Union[str, Path],
                            output_path: Optional[Union[str, Path]] = None) -> bool:
        """Stage 1 PKL 파일 시각화 (포즈만)
        
        Args:
            pkl_path: Stage 1 PKL 파일 경로
            video_path: 원본 비디오 경로
            output_path: 출력 비디오 경로 (None이면 표시만)
            
        Returns:
            성공 여부
        """
        try:
            # PKL 데이터 로드
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
            
            # FramePoses 리스트 추출
            if hasattr(pkl_data, 'poses_only'):
                frame_poses_list = pkl_data.poses_only
            elif isinstance(pkl_data, list):
                frame_poses_list = pkl_data
            else:
                logger.error(f"Unsupported PKL format in {pkl_path}")
                return False
            
            return self._visualize_frame_poses(
                frame_poses_list=frame_poses_list,
                video_path=video_path,
                output_path=output_path,
                show_track_id=False  # Stage 1은 track_id 없음
            )
            
        except Exception as e:
            logger.error(f"Error visualizing stage1 PKL {pkl_path}: {e}")
            return False
    
    def visualize_stage2_pkl(self, pkl_path: Union[str, Path], 
                            video_path: Union[str, Path],
                            output_path: Optional[Union[str, Path]] = None) -> bool:
        """Stage 2 PKL 파일 시각화 (포즈 + 트래킹 ID)
        
        Args:
            pkl_path: Stage 2 PKL 파일 경로
            video_path: 원본 비디오 경로
            output_path: 출력 비디오 경로 (None이면 표시만)
            
        Returns:
            성공 여부
        """
        try:
            # PKL 데이터 로드
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
            
            # FramePoses 리스트 추출
            if hasattr(pkl_data, 'poses_with_tracking'):
                frame_poses_list = pkl_data.poses_with_tracking
            elif hasattr(pkl_data, 'frame_data'):
                frame_poses_list = pkl_data.frame_data
            elif hasattr(pkl_data, 'poses_only'):
                frame_poses_list = pkl_data.poses_only
            elif isinstance(pkl_data, list):
                frame_poses_list = pkl_data
            else:
                logger.error(f"Unsupported PKL format in {pkl_path}")
                return False
            
            return self._visualize_frame_poses(
                frame_poses_list=frame_poses_list,
                video_path=video_path,
                output_path=output_path,
                show_track_id=True  # Stage 2는 track_id 표시
            )
            
        except Exception as e:
            logger.error(f"Error visualizing stage2 PKL {pkl_path}: {e}")
            return False
    
    def _visualize_frame_poses(self, frame_poses_list: List[FramePoses],
                              video_path: Union[str, Path],
                              output_path: Optional[Union[str, Path]] = None,
                              show_track_id: bool = False) -> bool:
        """프레임별 포즈 데이터 시각화
        
        Args:
            frame_poses_list: 프레임별 포즈 데이터 리스트
            video_path: 원본 비디오 경로
            output_path: 출력 비디오 경로 (None이면 표시만)
            show_track_id: 트래킹 ID 표시 여부
            
        Returns:
            성공 여부
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 설정
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        logger.info(f"Processing {total_frames} frames from {Path(video_path).name}")
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 해당 프레임의 포즈 데이터 찾기
                if frame_idx < len(frame_poses_list):
                    frame_poses = frame_poses_list[frame_idx]
                    
                    # 포즈 그리기 (복합점수 정렬 포함)
                    self._draw_poses(frame, frame_poses.persons, show_track_id, frame_idx)
                    
                    # 프레임 정보 표시
                    self._draw_frame_info(frame, frame_idx, len(frame_poses.persons))
                
                if writer:
                    writer.write(frame)
                else:
                    # 화면 표시 (ESC로 종료)
                    cv2.imshow('Annotation Visualization', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
                
                frame_idx += 1
            
            logger.info(f"Visualization completed: {frame_idx} frames processed")
            return True
            
        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            return False
            
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def _draw_poses(self, frame: np.ndarray, persons: List[PersonPose], show_track_id: bool, frame_idx: int = 0):
        """포즈 그리기 (복합점수 정렬 및 색상 구분 포함)
        
        Args:
            frame: 비디오 프레임
            persons: 사람 포즈 리스트
            show_track_id: 트랙 ID 표시 여부
            frame_idx: 프레임 인덱스 (로깅용)
        """
        if not persons:
            return
        
        # Stage 2에서는 복합점수로 정렬 (score 기준)
        if show_track_id:
            # 복합점수 기준으로 내림차순 정렬
            sorted_persons = sorted(persons, 
                                  key=lambda p: getattr(p, 'score', 0.0), 
                                  reverse=True)
        else:
            # Stage 1에서는 기본 정렬 유지
            sorted_persons = persons
        
        for rank, person in enumerate(sorted_persons):
            # 색상 결정: 상위 max_persons는 빨간색, 나머지는 파란색
            is_top_ranked = rank < self.max_persons
            person_color = self.top_person_color if is_top_ranked else self.other_person_color
            
            # 바운딩 박스 그리기
            if person.bbox is not None and len(person.bbox) >= 4:
                # bbox가 list인 경우 numpy array로 변환
                if isinstance(person.bbox, list):
                    bbox = np.array(person.bbox)
                else:
                    bbox = person.bbox
                x1, y1, x2, y2 = bbox[:4].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
                
                # 텍스트 정보 구성
                text_lines = []
                if show_track_id:
                    # Stage 2: Track ID, 정렬 순위, 복합점수 표시
                    if person.track_id is not None:
                        text_lines.append(f"Track: {person.track_id}")
                    text_lines.append(f"Rank: {rank + 1}")
                    if hasattr(person, 'score') and person.score is not None:
                        text_lines.append(f"Score: {person.score:.2f}")
                else:
                    # Stage 1: Person ID만 표시
                    text_lines.append(f"Person: {person.person_id}")
                
                # 텍스트 표시
                for i, text in enumerate(text_lines):
                    y_offset = y1 - 10 - (i * 20)  # 여러 줄 텍스트를 위해 간격 조정
                    cv2.putText(frame, text, (x1, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            
            # 키포인트와 스켈레톤 그리기
            if person.keypoints is not None and len(person.keypoints) >= 17:
                keypoints = person.keypoints[:17]  # COCO 17 키포인트
                
                # 키포인트 그리기
                for i, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.3:  # 신뢰도 임계값
                        cv2.circle(frame, (int(x), int(y)), 4, self.keypoint_color, -1)
                
                # 스켈레톤 그리기 (person_color 사용)
                for connection in self.skeleton_connections:
                    pt1_idx, pt2_idx = connection
                    if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                        keypoints[pt1_idx][2] > 0.3 and keypoints[pt2_idx][2] > 0.3):
                        
                        pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                        pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                        cv2.line(frame, pt1, pt2, person_color, 2)
    
    def _draw_frame_info(self, frame: np.ndarray, frame_idx: int, person_count: int):
        """프레임 정보 표시"""
        info_text = f"Frame: {frame_idx}, Persons: {person_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)