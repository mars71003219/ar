#!/usr/bin/env python3
"""
Video Overlay Module with ByteTrack Integration
비디오 오버레이 모듈 - ByteTrack 기반 Track ID 시각화 및 STGCN 객체 구분
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import logging
import os
import os.path as osp
from pathlib import Path
import tempfile
from fight_tracker import FightPrioritizedTracker

logger = logging.getLogger(__name__)

# MMPose imports for better visualization
try:
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import PoseDataSample
    from mmengine.structures import InstanceData
    MMPOSE_AVAILABLE = True
    logger.info("MMPose import 성공 - 고품질 시각화 사용 가능")
except ImportError as e:
    MMPOSE_AVAILABLE = False
    logger.warning(f"MMPose import 실패: {e} - 기본 시각화로 fallback")

class VideoOverlayGenerator:
    """
    ByteTrack 기반 비디오 오버레이 생성기
    - STGCN 입력 객체: 연녹색 형광
    - 기타 객체: 파란색
    - Track ID 표시
    """
    
    def __init__(self, skeleton_connections: Optional[List[Tuple[int, int]]] = None,
                 stgcn_joint_color: Tuple[int, int, int] = (0, 255, 128),  # 연녹색 형광 (BGR)
                 stgcn_skeleton_color: Tuple[int, int, int] = (0, 255, 128),  # 연녹색 형광
                 other_joint_color: Tuple[int, int, int] = (255, 0, 0),  # 파란색 (BGR)
                 other_skeleton_color: Tuple[int, int, int] = (255, 0, 0),  # 파란색
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 font_scale: float = 0.4,
                 thickness: int = 1,
                 point_radius: int = 1):
        """
        초기화
        
        Args:
            skeleton_connections: 스켈레톤 연결 정보 (키포인트 인덱스 쌍)
            stgcn_joint_color: STGCN 입력 객체 관절 색상 (연녹색 형광)
            stgcn_skeleton_color: STGCN 입력 객체 스켈레톤 색상 
            other_joint_color: 기타 객체 관절 색상 (파란색)
            other_skeleton_color: 기타 객체 스켈레톤 색상
            text_color: 텍스트 색상 (B, G, R)
            font_scale: 폰트 크기
            thickness: 선 두께
            point_radius: 관절 포인트 반지름
        """
        # COCO 17-point 스켈레톤 연결 (1-based에서 0-based로 변환)
        self.skeleton_connections = skeleton_connections or [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # 0-based
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)
        ]
        
        # STGCN 입력 객체 색상 (연녹색 형광)
        self.stgcn_joint_color = stgcn_joint_color
        self.stgcn_skeleton_color = stgcn_skeleton_color
        
        # 기타 객체 색상 (파란색)
        self.other_joint_color = other_joint_color
        self.other_skeleton_color = other_skeleton_color
        
        # 공통 설정
        self.text_color = text_color
        self.font_scale = font_scale
        self.thickness = thickness
        self.point_radius = point_radius
        
        # 단일 트래커 인스턴스 (오버레이용)
        self.tracker = None
        
        # MMPose 시각화기 (더 예쁜 오버레이용)
        self.visualizer = None
        self._init_mmpose_visualizer()
        
        # 키포인트 이름 (COCO 17-point)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        logger.info("ByteTrack 기반 비디오 오버레이 생성기 초기화 완료")
    
    def _init_mmpose_visualizer(self):
        """MMPose 시각화기 초기화 (rtmo_pose_track 스타일 적용)"""
        if not MMPOSE_AVAILABLE:
            return
            
        try:
            # MMPose 시각화기는 pose_model이 필요하므로 나중에 초기화
            # pose_model.cfg.visualizer를 사용해야 함
            self.visualizer = None  # 나중에 pose_model과 함께 초기화
            logger.info("MMPose 시각화기 설정 준비 완료 - pose_model 연결 대기 중")
            
        except Exception as e:
            logger.warning(f"MMPose 시각화기 설정 실패: {e}")
            self.visualizer = None
    
    def init_visualizer_with_pose_model(self, pose_model):
        """pose_model을 사용한 MMPose 시각화기 초기화 (rtmo_pose_track 방식)"""
        if not MMPOSE_AVAILABLE or pose_model is None:
            return False
            
        try:
            # MMPose 시각화기 직접 생성 (registry 충돌 방지)
            from mmpose.visualization import PoseLocalVisualizer
            
            # 시각화기 직접 초기화 (설정 최소화)
            self.visualizer = PoseLocalVisualizer()
            self.visualizer.set_dataset_meta(pose_model.dataset_meta)
            logger.info("MMPose 시각화기 직접 초기화 완료")
            return True
            
        except Exception as e:
            logger.warning(f"MMPose 시각화기 초기화 실패: {e}")
            self.visualizer = None
            return False
    
    def create_overlay_with_bytetrack(self, video_path: str, pose_results: List[Tuple],
                                    classification_result: Dict, output_path: str,
                                    num_person: int = 2, tracker: Optional[FightPrioritizedTracker] = None) -> bool:
        """
        ByteTrack 기반 오버레이 비디오 생성
        
        Args:
            video_path: 원본 비디오 경로
            pose_results: 포즈 추정 결과
            classification_result: 분류 결과
            output_path: 출력 비디오 경로
            num_person: STGCN 입력 인수 수
            tracker: FightPrioritizedTracker 인스턴스
            
        Returns:
            성공 여부
        """
        try:
            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"비디오 파일을 열 수 없습니다: {video_path}")
                return False
            
            # 비디오 정보 가져오기
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 출력 비디오 설정 (XVID 코덱 사용)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"출력 비디오를 생성할 수 없습니다: {output_path}")
                cap.release()
                return False
            
            # 트래커 초기화 (전달되지 않으면 새로 생성)
            if tracker is None:
                tracker = FightPrioritizedTracker(frame_width=width, frame_height=height)
            else:
                tracker.reset()
            
            self.tracker = tracker
            
            # detection 데이터 준비
            detections_list = []
            keypoints_list = []
            scores_list = []
            
            for keypoints, scores in pose_results:
                if keypoints is not None and scores is not None and len(keypoints) > 0:
                    keypoints_frame_list = [keypoints[i] for i in range(len(keypoints))]
                    scores_frame_list = [scores[i] for i in range(len(scores))]
                    
                    detections = tracker.create_detections_from_pose_results(
                        keypoints_frame_list, scores_frame_list
                    )
                    detections_list.append(detections)
                    keypoints_list.append(keypoints_frame_list)
                    scores_list.append(scores_frame_list)
                else:
                    detections_list.append(np.empty((0, 5)))
                    keypoints_list.append([])
                    scores_list.append([])
            
            # 분류 결과에서 window_results 추출
            window_results = classification_result.get('window_results', [])
            
            logger.info(f"오버레이 비디오 생성 시작: {total_frames}프레임")
            
            # 각 프레임 처리
            for frame_idx in range(len(pose_results)):
                success, frame = cap.read()
                if not success:
                    logger.warning(f"프레임 {frame_idx} 읽기 실패")
                    break
                
                try:
                    # ByteTracker 업데이트
                    if frame_idx < len(detections_list) and len(detections_list[frame_idx]) > 0:
                        active_tracks = tracker.update_with_detections(
                            detections_list[frame_idx], 
                            keypoints_list[frame_idx], 
                            scores_list[frame_idx]
                        )
                        
                        # STGCN 입력용 상위 Track ID 가져오기
                        selected_keypoints, selected_scores, selected_track_ids = tracker.select_fight_prioritized_people(num_person)
                        stgcn_track_ids = set(selected_track_ids)
                        
                        # 모든 active tracks 그리기
                        overlay_frame = self._draw_all_tracks_with_ids(
                            frame, active_tracks, stgcn_track_ids, score_threshold=0.3
                        )
                    else:
                        overlay_frame = frame.copy()
                        stgcn_track_ids = set()
                    
                    # 윈도우 예측 결과 표시
                    overlay_frame = self.add_window_prediction_text(
                        overlay_frame, frame_idx, window_results, 
                        window_size=30, stride=15, position=(30, 30)
                    )
                    
                    # Track ID 및 STGCN 선택 정보 표시
                    if stgcn_track_ids:
                        info_text = f"STGCN Input: {sorted(list(stgcn_track_ids))}"
                        cv2.putText(overlay_frame, info_text, (30, height - 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.thickness)
                    
                    out.write(overlay_frame)
                    
                except Exception as e:
                    logger.warning(f"프레임 {frame_idx} 처리 오류: {e}")
                    out.write(frame)  # 원본 프레임 사용
            
            # 리소스 정리
            cap.release()
            out.release()
            
            logger.info(f"오버레이 비디오 생성 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"오버레이 비디오 생성 실패: {e}")
            return False
    
    def _draw_all_tracks_with_ids(self, image: np.ndarray, tracks: List, 
                                 stgcn_track_ids: Set[int], score_threshold: float = 0.3) -> np.ndarray:
        """
        모든 트랙을 Track ID와 함께 그리기 (MMPose 시각화 + STGCN 입력 객체 구분)
        
        Args:
            image: 입력 이미지
            tracks: ByteTrack 트랙 리스트
            stgcn_track_ids: STGCN 입력으로 선택된 Track ID 집합
            score_threshold: 키포인트 신뢰도 임계값
            
        Returns:
            트랙이 그려진 이미지
        """
        # 트랙 데이터 준비
        keypoints_list = []
        scores_list = []
        track_ids = []
        
        for track in tracks:
            if track.keypoints is not None:
                keypoints = track.keypoints
                if len(keypoints.shape) == 3:
                    keypoints = keypoints[0]  # (1, 17, 2) -> (17, 2)
                
                keypoint_scores = track.keypoint_scores
                if keypoint_scores is not None and len(keypoint_scores.shape) == 2:
                    keypoint_scores = keypoint_scores[0]  # (1, 17) -> (17,)
                elif keypoint_scores is None:
                    keypoint_scores = np.ones(17) * track.score  # 바운딩 박스 신뢰도 사용
                
                keypoints_list.append(keypoints)
                scores_list.append(keypoint_scores)
                track_ids.append(track.track_id)
        
        # MMPose 시각화 또는 기본 시각화 사용
        if keypoints_list:
            return self._draw_with_mmpose_visualizer(image, keypoints_list, scores_list, stgcn_track_ids, track_ids)
        else:
            return image.copy()
    
    def _draw_with_mmpose_visualizer(self, image: np.ndarray, keypoints_list: List[np.ndarray], 
                                   scores_list: List[np.ndarray], stgcn_track_ids: Set[int],
                                   track_ids: List[int]) -> np.ndarray:
        """
        MMPose 시각화기를 사용한 예쁜 오버레이 생성 (STGCN 입력 객체 구분)
        
        Args:
            image: 입력 이미지
            keypoints_list: 키포인트 리스트 [(17, 2), ...]
            scores_list: 점수 리스트 [(17,), ...]
            stgcn_track_ids: STGCN 입력으로 선택된 Track ID 집합
            track_ids: Track ID 리스트
            
        Returns:
            시각화된 이미지
        """
        if not MMPOSE_AVAILABLE or self.visualizer is None:
            # Fallback to basic visualization
            return self._draw_all_tracks_with_ids_basic(image, keypoints_list, scores_list, stgcn_track_ids, track_ids)
        
        try:
            # STGCN 입력 객체와 기타 객체 분리
            stgcn_keypoints = []
            stgcn_scores = []
            other_keypoints = []
            other_scores = []
            
            for i, (kpts, scores, track_id) in enumerate(zip(keypoints_list, scores_list, track_ids)):
                if track_id in stgcn_track_ids:
                    stgcn_keypoints.append(kpts)
                    stgcn_scores.append(scores)
                else:
                    other_keypoints.append(kpts)
                    other_scores.append(scores)
            
            overlay_image = image.copy()
            
            # 1. 기타 객체들 먼저 그리기 (파란색)
            if other_keypoints:
                other_data_sample = self._create_pose_data_sample(other_keypoints, other_scores, use_stgcn_colors=False)
                self.visualizer.add_datasample(
                    'other_objects',
                    overlay_image,
                    data_sample=other_data_sample,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=False,
                    show_kpt_idx=False,
                    skeleton_style='mmpose'
                )
                overlay_image = self.visualizer.get_image()
            
            # 2. STGCN 입력 객체 그리기 (연녹색 형광)
            if stgcn_keypoints:
                stgcn_data_sample = self._create_pose_data_sample(stgcn_keypoints, stgcn_scores, use_stgcn_colors=True)
                self.visualizer.add_datasample(
                    'stgcn_objects',
                    overlay_image,
                    data_sample=stgcn_data_sample,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=False,
                    show_kpt_idx=False,
                    skeleton_style='mmpose'
                )
                overlay_image = self.visualizer.get_image()
            
            # 3. Track ID 라벨 추가
            for i, (kpts, track_id) in enumerate(zip(keypoints_list, track_ids)):
                is_stgcn_input = track_id in stgcn_track_ids
                head_point = self._get_head_position(kpts, scores_list[i], 0.3)
                if head_point is not None:
                    bg_color = (0, 200, 100) if is_stgcn_input else (200, 50, 0)
                    self._draw_track_id_label(overlay_image, head_point, track_id, bg_color, is_stgcn_input)
            
            return overlay_image
            
        except Exception as e:
            logger.warning(f"MMPose 시각화 실패, 기본 시각화로 대체: {e}")
            return self._draw_all_tracks_with_ids_basic(image, keypoints_list, scores_list, stgcn_track_ids, track_ids)
    
    def _create_pose_data_sample(self, keypoints_list: List[np.ndarray], scores_list: List[np.ndarray], 
                                use_stgcn_colors: bool = False) -> PoseDataSample:
        """MMPose용 PoseDataSample 생성"""
        data_sample = PoseDataSample()
        
        if keypoints_list:
            # keypoints: (N, 17, 3) 형태로 변환 (x, y, visibility)
            keypoints_3d = []
            for kpts, scores in zip(keypoints_list, scores_list):
                # (17, 2) + (17,) -> (17, 3)
                kpts_with_vis = np.concatenate([kpts, scores.reshape(-1, 1)], axis=1)
                keypoints_3d.append(kpts_with_vis)
            
            keypoints_array = np.array(keypoints_3d)  # (N, 17, 3)
            
            # InstanceData 생성
            pred_instances = InstanceData()
            pred_instances.keypoints = keypoints_array
            pred_instances.keypoint_scores = np.array([scores for scores in scores_list])
            
            # STGCN 입력 객체는 특별한 색상 사용
            if use_stgcn_colors:
                # 연녹색 형광 설정을 위한 메타데이터 (임시)
                pred_instances.track_id = [999] * len(keypoints_list)  # 특별한 ID로 표시
            
            data_sample.pred_instances = pred_instances
        
        return data_sample
    
    def _draw_all_tracks_with_ids_basic(self, image: np.ndarray, keypoints_list: List[np.ndarray], 
                                      scores_list: List[np.ndarray], stgcn_track_ids: Set[int],
                                      track_ids: List[int]) -> np.ndarray:
        """기본 시각화 방식 (MMPose 사용 불가시 Fallback)"""
        overlay_image = image.copy()
        
        for i, (kpts, scores, track_id) in enumerate(zip(keypoints_list, scores_list, track_ids)):
            is_stgcn_input = track_id in stgcn_track_ids
            
            # 색상 선택
            if is_stgcn_input:
                joint_color = self.stgcn_joint_color
                skeleton_color = self.stgcn_skeleton_color
                text_bg_color = (0, 200, 100)
            else:
                joint_color = self.other_joint_color
                skeleton_color = self.other_skeleton_color
                text_bg_color = (200, 50, 0)
            
            # 키포인트 그리기
            overlay_image = self._draw_single_person_with_color(
                overlay_image, kpts, scores, 0.3, joint_color, skeleton_color
            )
            
            # Track ID 표시
            head_point = self._get_head_position(kpts, scores, 0.3)
            if head_point is not None:
                self._draw_track_id_label(overlay_image, head_point, track_id, text_bg_color, is_stgcn_input)
        
        return overlay_image
    
    def _draw_single_person_with_color(self, image: np.ndarray, keypoints: np.ndarray, 
                                      scores: np.ndarray, score_threshold: float,
                                      joint_color: Tuple[int, int, int], 
                                      skeleton_color: Tuple[int, int, int]) -> np.ndarray:
        """
        지정된 색상으로 단일 인물의 키포인트 그리기
        
        Args:
            image: 입력 이미지
            keypoints: 키포인트 좌표 (17, 2)
            scores: 키포인트 신뢰도 (17,)
            score_threshold: 신뢰도 임계값
            joint_color: 관절 색상
            skeleton_color: 스켈레톤 색상
            
        Returns:
            키포인트가 그려진 이미지
        """
        overlay_image = image.copy()
        
        # 키포인트 그리기
        for i, (point, score) in enumerate(zip(keypoints, scores)):
            if score > score_threshold and point[0] > 0 and point[1] > 0:
                x, y = int(point[0]), int(point[1])
                cv2.circle(overlay_image, (x, y), self.point_radius, joint_color, -1)
        
        # 스켈레톤 연결선 그리기
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                scores[pt1_idx] > score_threshold and scores[pt2_idx] > score_threshold and
                keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
                keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
                
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                cv2.line(overlay_image, pt1, pt2, skeleton_color, self.thickness)
        
        return overlay_image
    
    def _get_head_position(self, keypoints: np.ndarray, scores: np.ndarray, 
                          score_threshold: float) -> Optional[Tuple[int, int]]:
        """
        머리 위치 계산 (코, 눈, 귀 순서로 우선순위)
        
        Args:
            keypoints: 키포인트 좌표 (17, 2)
            scores: 키포인트 신뢰도 (17,)
            score_threshold: 신뢰도 임계값
            
        Returns:
            머리 위치 (x, y) 또는 None
        """
        # 머리 부위 우선순위: 코(0), 왼쪽눈(1), 오른쪽눈(2), 왼쪽귀(3), 오른쪽귀(4)
        head_indices = [0, 1, 2, 3, 4]
        
        for idx in head_indices:
            if (idx < len(keypoints) and scores[idx] > score_threshold and 
                keypoints[idx][0] > 0 and keypoints[idx][1] > 0):
                return (int(keypoints[idx][0]), int(keypoints[idx][1]) - 20)  # 약간 위에 표시
        
        # 머리 부위를 찾을 수 없으면 어깨 중심 사용
        shoulder_indices = [5, 6]  # 왼쪽어깨, 오른쪽어깨
        valid_shoulders = []
        
        for idx in shoulder_indices:
            if (idx < len(keypoints) and scores[idx] > score_threshold and 
                keypoints[idx][0] > 0 and keypoints[idx][1] > 0):
                valid_shoulders.append(keypoints[idx])
        
        if valid_shoulders:
            shoulder_center = np.mean(valid_shoulders, axis=0)
            return (int(shoulder_center[0]), int(shoulder_center[1] - 30))  # 어깨 위에 표시
        
        return None
    
    def _draw_track_id_label(self, image: np.ndarray, position: Tuple[int, int], 
                           track_id: int, bg_color: Tuple[int, int, int], 
                           is_stgcn_input: bool) -> None:
        """
        Track ID 라벨 그리기
        
        Args:
            image: 입력 이미지
            position: 텍스트 위치 (x, y)
            track_id: 트랙 ID
            bg_color: 배경 색상
            is_stgcn_input: STGCN 입력 객체 여부
        """
        x, y = position
        
        # 텍스트 내용
        if is_stgcn_input:
            text = f"ID:{track_id}*"
            text_color = (255, 255, 255)  # 흰색
        else:
            text = f"ID:{track_id}"
            text_color = (255, 255, 255)  # 흰색
        
        # 텍스트 크기 측정
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, self.font_scale, self.thickness)
        
        # 배경 매각형 그리기
        padding = 3
        cv2.rectangle(image, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     bg_color, -1)
        
        # 텍스트 그리기
        cv2.putText(image, text, (x, y), font, self.font_scale, text_color, self.thickness)
    
    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray, 
                      scores: np.ndarray, score_threshold: float = 0.3, 
                      selected_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        이미지에 키포인트 그리기 (하위 호환성 유지)
        
        Args:
            image: 입력 이미지 (H, W, 3)
            keypoints: 키포인트 좌표 (17, 2) 또는 (N, 17, 2)
            scores: 키포인트 신뢰도 (17,) 또는 (N, 17)
            score_threshold: 신뢰도 임계값
            selected_indices: STGCN 입력으로 선택된 인물 인덱스 (형광색 표시용)
            
        Returns:
            키포인트가 그려진 이미지
        """
        logger.warning("기존 draw_keypoints 메서드는 더 이상 사용되지 않습니다. create_overlay_with_bytetrack 사용을 권장합니다.")
        
        overlay_image = image.copy()
        
        # 다중 인물 처리
        if len(keypoints.shape) == 3:
            for person_idx, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
                is_selected = selected_indices is not None and person_idx in selected_indices
                if is_selected:
                    joint_color = self.stgcn_joint_color
                    skeleton_color = self.stgcn_skeleton_color
                else:
                    joint_color = self.other_joint_color
                    skeleton_color = self.other_skeleton_color
                
                overlay_image = self._draw_single_person_with_color(
                    overlay_image, person_kpts, person_scores, score_threshold,
                    joint_color, skeleton_color
                )
        else:
            is_selected = selected_indices is not None and 0 in selected_indices
            if is_selected:
                joint_color = self.stgcn_joint_color
                skeleton_color = self.stgcn_skeleton_color
            else:
                joint_color = self.other_joint_color
                skeleton_color = self.other_skeleton_color
                
            overlay_image = self._draw_single_person_with_color(
                overlay_image, keypoints, scores, score_threshold,
                joint_color, skeleton_color
            )
        
        return overlay_image
    
    def _draw_single_person(self, image: np.ndarray, keypoints: np.ndarray, 
                           scores: np.ndarray, score_threshold: float, 
                           is_selected: bool = False) -> np.ndarray:
        """
        단일 인물의 키포인트 그리기
        
        Args:
            image: 입력 이미지
            keypoints: 키포인트 좌표 (17, 2)
            scores: 키포인트 신뢰도 (17,)
            score_threshold: 신뢰도 임계값
            is_selected: STGCN 입력으로 선택된 인물인지 여부
            
        Returns:
            키포인트가 그려진 이미지
        """
        overlay_image = image.copy()
        
        # 색상 선택: STGCN 입력 선택된 인물은 형광색, 일반 인물은 기본색
        if is_selected:
            joint_color = (0, 255, 255)    # 형광 노란색 (Cyan in BGR)
            skeleton_color = (0, 255, 255) # 형광 노란색
        else:
            joint_color = self.joint_color
            skeleton_color = self.skeleton_color
        
        # 키포인트 그리기
        for i, (point, score) in enumerate(zip(keypoints, scores)):
            if score > score_threshold and point[0] > 0 and point[1] > 0:
                x, y = int(point[0]), int(point[1])
                cv2.circle(overlay_image, (x, y), self.point_radius, joint_color, -1)
                
                # 키포인트 번호 표시 (선택사항)
                # cv2.putText(overlay_image, str(i), (x+5, y-5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.text_color, 1)
        
        # 스켈레톤 연결선 그리기
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                scores[pt1_idx] > score_threshold and scores[pt2_idx] > score_threshold and
                keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
                keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
                
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                cv2.line(overlay_image, pt1, pt2, skeleton_color, self.thickness)
        
        return overlay_image
    
    def add_window_prediction_text(self, image: np.ndarray, frame_idx: int, 
                                  window_results: List[Dict],
                                  window_size: int = 30, stride: int = 15,
                                  position: Tuple[int, int] = (30, 50)) -> np.ndarray:
        """
        현재 프레임의 윈도우별 실시간 예측 결과 표시
        
        Args:
            image: 입력 이미지
            frame_idx: 현재 프레임 인덱스
            window_results: 윈도우 결과 [{"pred": 0, "scores": [0.78, 0.22]}, ...]
            window_size: 윈도우 크기
            stride: 윈도우 간격
            position: 텍스트 위치
            
        Returns:
            텍스트가 추가된 이미지
        """
        overlay_image = image.copy()
        x, y = position
        
        if not window_results:
            # 데이터가 없으면 기본 메시지
            text = "Analyzing..."
            cv2.putText(overlay_image, text, (x, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.thickness)
            return overlay_image
        
        # 현재 프레임이 속한 윈도우 찾기
        current_window_idx = None
        for w_idx in range(len(window_results)):
            window_start = w_idx * stride
            window_end = window_start + window_size
            if window_start <= frame_idx < window_end:
                current_window_idx = w_idx
                break
        
        if current_window_idx is None or current_window_idx >= len(window_results):
            # 윈도우가 없으면 기본 메시지
            text = "Analyzing..."
            cv2.putText(overlay_image, text, (x, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.thickness)
            return overlay_image
        
        # 현재 윈도우의 예측 결과 추출
        window_data = window_results[current_window_idx]
        prediction = window_data["pred"]
        scores = window_data["scores"]
        nonfight_score = scores[0] if len(scores) > 0 else 0.5
        fight_score = scores[1] if len(scores) > 1 else 0.5
        
        prediction_label = 'Fight' if prediction == 1 else 'NonFight'
        
        # 텍스트 구성 (배경 박스 없이, 더 작은 글씨)
        text = f"{prediction_label}"
        score_text = f"Fight: {fight_score:.3f} | NonFight: {nonfight_score:.3f}"
        window_text = f"Window: {current_window_idx + 1}/{len(window_results)}"
        
        # 예측 결과에 따른 색상
        color = (0, 255, 0) if prediction == 1 else (0, 255, 255)  # Fight: 녹색, NonFight: 노란색
        
        # 텍스트 그리기 (배경 없이)
        cv2.putText(overlay_image, text, (x, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)
        cv2.putText(overlay_image, score_text, (x, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7, self.text_color, self.thickness)
        cv2.putText(overlay_image, window_text, (x, y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, self.text_color, self.thickness)
        
        return overlay_image
    
    def add_prediction_text(self, image: np.ndarray, prediction_result: Dict, 
                           position: Tuple[int, int] = (30, 50)) -> np.ndarray:
        """
        이미지에 예측 결과 텍스트 추가 (최종 결과용 - 하위 호환성)
        
        Args:
            image: 입력 이미지
            prediction_result: 예측 결과 딕셔너리
            position: 텍스트 위치 (x, y)
            
        Returns:
            텍스트가 추가된 이미지
        """
        overlay_image = image.copy()
        x, y = position
        
        # 예측 라벨
        prediction_label = prediction_result.get('prediction_label', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        
        # 배경 박스 없이 텍스트만 표시
        text = f"Final: {prediction_label}"
        conf_text = f"Confidence: {confidence:.3f}"
        
        # 예측 결과에 따른 색상 선택
        color = (0, 255, 0) if prediction_label == 'Fight' else (0, 255, 255)  # Fight: 녹색, NonFight: 노란색
        
        # 텍스트 그리기
        cv2.putText(overlay_image, text, (x, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)
        cv2.putText(overlay_image, conf_text, (x, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, self.text_color, self.thickness)
        
        return overlay_image
    
    def add_frame_info(self, image: np.ndarray, frame_idx: int, total_frames: int,
                      position: Tuple[int, int] = None) -> np.ndarray:
        """
        이미지에 프레임 정보 추가
        
        Args:
            image: 입력 이미지
            frame_idx: 현재 프레임 인덱스
            total_frames: 총 프레임 수
            position: 텍스트 위치 (None이면 우상단)
            
        Returns:
            프레임 정보가 추가된 이미지
        """
        overlay_image = image.copy()
        
        if position is None:
            h, w = image.shape[:2]
            position = (w - 200, 30)
        
        x, y = position
        frame_text = f"Frame: {frame_idx + 1}/{total_frames}"
        
        cv2.putText(overlay_image, frame_text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7, self.text_color, self.thickness)
        
        return overlay_image
    
    def create_overlay_video(self, video_path: str, pose_results: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
                           prediction_result: Dict, output_path: str,
                           fps: Optional[float] = None, show_frame_info: bool = True) -> bool:
        """
        오버레이 비디오 생성
        
        Args:
            video_path: 원본 비디오 경로
            pose_results: 프레임별 포즈 결과
            prediction_result: 전체 비디오 예측 결과
            output_path: 출력 비디오 경로
            fps: 출력 비디오 FPS (None이면 원본과 동일)
            show_frame_info: 프레임 정보 표시 여부
            
        Returns:
            성공 여부
        """
        try:
            # 원본 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"비디오를 열 수 없습니다: {video_path}")
                return False
            
            # 비디오 정보 가져오기
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps is None:
                fps = original_fps
            
            # 출력 디렉토리 생성
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            
            # 비디오 작성기 초기화
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logger.info(f"오버레이 비디오 생성 시작: {osp.basename(video_path)} -> {osp.basename(output_path)}")
            logger.info(f"비디오 정보: {width}x{height}, {fps}fps, {total_frames}프레임")
            
            # Fight-우선 트래커 초기화 (전체 비디오에서 재사용)
            from config import INFERENCE_CONFIG
            num_person = INFERENCE_CONFIG.get('num_person', 1)
            fight_tracker = None
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                overlay_frame = frame.copy()
                
                # 포즈 오버레이 (해당 프레임에 포즈 데이터가 있는 경우)
                if frame_idx < len(pose_results):
                    keypoints, scores = pose_results[frame_idx]
                    if keypoints is not None and scores is not None:
                        
                        if len(keypoints.shape) == 3 and len(keypoints) > num_person:
                            # Fight-우선 트래커 초기화 (첫 번째 프레임에서만)
                            if fight_tracker is None:
                                from fight_tracker import FightPrioritizedTracker
                                fight_tracker = FightPrioritizedTracker(
                                    frame_width=width, 
                                    frame_height=height,
                                    composite_weights=INFERENCE_CONFIG.get('composite_weights')
                                )
                            
                            # 상위 N개 인물 선택 (Fight-우선 정렬 사용)
                            fight_order = fight_tracker.get_fight_prioritized_order(
                                [keypoints[i] for i in range(len(keypoints))],
                                [scores[i] for i in range(len(scores))]
                            )
                            selected_indices = fight_order[:num_person]
                            
                            # 선택된 인물들만 표시
                            selected_keypoints = keypoints[selected_indices]
                            selected_scores = scores[selected_indices]
                            overlay_frame = self.draw_keypoints(overlay_frame, selected_keypoints, selected_scores,
                                                              selected_indices=list(range(len(selected_indices))))
                        else:
                            # 모든 인물 표시 (num_person보다 적거나 같음)
                            overlay_frame = self.draw_keypoints(overlay_frame, keypoints, scores,
                                                              selected_indices=list(range(min(len(keypoints) if len(keypoints.shape) == 3 else 1, num_person))))
                
                # 윈도우별 실시간 예측 결과 텍스트 추가
                window_results = prediction_result.get('window_results', [])
                
                if window_results:
                    # 최적화된 구조 사용
                    overlay_frame = self.add_window_prediction_text(
                        overlay_frame, frame_idx, window_results=window_results, 
                        window_size=30, stride=15
                    )
                else:
                    # 윈도우 데이터가 없으면 최종 결과 표시
                    overlay_frame = self.add_prediction_text(overlay_frame, prediction_result)
                
                # 프레임 정보 추가
                if show_frame_info:
                    overlay_frame = self.add_frame_info(overlay_frame, frame_idx, total_frames)
                
                # 프레임 작성
                out.write(overlay_frame)
                
                frame_idx += 1
                
                # 진행 상황 출력
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"오버레이 진행: {progress:.1f}% ({frame_idx}/{total_frames})")
            
            # 리소스 해제
            cap.release()
            out.release()
            
            logger.info(f"오버레이 비디오 생성 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"오버레이 비디오 생성 실패: {e}")
            return False
    
    def create_comparison_video(self, video_path: str, pose_results: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
                              prediction_result: Dict, ground_truth_label: str,
                              output_path: str, fps: Optional[float] = None) -> bool:
        """
        예측과 실제 라벨을 비교하는 비디오 생성
        
        Args:
            video_path: 원본 비디오 경로
            pose_results: 프레임별 포즈 결과
            prediction_result: 예측 결과
            ground_truth_label: 실제 라벨
            output_path: 출력 비디오 경로
            fps: 출력 비디오 FPS
            
        Returns:
            성공 여부
        """
        try:
            # 예측 결과에 실제 라벨 정보 추가
            enhanced_result = prediction_result.copy()
            enhanced_result['ground_truth'] = ground_truth_label
            enhanced_result['is_correct'] = prediction_result['prediction_label'] == ground_truth_label
            
            # 기본 오버레이 비디오 생성
            return self.create_overlay_video(video_path, pose_results, enhanced_result, output_path, fps)
            
        except Exception as e:
            logger.error(f"비교 비디오 생성 실패: {e}")
            return False
    
    def add_enhanced_prediction_text(self, image: np.ndarray, prediction_result: Dict, 
                                   position: Tuple[int, int] = (30, 50)) -> np.ndarray:
        """
        향상된 예측 결과 텍스트 추가 (실제 라벨 비교 포함)
        
        Args:
            image: 입력 이미지
            prediction_result: 예측 결과 (ground_truth, is_correct 포함)
            position: 텍스트 위치
            
        Returns:
            텍스트가 추가된 이미지
        """
        overlay_image = image.copy()
        x, y = position
        
        prediction_label = prediction_result.get('prediction_label', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        ground_truth = prediction_result.get('ground_truth', 'Unknown')
        is_correct = prediction_result.get('is_correct', False)
        
        # 텍스트 구성
        pred_text = f"Prediction: {prediction_label}"
        conf_text = f"Confidence: {confidence:.3f}"
        gt_text = f"Ground Truth: {ground_truth}"
        result_text = f"Result: {'CORRECT' if is_correct else 'INCORRECT'}"
        
        texts = [pred_text, conf_text, gt_text, result_text]
        
        # 배경 박스 크기 계산
        max_w = 0
        total_h = 0
        line_heights = []
        
        for text in texts:
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, self.thickness)
            max_w = max(max_w, w)
            line_heights.append(h)
            total_h += h + 5
        
        # 반투명 배경 박스
        overlay = overlay_image.copy()
        cv2.rectangle(overlay, (x-10, y-10), (x + max_w + 20, y + total_h + 10), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, overlay_image, 1 - alpha, 0, overlay_image)
        
        # 텍스트 그리기
        current_y = y
        for i, text in enumerate(texts):
            if i == 0:  # Prediction
                color = (0, 255, 0) if prediction_label == 'Fight' else (0, 255, 255)
            elif i == 3:  # Result
                color = (0, 255, 0) if is_correct else (0, 0, 255)
            else:
                color = self.text_color
            
            cv2.putText(overlay_image, text, (x, current_y + line_heights[i]), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, color, self.thickness)
            current_y += line_heights[i] + 5
        
        return overlay_image
    
    def batch_create_overlay_videos(self, video_results: List[Dict], output_dir: str,
                                  show_progress: bool = True) -> List[str]:
        """
        배치로 오버레이 비디오 생성
        
        Args:
            video_results: 비디오별 결과 리스트
            output_dir: 출력 디렉토리
            show_progress: 진행 상황 표시 여부
            
        Returns:
            생성된 비디오 경로 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_videos = []
        
        for i, result in enumerate(video_results):
            try:
                video_path = result['video_path']
                pose_results = result.get('pose_results', [])
                prediction_result = result.get('prediction_result', {})
                
                video_name = osp.splitext(osp.basename(video_path))[0]
                output_path = osp.join(output_dir, f"{video_name}_overlay.mp4")
                
                if show_progress:
                    logger.info(f"오버레이 생성 ({i+1}/{len(video_results)}): {video_name}")
                
                success = self.create_overlay_video(video_path, pose_results, prediction_result, output_path)
                
                if success:
                    generated_videos.append(output_path)
                else:
                    logger.warning(f"오버레이 생성 실패: {video_name}")
                    
            except Exception as e:
                logger.error(f"배치 오버레이 생성 실패: {e}")
        
        logger.info(f"배치 오버레이 생성 완료: {len(generated_videos)}/{len(video_results)}개 성공")
        return generated_videos