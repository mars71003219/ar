"""
Inference 결과 시각화 도구

실시간 추론 결과를 원본 비디오에 오버레이하여 시각화합니다.
- 포즈 추정 키포인트 표시
- 윈도우별 분류 결과 표시
- 상위 정렬 객체는 빨간색 박스, 나머지는 파란색 박스
- 트래킹 ID 및 정렬 순위 표시
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from ..utils.import_utils import safe_import_data_structure
except ImportError:
    try:
        from utils.import_utils import safe_import_data_structure
    except ImportError:
        def safe_import_data_structure():
            try:
                from utils.data_structure import PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult
                return PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult
            except ImportError:
                from ..utils.data_structure import PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult
                return PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult

try:
    from .pose_visualizer import PoseVisualizer
except ImportError:
    from visualization.pose_visualizer import PoseVisualizer

PersonPose, FramePoses, ClassificationResult, AnnotationData, PipelineResult = safe_import_data_structure()


class InferenceVisualizer:
    """Inference 결과 시각화 클래스"""
    
    def __init__(self, max_persons: int = 4):
        """
        Args:
            max_persons: ST-GCN++에 입력되는 최대 person 수
        """
        self.max_persons = max_persons
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
        
        # 기준 해상도 (640x360)
        self.base_width = 640
        self.base_height = 360
        self.fight_color = (0, 0, 255)  # 빨간색 (Fight 감지)
        self.nonfight_color = (0, 255, 0)  # 초록색 (NonFight)
    
    def _get_adaptive_font_scale(self, image_width: int, image_height: int, base_scale: float = 0.4) -> float:
        """해상도에 비례한 폰트 크기 계산"""
        # 가로세로 비율을 모두 고려한 스케일링 팩터
        width_scale = image_width / self.base_width
        height_scale = image_height / self.base_height
        # 작은 쪽을 기준으로 스케일링 (너무 크거나 작아지지 않게)
        scale_factor = min(width_scale, height_scale)
        # 최소/최대 크기 제한
        scale_factor = max(0.3, min(2.0, scale_factor))
        return base_scale * scale_factor
    
    def _get_adaptive_thickness(self, image_width: int, image_height: int, base_thickness: int = 1) -> int:
        """해상도에 비례한 두께 계산"""
        width_scale = image_width / self.base_width
        height_scale = image_height / self.base_height
        scale_factor = min(width_scale, height_scale)
        # 최소 1, 최대 3
        thickness = max(1, min(3, int(base_thickness * scale_factor + 0.5)))
        return thickness
    
    def visualize_inference_results(self, 
                                   input_video_path: str,
                                   classification_results: List[Dict[str, Any]],
                                   output_video_path: str,
                                   frame_poses_results: Optional[List[FramePoses]] = None,
                                   rtmo_poses_results: Optional[List[FramePoses]] = None,
                                   window_size: int = 100,
                                   stride: int = 50) -> bool:
        """
        Inference 결과 시각화
        
        Args:
            input_video_path: 원본 비디오 경로
            classification_results: 분류 결과 리스트
            output_video_path: 출력 비디오 경로
            frame_poses_results: 프레임별 포즈 데이터 (트래킹 완료, 바운딩박스용)
            rtmo_poses_results: RTMO 원본 포즈 데이터 (키포인트 표시용)
            window_size: 윈도우 크기
            stride: 스트라이드 간격
            
        Returns:
            성공 여부
        """
        try:
            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {input_video_path}")
                return False
            
            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logging.info(f"Video info: {width}x{height}, {fps:.1f}FPS, {total_frames} frames")
            
            # 비디오 라이터 초기화
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # 분류 결과를 프레임별로 매핑
            frame_classifications = self._map_classifications_to_frames(
                classification_results, window_size, stride, total_frames
            )
            
            # 포즈 데이터를 프레임별로 매핑
            frame_poses_mapping = {}
            if frame_poses_results:
                for frame_poses in frame_poses_results:
                    frame_poses_mapping[frame_poses.frame_idx] = frame_poses
                
                frame_keys = list(frame_poses_mapping.keys())
                logging.info(f"Frame poses mapping: {len(frame_poses_mapping)} frames (range: {min(frame_keys) if frame_keys else 'N/A'}-{max(frame_keys) if frame_keys else 'N/A'})")
                logging.info(f"First 10 frame_poses indices: {sorted(frame_keys)[:10]}")
            
            # RTMO 원본 포즈 데이터를 프레임별로 매핑 (키포인트 표시용)
            rtmo_poses_mapping = {}
            if rtmo_poses_results:
                for rtmo_poses in rtmo_poses_results:
                    rtmo_poses_mapping[rtmo_poses.frame_idx] = rtmo_poses
                
                rtmo_keys = list(rtmo_poses_mapping.keys())
                logging.info(f"RTMO poses mapping: {len(rtmo_poses_mapping)} frames (range: {min(rtmo_keys) if rtmo_keys else 'N/A'}-{max(rtmo_keys) if rtmo_keys else 'N/A'})")
                logging.info(f"First 10 RTMO indices: {sorted(rtmo_keys)[:10]}")
                
                # 프레임 인덱스 오프셋 감지 및 보정
                if rtmo_keys and min(rtmo_keys) > 0:
                    offset = min(rtmo_keys)
                    logging.info(f"Detected frame index offset: {offset}, applying correction")
                    # 오프셋 보정: 모든 키를 0부터 시작하도록 조정
                    corrected_rtmo_mapping = {}
                    for original_idx, poses in rtmo_poses_mapping.items():
                        corrected_rtmo_mapping[original_idx - offset] = poses
                    rtmo_poses_mapping = corrected_rtmo_mapping
                    
                if frame_poses_mapping and min(frame_poses_mapping.keys()) > 0:
                    offset = min(frame_poses_mapping.keys())
                    logging.info(f"Detected frame_poses index offset: {offset}, applying correction")
                    # 오프셋 보정: 모든 키를 0부터 시작하도록 조정
                    corrected_frame_mapping = {}
                    for original_idx, poses in frame_poses_mapping.items():
                        corrected_frame_mapping[original_idx - offset] = poses
                    frame_poses_mapping = corrected_frame_mapping
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 현재 프레임의 분류 결과와 포즈 데이터 가져오기
                current_classification = frame_classifications.get(frame_idx)
                current_frame_poses = frame_poses_mapping.get(frame_idx)  # 트래킹 완료 데이터
                current_rtmo_poses = rtmo_poses_mapping.get(frame_idx)    # RTMO 원본 데이터
                
                # 프레임 시각화
                vis_frame = self._visualize_frame_with_classification(
                    frame, frame_idx, current_classification, current_frame_poses, current_rtmo_poses
                )
                
                writer.write(vis_frame)
                frame_idx += 1
                
                # 진행률 로깅
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logging.info(f"Visualization progress: {progress:.1f}%")
            
            cap.release()
            writer.release()
            
            logging.info(f"Visualization completed: {output_video_path}")
            return True
            
        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}")
            return False
    
    def _map_classifications_to_frames(self, 
                                     classification_results: List[Dict[str, Any]],
                                     window_size: int,
                                     stride: int,
                                     total_frames: int) -> Dict[int, Dict[str, Any]]:
        """분류 결과를 프레임별로 매핑"""
        frame_classifications = {}
        
        for result in classification_results:
            window_start = result.get('window_start', 0)
            window_end = result.get('window_end', window_start + window_size)
            
            # 윈도우 내의 모든 프레임에 분류 결과 적용
            # 단, 스트라이드 간격을 고려하여 새로운 윈도우 정보만 사용
            for frame_idx in range(window_start, min(window_end, total_frames)):
                # 이전 윈도우 결과가 없거나, 새로운 윈도우 영역인 경우만 업데이트
                if (frame_idx not in frame_classifications or 
                    frame_idx >= window_start + stride):
                    frame_classifications[frame_idx] = result
        
        return frame_classifications
    
    def _visualize_frame_with_classification(self,
                                           frame: np.ndarray,
                                           frame_idx: int,
                                           classification: Optional[Dict[str, Any]],
                                           frame_poses: Optional[FramePoses] = None,
                                           rtmo_poses: Optional[FramePoses] = None) -> np.ndarray:
        """
        프레임에 분류 결과 및 포즈 오버레이 (사용자 요구사항 기반)
        
        오버레이 순서:
        1. RTMO 원본 포즈 추정 결과 기반 관절 포인트 표시 (순수 키포인트 데이터)
        2. 트래킹 완료 데이터 기반 바운딩박스와 텍스트: 트래킹ID + 복합점수 기반 정렬순서 (기본 파란색 배경)
        3. num_persons까지 포함되는 상위 정렬 객체는 빨간색 배경으로 구분
        """
        vis_frame = frame.copy()
        
        # 1. RTMO 원본 포즈 추정 결과 기반 관절 포인트 표시 (모든 탐지된 사람)
        # RTMO 원본 데이터가 없으면 트래킹 완료 데이터에서라도 키포인트 표시
        poses_for_keypoints = rtmo_poses if rtmo_poses and rtmo_poses.persons else frame_poses
        
        # 디버깅: 데이터 상태 확인
        rtmo_count = len(rtmo_poses.persons) if rtmo_poses and rtmo_poses.persons else 0
        frame_count = len(frame_poses.persons) if frame_poses and frame_poses.persons else 0
        
        if frame_idx % 30 == 0:  # 30프레임마다 로깅
            logging.debug(f"Frame {frame_idx}: RTMO={rtmo_count}, Tracked={frame_count}")
        
        if poses_for_keypoints and poses_for_keypoints.persons:
            # 원본 데이터에서 키포인트만 추출하여 표시 (점수 기반 정렬 없이)
            persons_for_keypoints = [(person, 0) for person in poses_for_keypoints.persons]
            vis_frame = self._draw_rtmo_pose_keypoints(vis_frame, persons_for_keypoints)
        else:
            # 데이터가 없는 경우 로깅 (더 적게 출력)
            if frame_idx % 150 == 0:  # 150프레임(5초)마다만 로깅
                logging.debug(f"Frame {frame_idx}: No pose data available for keypoints")
        
        # 2. 트래킹 완료 데이터 기반 바운딩박스와 텍스트 정보 표시
        if frame_poses and frame_poses.persons:
            # 복합점수 기반 정렬 (높은 점수 순)
            sorted_persons = self._get_sorted_persons_by_score(frame_poses.persons)
            
            # 바운딩박스와 텍스트 정보 표시 (트래킹ID + 순위번호)
            vis_frame = self._draw_boxes_with_ranking_info(vis_frame, sorted_persons)
        
        # 기본 프레임 정보 표시
        self._draw_frame_info(vis_frame, frame_idx, classification)
        
        # 듀얼 서비스 UI 표시
        self._draw_dual_service_ui(vis_frame, frame_idx, classification)
        
        # 분류 결과가 있는 경우 추가 정보 표시
        if classification:
            self._draw_classification_overlay(vis_frame, classification)
        
        return vis_frame
    
    def _get_sorted_persons_by_score(self, persons: List[PersonPose]) -> List[Tuple[PersonPose, int]]:
        """복합점수 기반으로 사람들을 정렬하고 순위 정보 추가"""
        persons_with_score = []
        for person in persons:
            if hasattr(person, 'composite_score') and person.composite_score is not None:
                score = person.composite_score
            elif hasattr(person, 'bbox_score') and person.bbox_score is not None:
                score = person.bbox_score
            elif hasattr(person, 'score') and person.score is not None:
                score = person.score
            else:
                score = 0.0
            
            persons_with_score.append((person, score))
        
        # 점수 기준 내림차순 정렬
        persons_with_score.sort(key=lambda x: x[1], reverse=True)
        
        # 순위 정보 추가 (1부터 시작)
        return [(person, rank + 1) for rank, (person, score) in enumerate(persons_with_score)]
    
    def _draw_rtmo_pose_keypoints(self, image: np.ndarray, sorted_persons: List[Tuple[PersonPose, int]]) -> np.ndarray:
        """1. RTMO 포즈 추정 결과 기반 관절 포인트 표시"""
        vis_image = image.copy()
        
        # COCO 17 키포인트 연결 정보 (skeleton)
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 팔
            [5, 11], [6, 12], [5, 6],  # 어깨
            [5, 7], [6, 8], [7, 9], [8, 10],  # 팔 연결
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # 머리
            [0, 5], [0, 6]  # 머리-어깨 연결
        ]
        
        # 키포인트 색상 (COCO 색상 팔레트)
        pose_kpt_color = [
            [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],  # 머리 (하늘색)
            [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],  # 어깨, 팔꿈치 (초록, 주황)
            [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],  # 손목, 골반 (초록, 주황)
            [255, 0, 255], [255, 0, 255], [255, 0, 255], [255, 0, 255]  # 다리 (자홍색)
        ]
        
        # 스켈레톤 색상
        pose_limb_color = [[0, 255, 0]] * len(skeleton)
        
        # 각 person의 키포인트 그리기
        for person, rank in sorted_persons:
            if len(person.keypoints) < 17:
                continue
                
            # 스켈레톤 연결선 그리기
            for i, (start_idx, end_idx) in enumerate(skeleton):
                if start_idx >= len(person.keypoints) or end_idx >= len(person.keypoints):
                    continue
                    
                start_kpt = person.keypoints[start_idx]
                end_kpt = person.keypoints[end_idx]
                
                # 두 키포인트가 모두 유효한 경우에만 연결선 그리기
                if (len(start_kpt) >= 3 and len(end_kpt) >= 3 and 
                    start_kpt[2] > 0.3 and end_kpt[2] > 0.3):
                    
                    start_point = (int(start_kpt[0]), int(start_kpt[1]))
                    end_point = (int(end_kpt[0]), int(end_kpt[1]))
                    
                    # 연결선 그리기
                    cv2.line(vis_image, start_point, end_point, 
                            pose_limb_color[i % len(pose_limb_color)], 2)
            
            # 키포인트 원 그리기
            for i, keypoint in enumerate(person.keypoints):
                if len(keypoint) >= 3 and keypoint[2] > 0.3:  # 신뢰도 임계값
                    x, y = int(keypoint[0]), int(keypoint[1])
                    
                    # 키포인트 원 그리기
                    color = pose_kpt_color[i % len(pose_kpt_color)]
                    cv2.circle(vis_image, (x, y), 4, color, -1)  # 채워진 원
                    cv2.circle(vis_image, (x, y), 4, (255, 255, 255), 1)  # 테두리
        
        return vis_image
    
    def _draw_boxes_with_ranking_info(self, image: np.ndarray, sorted_persons: List[Tuple[PersonPose, int]]) -> np.ndarray:
        """2. 바운딩박스와 텍스트 정보 표시 (트래킹ID + 순위번호, 색상 구분)"""
        vis_image = image.copy()
        
        for person, rank in sorted_persons:
            if person.bbox is None or len(person.bbox) < 4:
                continue
            
            # 바운딩박스 좌표
            x1, y1, x2, y2 = person.bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # num_persons 범위 내 여부에 따른 색상 결정
            is_top_ranked = rank <= self.max_persons
            box_color = self.top_ranked_color if is_top_ranked else self.normal_color  # 빨간색 또는 파란색
            text_bg_color = self.top_ranked_color if is_top_ranked else self.normal_color
            text_color = (255, 255, 255)  # 흰색 텍스트
            
            # 바운딩박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, 2)
            
            # 텍스트 정보 생성 (트래킹ID + 순위번호)
            text_parts = []
            if hasattr(person, 'track_id') and person.track_id is not None:
                text_parts.append(f"ID:{person.track_id}")
            text_parts.append(f"#{rank}")
            
            if hasattr(person, 'composite_score') and person.composite_score is not None:
                text_parts.append(f"({person.composite_score:.2f})")
            elif hasattr(person, 'bbox_score') and person.bbox_score is not None:
                text_parts.append(f"({person.bbox_score:.2f})")
            
            text = " ".join(text_parts)
            
            # 텍스트 크기 계산 (해상도에 비례)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self._get_adaptive_font_scale(vis_image.shape[1], vis_image.shape[0], 0.5)  # 0.7 -> 0.5 베이스
            thickness = self._get_adaptive_thickness(vis_image.shape[1], vis_image.shape[0], 1)  # 2 -> 1 베이스
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # 텍스트 배경 박스 (박스 상단)
            bg_x1, bg_y1 = x1, y1 - text_size[1] - 10
            bg_x2, bg_y2 = x1 + text_size[0] + 10, y1
            
            # 화면 경계 확인
            if bg_y1 < 0:
                bg_y1, bg_y2 = y2, y2 + text_size[1] + 10
            
            # 텍스트 배경 그리기
            cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), text_bg_color, -1)
            cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
            
            # 텍스트 그리기
            text_x = x1 + 5
            text_y = bg_y1 + text_size[1] + 5 if bg_y1 >= 0 else y2 + text_size[1] + 5
            cv2.putText(vis_image, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        return vis_image
    
    def _draw_frame_info(self, 
                        image: np.ndarray, 
                        frame_idx: int,
                        classification: Optional[Dict[str, Any]]):
        """기본 프레임 정보 그리기 - 듀얼 서비스 지원"""
        # 프레임 번호
        frame_text = f"Frame: {frame_idx}"
        
        # 듀얼 서비스 정보 표시
        if classification:
            # Fight 및 Falldown 정보 준비
            fight_class = classification.get('fight_predicted_class', 'Unknown')
            fight_confidence = classification.get('fight_confidence', 0.0)
            falldown_class = classification.get('falldown_predicted_class', 'Unknown')
            falldown_confidence = classification.get('falldown_confidence', 0.0)
            
            # 기존 단일 서비스 호환성
            if fight_class == 'Unknown' and falldown_class == 'Unknown':
                pred_class = classification.get('predicted_class', 'Unknown')
                confidence = classification.get('confidence', 0.0)
                
                # 단일 서비스 표시
                class_text = f"{pred_class} ({confidence:.3f})"
                
                if pred_class == 'Fight':
                    text_color = self.fight_color
                    bg_color = (0, 0, 128)
                else:
                    text_color = self.nonfight_color
                    bg_color = (0, 64, 0)
            else:
                # 듀얼 서비스 표시
                fight_text = f"Fight: {fight_class} ({fight_confidence:.3f})"
                falldown_text = f"Falldown: {falldown_class} ({falldown_confidence:.3f})"
                class_text = f"{fight_text} | {falldown_text}"
                
                # 두 서비스 중 하나라도 이벤트가 있으면 빨간색
                if fight_class == 'Fight' or falldown_class == 'Falldown':
                    text_color = self.fight_color
                    bg_color = (0, 0, 128)
                else:
                    text_color = self.nonfight_color
                    bg_color = (0, 64, 0)
        else:
            class_text = "No Classification"
            text_color = (255, 255, 255)
            bg_color = (64, 64, 64)
        
        # 상단 정보 박스
        info_text = f"{frame_text} | {class_text}"
        # 해상도에 비례한 폰트 크기와 두께
        font_scale = self._get_adaptive_font_scale(image.shape[1], image.shape[0], 0.3)  # 0.4 -> 0.3 베이스
        thickness = self._get_adaptive_thickness(image.shape[1], image.shape[0], 1)
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # 반투명 검은색 배경 박스 (최대 가장자리로 이동)
        x_start = 2
        y_start = 5
        overlay = image.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + text_size[0] + 10, y_start + text_size[1] + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)  # 검은색 반투명 배경
        cv2.rectangle(image, (x_start, y_start), (x_start + text_size[0] + 10, y_start + text_size[1] + 10), text_color, thickness)
        
        # 텍스트
        cv2.putText(image, info_text, (x_start + 3, y_start + text_size[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    def _draw_dual_service_ui(self, 
                             image: np.ndarray, 
                             frame_idx: int,
                             classification: Optional[Dict[str, Any]]):
        """듀얼 서비스 UI 표시 - 좌측 상단, 좌측 중앙, 우측 상단"""
        
        try:
            # 안전성 확인
            if classification is None:
                # 기본 classification 딕셔너리 생성
                classification = {
                    'window_number': 0,
                    'fight_predicted_class': 'Normal',
                    'fight_confidence': 0.0,
                    'falldown_predicted_class': 'Normal', 
                    'falldown_confidence': 0.0,
                    'pose_fps': 30.0,
                    'fight_cls_fps': 0.0,
                    'falldown_cls_fps': 0.0,
                    'show_classification': True,
                    'show_keypoints': True,
                    'show_tracking_ids': True,
                    'predicted_class': 'Normal',
                    'confidence': 0.0
                }
            
            # 1. 좌측 상단 정보 수정 (Window 번호와 점수)
            self._draw_left_top_info(image, classification)
            
            # 2. 좌측 중앙 FPS 정보
            self._draw_left_center_fps(image, classification)
            
            # 3. 우측 상단 이벤트 상태
            self._draw_right_top_events(image, classification)
        except Exception as e:
            logging.error(f"Error in _draw_dual_service_ui: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_left_top_info(self, image: np.ndarray, classification: Optional[Dict[str, Any]]):
        """좌측 상단 정보: Window {number}: Fight {score}, Window {number}: Falldown {score}"""
        y_start = 10  # 더욱 위쪽으로 이동
        x_start = 2   # 최대 가장자리로 이동
        
        if classification:
            # 윈도우 번호 추출 (메타데이터에서)
            window_number = classification.get('window_number', 1)
            
            # Fight 정보
            fight_confidence = classification.get('fight_confidence', 0.0)
            fight_text = f"Window {window_number}: Fight {fight_confidence:.3f}"
            
            # Falldown 정보  
            falldown_confidence = classification.get('falldown_confidence', 0.0)
            falldown_text = f"Window {window_number}: Falldown {falldown_confidence:.3f}"
        else:
            fight_text = "Window -: Fight 0.000"
            falldown_text = "Window -: Falldown 0.000"
        
        # 해상도에 비례한 폰트 크기와 두께
        font_scale = self._get_adaptive_font_scale(image.shape[1], image.shape[0], 0.3)  # 0.4 -> 0.3 베이스
        thickness = self._get_adaptive_thickness(image.shape[1], image.shape[0], 1)
        
        # 두 텍스트 중 더 긴 것을 기준으로 배경 폭 결정 (정렬을 위해)
        fight_size = cv2.getTextSize(fight_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        falldown_size = cv2.getTextSize(falldown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        max_width = max(fight_size[0], falldown_size[0])
        
        # Fight 정보 표시 (검은색 배경)
        overlay = image.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + max_width + 10, y_start + fight_size[1] + 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)  # 검은색 반투명 배경
        cv2.putText(image, fight_text, (x_start + 3, y_start + fight_size[1] + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Falldown 정보 표시 (검은색 배경)
        y_start += fight_size[1] + 12
        overlay = image.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + max_width + 10, y_start + falldown_size[1] + 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)  # 검은색 반투명 배경
        cv2.putText(image, falldown_text, (x_start + 3, y_start + falldown_size[1] + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    def _draw_left_center_fps(self, image: np.ndarray, classification: Optional[Dict[str, Any]]):
        """좌측 중앙 FPS 정보: pose: fps, fight cls: fps, falldown cls: fps"""
        height, width = image.shape[:2]
        # y_center = height // 2
        
        # FPS 정보 준비 (실제 FPS는 파이프라인에서 계산되어야 함)
        if classification:
            pose_fps = classification.get('pose_fps', 30.0)
            fight_fps = classification.get('fight_cls_fps', 10.0)
            falldown_fps = classification.get('falldown_cls_fps', 10.0)
        else:
            pose_fps = 30.0
            fight_fps = 10.0
            falldown_fps = 10.0
        
        fps_lines = [
            f"pose: {pose_fps:.1f}fps",
            f"fight cls: {fight_fps:.1f}fps", 
            f"falldown cls: {falldown_fps:.1f}fps"
        ]
        
        x_start = 2  # 최대 가장자리로 이동
        y_pos = 60
        
        # 해상도에 비례한 폰트 크기와 두께
        font_scale = self._get_adaptive_font_scale(image.shape[1], image.shape[0], 0.3)  # 0.4 -> 0.3 베이스
        thickness = self._get_adaptive_thickness(image.shape[1], image.shape[0], 1)
        
        # 모든 텍스트 중 가장 긴 것을 기준으로 배경 폭 결정 (정렬을 위해)
        max_width = 0
        for line in fps_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])
        
        for line in fps_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            # 반투명 검은색 배경 생성
            overlay = image.copy()
            cv2.rectangle(overlay, (x_start, y_pos - 4), (x_start + max_width + 8, y_pos + text_size[1] + 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)  # 검은색 반투명 배경
            cv2.putText(image, line, (x_start + 3, y_pos + text_size[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            y_pos += text_size[1] + 8  # 간격도 축소
    
    def _draw_right_top_events(self, image: np.ndarray, classification: Optional[Dict[str, Any]]):
        """우측 상단 이벤트 상태: Fight Event: {Fight or Normal}, Falldown Event: {Falldown or Normal}"""
        height, width = image.shape[:2]
        
        if classification:
            # (이벤트 상태 결정 로직은 동일)
            fight_class = classification.get('fight_predicted_class', 'Normal')
            falldown_class = classification.get('falldown_predicted_class', 'Normal')
            
            if fight_class == 'Unknown' and falldown_class == 'Unknown':
                pred_class = classification.get('predicted_class', 'Normal')
                if pred_class == 'Fight':
                    fight_event = 'Fight'
                    falldown_event = 'Normal'
                else:
                    fight_event = 'Normal'
                    falldown_event = 'Normal'
            else:
                fight_event = fight_class if fight_class != 'Unknown' else 'Normal'
                falldown_event = falldown_class if falldown_class != 'Unknown' else 'Normal'
        else:
            fight_event = 'Normal'
            falldown_event = 'Normal'
        
        event_lines = [
            f"Fight Event: {fight_event}",
            f"Falldown Event: {falldown_event}"
        ]
        
        # 해상도에 비례한 폰트 크기와 두께
        font_scale = self._get_adaptive_font_scale(image.shape[1], image.shape[0], 0.3)  # 0.4 -> 0.3 베이스
        thickness = self._get_adaptive_thickness(image.shape[1], image.shape[0], 1)
        
        # --- 텍스트 정렬을 위한 수정 부분 ---
        # 1. 모든 텍스트 라인의 너비와 높이를 미리 계산한다.
        text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in event_lines]
        
        # 2. 가장 긴 텍스트의 너비를 찾는다.
        max_width = max(size[0] for size in text_sizes)
        
        # 3. 가장 긴 텍스트를 기준으로 모든 라인에 적용될 고정된 시작 x 좌표를 계산한다.
        x_start = width - max_width - 10
        # --- 수정 끝 ---
        
        y_start = 10
        for i, line in enumerate(event_lines):
            # 현재 라인의 높이 정보를 가져온다.
            current_text_height = text_sizes[i][1]
            y_pos = y_start + i * (current_text_height + 12)
            
            # 이벤트에 따른 텍스트 색상 구분
            if 'Fight Event' in line and fight_event == 'Fight':
                text_color = (0, 0, 255)  # 빨간색 텍스트
            elif 'Falldown Event' in line and falldown_event == 'Falldown':
                text_color = (0, 255, 255)  # 노란색 텍스트
            else:
                text_color = (255, 255, 255)  # 흰색 텍스트
            
            # 반투명 검은색 배경 생성 (배경 박스도 최대 너비 기준으로 통일)
            overlay = image.copy()
            cv2.rectangle(overlay, (x_start - 8, y_pos - 4), 
                        (x_start + max_width + 8, y_pos + current_text_height + 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            
            # 고정된 x_start 좌표를 사용하여 텍스트를 그린다.
            cv2.putText(image, line, (x_start, y_pos + current_text_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    def _draw_classification_overlay(self, 
                                   image: np.ndarray,
                                   classification: Dict[str, Any]):
        """분류 결과 오버레이 그리기"""
        # 윈도우 정보
        window_start = classification.get('window_start', 0)
        window_end = classification.get('window_end', 0)
        pred_class = classification.get('predicted_class', 'Unknown')
        confidence = classification.get('confidence', 0.0)
        probabilities = classification.get('probabilities', [])
        
        # 하단 분류 정보 박스
        info_lines = [
            f"Window: {window_start}-{window_end}",
            f"Prediction: {pred_class}",
            f"Confidence: {confidence:.4f}"
        ]
        
        if len(probabilities) >= 2:
            info_lines.append(f"NonFight: {probabilities[0]:.3f}")
            info_lines.append(f"Fight: {probabilities[1]:.3f}")
        
        # 정보 박스 그리기
        y_start = image.shape[0] - 150
        box_height = len(info_lines) * 25 + 20
        
        # 배경
        if pred_class == 'Fight':
            bg_color = (0, 0, 128)  # 어두운 빨간색
            text_color = (255, 255, 255)
        else:
            bg_color = (0, 64, 0)  # 어두운 초록색
            text_color = (255, 255, 255)
        
        cv2.rectangle(image, (10, y_start), (350, y_start + box_height), bg_color, -1)
        cv2.rectangle(image, (10, y_start), (350, y_start + box_height), text_color, 2)
        
        # 텍스트 그리기 (해상도에 비례한 폰트)
        font_scale = self._get_adaptive_font_scale(image.shape[1], image.shape[0], 0.4)  # 0.6 -> 0.4 베이스
        thickness = self._get_adaptive_thickness(image.shape[1], image.shape[0], 1)
        for i, line in enumerate(info_lines):
            y_pos = y_start + 20 + i * 25
            cv2.putText(image, line, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        # 신뢰도 막대 그래프
        if len(probabilities) >= 2:
            self._draw_confidence_bar(image, probabilities)
    
    def _draw_confidence_bar(self, image: np.ndarray, probabilities: List[float]):
        """신뢰도 막대 그래프 그리기"""
        if len(probabilities) < 2:
            return
        
        # 막대 그래프 영역
        bar_x = image.shape[1] - 200
        bar_y = 50
        bar_width = 150
        bar_height = 20
        
        # NonFight 막대
        nonfight_prob = probabilities[0]
        nonfight_width = int(bar_width * nonfight_prob)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + nonfight_width, bar_y + bar_height), 
                     self.nonfight_color, -1)
        # 해상도에 비례한 폰트 크기와 두께
        font_scale = self._get_adaptive_font_scale(image.shape[1], image.shape[0], 0.35)  # 0.5 -> 0.35 베이스
        thickness = self._get_adaptive_thickness(image.shape[1], image.shape[0], 1)
        
        cv2.putText(image, f"NonFight: {nonfight_prob:.3f}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.nonfight_color, thickness)
        
        # Fight 막대
        fight_prob = probabilities[1]
        fight_width = int(bar_width * fight_prob)
        cv2.rectangle(image, (bar_x, bar_y + 30), (bar_x + fight_width, bar_y + 30 + bar_height), 
                     self.fight_color, -1)
        cv2.putText(image, f"Fight: {fight_prob:.3f}", (bar_x, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.fight_color, thickness)
        
        # 테두리
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 1)
        cv2.rectangle(image, (bar_x, bar_y + 30), (bar_x + bar_width, bar_y + 30 + bar_height), 
                     (255, 255, 255), 1)
    
    def visualize_frame(self, 
                       frame: np.ndarray, 
                       frame_poses: Optional[FramePoses] = None, 
                       classification: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        단일 프레임 시각화 (실시간 처리용)
        
        Args:
            frame: 원본 프레임
            frame_poses: 포즈 데이터 (선택적)
            classification: 분류 결과 (선택적)
            
        Returns:
            시각화된 프레임
        """
        vis_frame = frame.copy()
        
        # 포즈 데이터가 있으면 시각화
        if frame_poses and frame_poses.persons:
            sorted_persons = self._get_sorted_persons_by_score(frame_poses.persons)
            
            # 키포인트 그리기
            vis_frame = self._draw_rtmo_pose_keypoints(vis_frame, sorted_persons)
            
            # 바운딩 박스와 ID 정보 그리기
            vis_frame = self._draw_boxes_with_ranking_info(vis_frame, sorted_persons)
        
        # 분류 결과가 있으면 오버레이 표시
        if classification:
            try:
                # 듀얼 서비스 UI 표시
                if any(key.startswith(('fight_', 'falldown_')) for key in classification.keys()):
                    self._draw_dual_service_ui(vis_frame, 0, classification)
                else:
                    # 단일 서비스 UI 표시
                    self._draw_classification_overlay(
                        vis_frame, 
                        classification.get('predicted_class', 'Unknown'),
                        classification.get('confidence', 0.0),
                        classification.get('probabilities', [1.0, 0.0])
                    )
            except Exception as e:
                # 시각화 에러가 발생해도 계속 진행
                import logging
                logging.warning(f"Visualization error: {e}")
                pass
        
        return vis_frame


def create_inference_visualization(input_video: str,
                                 classification_results: List[Dict[str, Any]], 
                                 output_path: str,
                                 frame_poses_results: Optional[List[FramePoses]] = None,
                                 rtmo_poses_results: Optional[List[FramePoses]] = None,
                                 window_size: int = 100,
                                 stride: int = 50) -> bool:
    """
    Inference 결과 시각화 생성 (편의 함수)
    
    Args:
        input_video: 입력 비디오 경로
        classification_results: 분류 결과 리스트
        output_path: 출력 비디오 경로
        frame_poses_results: 프레임별 포즈 데이터 (트래킹 완료)
        rtmo_poses_results: RTMO 원본 포즈 데이터 (키포인트 표시용)
        window_size: 윈도우 크기
        stride: 스트라이드 간격
        
    Returns:
        성공 여부
    """
    visualizer = InferenceVisualizer()
    return visualizer.visualize_inference_results(
        input_video_path=input_video,
        classification_results=classification_results,
        output_video_path=output_path,
        frame_poses_results=frame_poses_results,
        rtmo_poses_results=rtmo_poses_results,
        window_size=window_size,
        stride=stride
    )