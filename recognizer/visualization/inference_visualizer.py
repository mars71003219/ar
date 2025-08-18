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
        self.fight_color = (0, 0, 255)  # 빨간색 (Fight 감지)
        self.nonfight_color = (0, 255, 0)  # 초록색 (NonFight)
    
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
            
            # 텍스트 크기 계산
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
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
        """기본 프레임 정보 그리기"""
        # 프레임 번호
        frame_text = f"Frame: {frame_idx}"
        
        # 분류 결과 정보
        if classification:
            pred_class = classification.get('predicted_class', 'Unknown')
            confidence = classification.get('confidence', 0.0)
            class_text = f"{pred_class} ({confidence:.3f})"
            
            # 분류 결과에 따른 색상
            if pred_class == 'Fight':
                text_color = self.fight_color
                bg_color = (0, 0, 128)  # 어두운 빨간색
            else:
                text_color = self.nonfight_color
                bg_color = (0, 64, 0)  # 어두운 초록색
        else:
            class_text = "No Classification"
            text_color = (255, 255, 255)
            bg_color = (64, 64, 64)  # 회색
        
        # 상단 정보 박스
        info_text = f"{frame_text} | {class_text}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # 배경 박스
        cv2.rectangle(image, (10, 10), (text_size[0] + 30, text_size[1] + 30), bg_color, -1)
        cv2.rectangle(image, (10, 10), (text_size[0] + 30, text_size[1] + 30), text_color, 2)
        
        # 텍스트
        cv2.putText(image, info_text, (20, text_size[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
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
        
        # 텍스트 그리기
        for i, line in enumerate(info_lines):
            y_pos = y_start + 20 + i * 25
            cv2.putText(image, line, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
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
        cv2.putText(image, f"NonFight: {nonfight_prob:.3f}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.nonfight_color, 1)
        
        # Fight 막대
        fight_prob = probabilities[1]
        fight_width = int(bar_width * fight_prob)
        cv2.rectangle(image, (bar_x, bar_y + 30), (bar_x + fight_width, bar_y + 30 + bar_height), 
                     self.fight_color, -1)
        cv2.putText(image, f"Fight: {fight_prob:.3f}", (bar_x, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.fight_color, 1)
        
        # 테두리
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 1)
        cv2.rectangle(image, (bar_x, bar_y + 30), (bar_x + bar_width, bar_y + 30 + bar_height), 
                     (255, 255, 255), 1)


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