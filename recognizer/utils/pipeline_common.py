"""
파이프라인 공통 유틸리티
realtime, analysis, visualize 모드에서 공통으로 사용되는 기능들
"""

import logging
from typing import List, Dict, Any
import numpy as np
from .data_structure import FramePoses, PersonPose

logger = logging.getLogger(__name__)


class PipelineCommonUtils:
    """파이프라인 공통 유틸리티 클래스"""
    
    @staticmethod
    def convert_to_stgcn_format(frame_poses_list: List[FramePoses], max_persons: int = 4):
        """프레임 포즈 리스트를 ST-GCN 형식으로 변환"""
        T = len(frame_poses_list)
        M = max_persons
        V = 17  # COCO 키포인트 수
        C = 2   # x, y (confidence는 별도로 저장)
        
        keypoint = np.zeros((M, T, V, C), dtype=np.float32)  # MMAction2 표준: [M, T, V, C]
        keypoint_score = np.zeros((M, T, V), dtype=np.float32)
        
        # 실제 데이터 통계 수집
        person_count_per_frame = []
        valid_keypoint_count = 0
        total_keypoint_count = 0
        
        for t, frame_poses in enumerate(frame_poses_list):
            person_count_per_frame.append(len(frame_poses.persons))
            
            for m, person in enumerate(frame_poses.persons[:max_persons]):
                total_keypoint_count += 1
                
                if isinstance(person.keypoints, np.ndarray):
                    kpts = person.keypoints
                    if kpts.shape == (V, 3):  # [17, 3] 형태 (x, y, confidence)
                        keypoint[m, t, :, :] = kpts[:, :2]  # x, y만 사용
                        keypoint_score[m, t, :] = kpts[:, 2]  # confidence
                        valid_keypoint_count += 1
                    elif kpts.shape == (V, 2):  # [17, 2] 형태 (x, y만)
                        keypoint[m, t, :, :] = kpts
                        keypoint_score[m, t, :] = 1.0  # 기본 confidence
                        valid_keypoint_count += 1
                    elif len(kpts.flatten()) >= V * 2:
                        # 1D 배열인 경우
                        reshaped = kpts.flatten()[:V*3].reshape(-1, 3) if len(kpts.flatten()) >= V*3 else kpts.flatten()[:V*2].reshape(-1, 2)
                        if reshaped.shape[1] >= 2:
                            keypoint[m, t, :reshaped.shape[0], :] = reshaped[:, :2]
                            if reshaped.shape[1] >= 3:
                                keypoint_score[m, t, :reshaped.shape[0]] = reshaped[:, 2]
                            else:
                                keypoint_score[m, t, :reshaped.shape[0]] = 1.0
                            valid_keypoint_count += 1
        
        # 통계 로그 출력
        avg_person_count = np.mean(person_count_per_frame) if person_count_per_frame else 0
        max_person_count = max(person_count_per_frame) if person_count_per_frame else 0
        data_fill_ratio = valid_keypoint_count / total_keypoint_count if total_keypoint_count > 0 else 0
        
        logger.debug(f"Window conversion stats - Avg persons: {avg_person_count:.1f}, Max persons: {max_person_count}")
        logger.debug(f"Valid keypoints: {valid_keypoint_count}/{total_keypoint_count} ({data_fill_ratio:.3f})")
        
        return keypoint, keypoint_score
    
    @staticmethod
    def apply_composite_scores(frames: List[FramePoses], scorer) -> List[FramePoses]:
        """윈도우 프레임들에 복합점수를 적용"""
        if scorer and any(len(frame.persons) > 0 for frame in frames):
            person_scores = scorer.calculate_scores(frames)
            logger.debug(f"Calculated composite scores for {len(person_scores)} persons")
            
            # 복합점수를 각 프레임의 person에 적용
            scored_frames = []
            for frame in frames:
                scored_persons = []
                for person in frame.persons:
                    if person.track_id and person.track_id in person_scores:
                        # 복합점수 정보 추가
                        person.metadata = getattr(person, 'metadata', {})
                        person.metadata['composite_score'] = person_scores[person.track_id].composite_score
                        person.metadata['movement_score'] = person_scores[person.track_id].movement_score
                        person.metadata['interaction_score'] = person_scores[person.track_id].interaction_score
                    scored_persons.append(person)
                
                # 프레임 복사본 생성
                from .data_structure import FramePoses
                scored_frame = FramePoses(
                    frame_idx=frame.frame_idx,
                    persons=scored_persons,
                    timestamp=frame.timestamp,
                    image_shape=frame.image_shape,
                    metadata=frame.metadata
                )
                scored_frames.append(scored_frame)
            
            return scored_frames
        else:
            logger.debug("No persons to score, using original frames")
            return frames
    
    @staticmethod
    def count_unique_persons(poses: List[FramePoses]) -> int:
        """고유 인물 수 계산"""
        person_ids = set()
        for frame_poses in poses:
            for pose in frame_poses.persons:
                if pose.person_id is not None:
                    person_ids.add(pose.person_id)
        return len(person_ids)
    
    @staticmethod
    def create_window_annotation(scored_frames: List[FramePoses], window_number: int):
        """윈도우 어노테이션 생성 (MMAction2 표준 형식)"""
        from .data_structure import WindowAnnotation
        
        try:
            keypoint, keypoint_score = PipelineCommonUtils.convert_to_stgcn_format(scored_frames)
            
            # 이미지 크기 추정 (첫 번째 유효한 프레임에서)
            img_shape = (640, 640)  # 기본값
            for frame_poses in scored_frames:
                if frame_poses.image_shape is not None:
                    img_shape = frame_poses.image_shape
                    break
            
            window_annotation = WindowAnnotation(
                window_idx=window_number - 1,
                start_frame=scored_frames[0].frame_idx,
                end_frame=scored_frames[-1].frame_idx,
                keypoint=keypoint,
                keypoint_score=keypoint_score,
                frame_dir=f"window_{window_number}",
                img_shape=img_shape,
                original_shape=img_shape,
                total_frames=len(scored_frames),
                label=0
            )
            
            return window_annotation
            
        except Exception as e:
            logger.error(f"Error creating window annotation: {e}")
            # 간단한 윈도우 객체로 fallback
            class SimpleWindow:
                def __init__(self, frames, window_num):
                    self.frames = frames
                    self.start_frame = frames[0].frame_idx
                    self.end_frame = frames[-1].frame_idx
                    self.window_idx = window_num - 1
            return SimpleWindow(scored_frames, window_number)