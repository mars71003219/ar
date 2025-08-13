"""
분리형 파이프라인 데이터 구조
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from ...utils.data_structure import PersonPose, FramePoses, WindowAnnotation, ClassificationResult


@dataclass
class StageResult:
    """단계별 처리 결과"""
    stage_name: str
    input_path: str
    output_path: str
    processing_time: float
    metadata: Dict[str, Any]


@dataclass 
class VisualizationData:
    """시각화용 데이터 구조"""
    video_name: str
    frame_data: List[FramePoses]
    stage_info: Dict[str, Any]
    
    # Stage 1: 포즈 추정만
    poses_only: Optional[List[FramePoses]] = None
    
    # Stage 2: 포즈 + 트래킹
    poses_with_tracking: Optional[List[FramePoses]] = None
    tracking_info: Optional[Dict[str, Any]] = None
    
    # Stage 3: 포즈 + 트래킹 + 복합점수
    poses_with_scores: Optional[List[FramePoses]] = None
    scoring_info: Optional[Dict[str, Any]] = None
    classification_results: Optional[List[ClassificationResult]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (PKL 저장용)"""
        return {
            'video_name': self.video_name,
            'frame_data': [frame.to_dict() for frame in self.frame_data],
            'stage_info': self.stage_info,
            'poses_only': [frame.to_dict() for frame in self.poses_only] if self.poses_only else None,
            'poses_with_tracking': [frame.to_dict() for frame in self.poses_with_tracking] if self.poses_with_tracking else None,
            'tracking_info': self.tracking_info,
            'poses_with_scores': [frame.to_dict() for frame in self.poses_with_scores] if self.poses_with_scores else None,
            'scoring_info': self.scoring_info,
            'classification_results': [result.to_dict() for result in self.classification_results] if self.classification_results else None
        }


@dataclass
class STGCNData:
    """STGCN 훈련용 데이터 구조"""
    video_name: str
    keypoints_sequence: np.ndarray  # [T, V, C] - time, joints, channels
    label: int
    confidence: float
    metadata: Dict[str, Any]
    
    # 추가 정보
    person_id: Optional[int] = None
    window_info: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'video_name': self.video_name,
            'keypoints_sequence': self.keypoints_sequence.tolist() if isinstance(self.keypoints_sequence, np.ndarray) else self.keypoints_sequence,
            'label': self.label,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'person_id': self.person_id,
            'window_info': self.window_info,
            'quality_score': self.quality_score
        }
    
    @classmethod
    def from_window_annotation(cls, annotation: WindowAnnotation, video_name: str) -> 'STGCNData':
        """WindowAnnotation에서 STGCNData 생성"""
        return cls(
            video_name=video_name,
            keypoints_sequence=annotation.keypoints_sequence,
            label=annotation.label,
            confidence=annotation.confidence,
            metadata=annotation.metadata,
            person_id=annotation.person_id,
            window_info={
                'start_frame': annotation.start_frame,
                'end_frame': annotation.end_frame,
                'window_size': annotation.end_frame - annotation.start_frame + 1
            }
        )