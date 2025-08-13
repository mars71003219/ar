"""
트래킹 모듈 기본 클래스

모든 트래킹 모델이 구현해야 하는 표준 인터페이스를 정의합니다.
pose_estimation -> tracking -> scoring 파이프라인에서 정확한 입/출력을 보장합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..utils.data_structure import PersonPose, FramePoses, TrackingConfig


class TrackedObject:
    """트래킹된 객체 정보"""
    
    def __init__(self, track_id: int, bbox: List[float], keypoints: np.ndarray, 
                 score: float, age: int = 1, hits: int = 1):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.keypoints = keypoints  # [17, 3] for COCO format
        self.score = score
        self.age = age
        self.hits = hits
        self.time_since_update = 0
        
        # 추가 메타데이터
        self.velocity = np.zeros(4)  # [dx, dy, ds, dr] for bbox
        self.confidence_history = [score]
        
    def predict(self):
        """객체 상태 예측 (칼만 필터 등)"""
        pass
        
    def update(self, bbox: List[float], keypoints: np.ndarray, score: float):
        """트래킹 정보 업데이트"""
        self.bbox = bbox
        self.keypoints = keypoints
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.confidence_history.append(score)
        
        # 히스토리 길이 제한
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)
    
    def mark_missed(self):
        """추적 실패 표시"""
        self.time_since_update += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'keypoints': self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints,
            'score': self.score,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.0
        }
    
    @classmethod
    def from_person_pose(cls, person: PersonPose, track_id: int) -> 'TrackedObject':
        """PersonPose에서 생성"""
        return cls(
            track_id=track_id,
            bbox=person.bbox,
            keypoints=person.keypoints,
            score=person.score
        )


class BaseTracker(ABC):
    """트래킹 모듈 기본 클래스
    
    pose_estimation의 PersonPose 리스트를 입력받아
    track_id가 할당된 PersonPose 리스트를 출력합니다.
    """
    
    def __init__(self, config: TrackingConfig):
        """
        Args:
            config: 트래킹 설정
        """
        self.config = config
        self.tracker_name = config.tracker_name
        self.device = config.device
        
        # 트래킹 파라미터
        self.high_threshold = config.high_threshold
        self.low_threshold = config.low_threshold
        self.max_disappeared = config.max_disappeared
        self.min_hits = config.min_hits
        
        # 트래킹 상태
        self.frame_id = 0
        self.is_initialized = False
        
        # 활성 트랙 관리
        self.active_tracks: List[TrackedObject] = []
        self.lost_tracks: List[TrackedObject] = []
        self.removed_tracks: List[TrackedObject] = []
        
        # 통계
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'removed_tracks': 0,
            'frames_processed': 0,
            'avg_tracking_time': 0.0
        }
        
    @abstractmethod
    def initialize_tracker(self) -> bool:
        """트래커 초기화
        
        Returns:
            초기화 성공 여부
        """
        pass
        
    @abstractmethod
    def update(self, detections: List[PersonPose]) -> List[TrackedObject]:
        """트래킹 업데이트
        
        Args:
            detections: 현재 프레임 검출 결과 (PersonPose 리스트)
            
        Returns:
            트래킹된 객체 정보 리스트 (확인된 트랙만)
        """
        pass
        
    @abstractmethod
    def reset(self):
        """트래커 리셋"""
        pass
    
    def track_frame_poses(self, frame_poses: FramePoses) -> FramePoses:
        """FramePoses에 트래킹 적용
        
        Args:
            frame_poses: 입력 프레임 포즈 데이터 (pose_estimation 출력)
            
        Returns:
            트래킹이 적용된 FramePoses (scoring 입력용)
        """
        # 입력 검증
        if not frame_poses.persons:
            return FramePoses(
                frame_idx=frame_poses.frame_idx,
                persons=[],
                timestamp=frame_poses.timestamp,
                image_shape=frame_poses.image_shape
            )
        
        # 검출 결과 전처리
        valid_detections = self.validate_detections(frame_poses.persons)
        
        # 트래킹 업데이트
        tracked_objects = self.update(valid_detections)
        
        # TrackedObject를 PersonPose로 변환
        tracked_persons = []
        for tracked_obj in tracked_objects:
            person = PersonPose(
                person_id=tracked_obj.track_id,
                bbox=tracked_obj.bbox,
                keypoints=tracked_obj.keypoints,
                score=tracked_obj.score,
                track_id=tracked_obj.track_id,
                timestamp=frame_poses.timestamp
            )
            tracked_persons.append(person)
        
        # 트래킹된 FramePoses 생성
        tracked_frame = FramePoses(
            frame_idx=frame_poses.frame_idx,
            persons=tracked_persons,
            timestamp=frame_poses.timestamp,
            image_shape=frame_poses.image_shape
        )
        
        return tracked_frame
    
    def track_video_poses(self, frame_poses_list: List[FramePoses]) -> List[FramePoses]:
        """전체 비디오 포즈에 트래킹 적용
        
        Args:
            frame_poses_list: 비디오의 모든 프레임 포즈 데이터 (pose_estimation 출력)
            
        Returns:
            트래킹이 적용된 프레임 포즈 리스트 (scoring 입력용)
        """
        if not self.is_initialized:
            if not self.initialize_tracker():
                return frame_poses_list
        
        # 트래커 리셋
        self.reset()
        
        tracked_frames = []
        
        for frame_poses in frame_poses_list:
            self.frame_id = frame_poses.frame_idx
            tracked_frame = self.track_frame_poses(frame_poses)
            tracked_frames.append(tracked_frame)
            
            # 통계 업데이트
            self.stats['frames_processed'] += 1
            self.stats['active_tracks'] = len(self.active_tracks)
            self.stats['lost_tracks'] = len(self.lost_tracks)
            self.stats['removed_tracks'] = len(self.removed_tracks)
        
        return tracked_frames
    
    def validate_detections(self, detections: List[PersonPose]) -> List[PersonPose]:
        """검출 결과 유효성 검사"""
        valid_detections = []
        
        for detection in detections:
            # 바운딩 박스 유효성 확인
            if len(detection.bbox) == 4:
                x1, y1, x2, y2 = detection.bbox
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                    # 키포인트 유효성 확인
                    if isinstance(detection.keypoints, np.ndarray):
                        if detection.keypoints.shape == (17, 3):
                            valid_detections.append(detection)
                    elif isinstance(detection.keypoints, list):
                        if len(detection.keypoints) == 51:  # 17 * 3
                            # 리스트를 numpy 배열로 변환
                            kpts = np.array(detection.keypoints).reshape(17, 3)
                            detection.keypoints = kpts
                            valid_detections.append(detection)
        
        return valid_detections
    
    def get_tracker_info(self) -> Dict[str, Any]:
        """트래커 정보 반환"""
        return {
            'tracker_name': self.tracker_name,
            'high_threshold': self.high_threshold,
            'low_threshold': self.low_threshold,
            'max_disappeared': self.max_disappeared,
            'min_hits': self.min_hits,
            'current_frame': self.frame_id,
            'is_initialized': self.is_initialized,
            'statistics': self.stats.copy(),
            'active_track_count': len(self.active_tracks),
            'lost_track_count': len(self.lost_tracks)
        }
    
    def get_active_track_ids(self) -> List[int]:
        """현재 활성 트랙 ID 리스트 반환"""
        return [track.track_id for track in self.active_tracks]
    
    def get_track_history(self, track_id: int) -> List[Dict[str, Any]]:
        """특정 트랙의 히스토리 반환"""
        # 하위 클래스에서 구현
        return []
    
    def set_thresholds(self, high_threshold: Optional[float] = None,
                      low_threshold: Optional[float] = None):
        """임계값 설정
        
        Args:
            high_threshold: 높은 신뢰도 임계값
            low_threshold: 낮은 신뢰도 임계값
        """
        if high_threshold is not None:
            self.high_threshold = high_threshold
            self.config.high_threshold = high_threshold
            
        if low_threshold is not None:
            self.low_threshold = low_threshold
            self.config.low_threshold = low_threshold
    
    def cleanup(self):
        """리소스 정리"""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.reset()
        self.is_initialized = False
    
    def __enter__(self):
        """Context manager 진입"""
        if not self.is_initialized:
            self.initialize_tracker()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.cleanup()