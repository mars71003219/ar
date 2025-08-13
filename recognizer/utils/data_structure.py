"""
표준 데이터 구조 정의

행동 탐지 시스템에서 사용되는 모든 데이터 구조를 정의합니다.
모든 모듈 간 일관된 데이터 형식을 보장합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from enum import Enum
from .gpu_manager import get_gpu_manager, setup_multi_gpu
import multiprocessing as mp


class TaskType(Enum):
    """작업 유형"""
    POSE_ESTIMATION = "pose_estimation"
    TRACKING = "tracking"
    SCORING = "scoring"
    CLASSIFICATION = "classification"


class LabelType(Enum):
    """레이블 유형"""
    FIGHT = 1
    NON_FIGHT = 0


@dataclass
class PersonPose:
    """개별 person 포즈 데이터"""
    person_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # [17, 3] (x, y, confidence)
    score: float
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'person_id': self.person_id,
            'bbox': self.bbox,
            'keypoints': self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints,
            'score': self.score,
            'track_id': self.track_id,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonPose':
        """딕셔너리에서 생성"""
        keypoints = np.array(data['keypoints']) if isinstance(data['keypoints'], list) else data['keypoints']
        return cls(
            person_id=data['person_id'],
            bbox=data['bbox'],
            keypoints=keypoints,
            score=data['score'],
            track_id=data.get('track_id'),
            timestamp=data.get('timestamp')
        )


@dataclass
class FramePoses:
    """프레임별 포즈 데이터"""
    frame_idx: int
    persons: List[PersonPose]
    timestamp: float
    image_shape: Optional[Tuple[int, int]] = None  # (H, W)
    
    def get_person_count(self) -> int:
        """person 수 반환"""
        return len(self.persons)
    
    def get_person_by_track_id(self, track_id: int) -> Optional[PersonPose]:
        """track_id로 person 검색"""
        for person in self.persons:
            if person.track_id == track_id:
                return person
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'frame_idx': self.frame_idx,
            'persons': [person.to_dict() for person in self.persons],
            'timestamp': self.timestamp,
            'image_shape': self.image_shape
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FramePoses':
        """딕셔너리에서 생성"""
        return cls(
            frame_idx=data['frame_idx'],
            persons=[PersonPose.from_dict(p) for p in data['persons']],
            timestamp=data['timestamp'],
            image_shape=data.get('image_shape')
        )


@dataclass
class WindowAnnotation:
    """윈도우 어노테이션 데이터 (ST-GCN++ 형식)"""
    window_idx: int
    start_frame: int
    end_frame: int
    keypoint: np.ndarray  # [T, M, V, C] - Time, Max_persons, Vertices, Coordinates
    keypoint_score: np.ndarray  # [T, M, V] - confidence scores
    frame_dir: str
    img_shape: Tuple[int, int]  # (H, W)
    original_shape: Tuple[int, int]  # (H, W)
    total_frames: int
    label: int  # 0: NonFight, 1: Fight
    
    # 추가 메타데이터
    video_name: Optional[str] = None
    composite_scores: Optional[Dict[int, float]] = None  # track_id별 복합점수
    person_rankings: Optional[List[Tuple[int, float]]] = None  # (track_id, score) 순위
    
    def get_shape_info(self) -> Dict[str, Any]:
        """형태 정보 반환"""
        T, M, V, C = self.keypoint.shape
        return {
            'temporal_frames': T,
            'max_persons': M,
            'num_joints': V,
            'coordinates': C,
            'total_frames': self.total_frames
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (numpy 배열 제외)"""
        return {
            'window_idx': self.window_idx,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'frame_dir': self.frame_dir,
            'img_shape': self.img_shape,
            'original_shape': self.original_shape,
            'total_frames': self.total_frames,
            'label': self.label,
            'video_name': self.video_name,
            'composite_scores': self.composite_scores,
            'person_rankings': self.person_rankings,
            'shape_info': self.get_shape_info()
        }


@dataclass
class ClassificationResult:
    """분류 결과 데이터"""
    prediction: int  # 0: NonFight, 1: Fight
    confidence: float
    probabilities: List[float]  # [NonFight_prob, Fight_prob]
    features: Optional[np.ndarray] = None
    
    # 추가 정보
    model_name: Optional[str] = None
    processing_time: Optional[float] = None
    input_shape: Optional[Tuple] = None
    
    def get_predicted_label(self) -> str:
        """예측 레이블을 문자열로 반환"""
        return "Fight" if self.prediction == 1 else "NonFight"
    
    def get_confidence_percentage(self) -> float:
        """신뢰도를 퍼센트로 반환"""
        return self.confidence * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'prediction': self.prediction,
            'predicted_label': self.get_predicted_label(),
            'confidence': self.confidence,
            'confidence_percentage': self.get_confidence_percentage(),
            'probabilities': self.probabilities,
            'model_name': self.model_name,
            'processing_time': self.processing_time,
            'input_shape': self.input_shape
        }


# 설정 구조체들
@dataclass
class PoseEstimationConfig:
    """포즈 추정 설정"""
    model_name: str
    config_file: str = ""
    model_path: str = ""
    device: Union[str, int, List] = 'cuda:0'
    batch_size: int = 1
    input_size: List[int] = field(default_factory=lambda: [640, 640])
    score_threshold: float = 0.3
    nms_threshold: float = 0.65
    max_detections: int = 100
    
    # 멀티 GPU 관련 설정
    gpu_allocation_strategy: str = 'round_robin'  # round_robin, memory_based, first_available
    use_data_parallel: bool = False  # DataParallel 사용 여부
    use_distributed: bool = False    # DistributedDataParallel 사용 여부
    
    def __post_init__(self):
        """초기화 후 GPU 설정"""
        if isinstance(self.device, (str, int, list)):
            self._allocated_devices = setup_multi_gpu(
                self.device, 
                'pose_estimator', 
                self.gpu_allocation_strategy
            )
        else:
            self._allocated_devices = ['cpu']
    
    def get_primary_device(self) -> str:
        """주 디바이스 반환"""
        return self._allocated_devices[0] if self._allocated_devices else 'cpu'
    
    def get_all_devices(self) -> List[str]:
        """모든 할당된 디바이스 반환"""
        return self._allocated_devices.copy()
    
    def is_multi_gpu(self) -> bool:
        """멀티 GPU 사용 여부"""
        return len(self._allocated_devices) > 1 and 'cuda' in self._allocated_devices[0]
    
    def get_device_count(self) -> int:
        """할당된 GPU 수"""
        return len([d for d in self._allocated_devices if 'cuda' in d])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'config_file': self.config_file,
            'model_path': self.model_path,
            'device': self.device,
            'batch_size': self.batch_size,
            'input_size': self.input_size,
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'gpu_allocation_strategy': self.gpu_allocation_strategy,
            'use_data_parallel': self.use_data_parallel,
            'use_distributed': self.use_distributed,
            'allocated_devices': self._allocated_devices
        }


@dataclass
class TrackingConfig:
    """트래킹 설정"""
    tracker_name: str
    high_threshold: float = 0.6
    low_threshold: float = 0.1
    init_track_threshold: float = 0.7
    match_iou_high: float = 0.1
    match_iou_low: float = 0.5
    match_iou_tentative: float = 0.3
    num_tentatives: int = 3
    num_frames_retain: int = 30
    max_disappeared: int = 30
    min_hits: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tracker_name': self.tracker_name,
            'high_threshold': self.high_threshold,
            'low_threshold': self.low_threshold,
            'init_track_threshold': self.init_track_threshold,
            'match_iou_high': self.match_iou_high,
            'match_iou_low': self.match_iou_low,
            'match_iou_tentative': self.match_iou_tentative,
            'num_tentatives': self.num_tentatives,
            'num_frames_retain': self.num_frames_retain,
            'max_disappeared': self.max_disappeared,
            'min_hits': self.min_hits
        }


@dataclass  
class ScoringConfig:
    """복합점수 계산 설정"""
    scorer_name: str
    movement_weight: float = 0.3
    position_weight: float = 0.2
    interaction_weight: float = 0.1
    temporal_consistency_weight: float = 0.2
    persistence_weight: float = 0.2
    quality_threshold: float = 0.3
    min_track_length: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scorer_name': self.scorer_name,
            'movement_weight': self.movement_weight,
            'position_weight': self.position_weight,
            'interaction_weight': self.interaction_weight,
            'temporal_consistency_weight': self.temporal_consistency_weight,
            'persistence_weight': self.persistence_weight,
            'quality_threshold': self.quality_threshold,
            'min_track_length': self.min_track_length
        }
    
    def get_weights_as_list(self) -> List[float]:
        """가중치를 리스트로 반환"""
        return [
            self.movement_weight,
            self.position_weight,
            self.interaction_weight,
            self.temporal_consistency_weight,
            self.persistence_weight
        ]


@dataclass
class ActionClassificationConfig:
    """행동 분류 설정"""
    model_name: str
    config_file: str = ""
    model_path: str = ""
    device: Union[str, int, List] = 'cuda:0'
    window_size: int = 100
    class_names: List[str] = field(default_factory=lambda: ['NonFight', 'Fight'])
    confidence_threshold: float = 0.5
    input_format: str = 'stgcn'
    expected_keypoint_count: int = 17
    coordinate_dimensions: int = 2
    max_persons: int = 2
    
    # 멀티 GPU 관련 설정
    gpu_allocation_strategy: str = 'round_robin'
    use_data_parallel: bool = False
    use_distributed: bool = False
    batch_inference: bool = False
    max_batch_size: int = 8
    
    def __post_init__(self):
        """초기화 후 GPU 설정"""
        if isinstance(self.device, (str, int, list)):
            self._allocated_devices = setup_multi_gpu(
                self.device, 
                'action_classifier', 
                self.gpu_allocation_strategy
            )
        else:
            self._allocated_devices = ['cpu']
    
    def get_primary_device(self) -> str:
        """주 디바이스 반환"""
        return self._allocated_devices[0] if self._allocated_devices else 'cpu'
    
    def get_all_devices(self) -> List[str]:
        """모든 할당된 디바이스 반환"""
        return self._allocated_devices.copy()
    
    def is_multi_gpu(self) -> bool:
        """멀티 GPU 사용 여부"""
        return len(self._allocated_devices) > 1 and 'cuda' in self._allocated_devices[0]
    
    def get_device_count(self) -> int:
        """할당된 GPU 수"""
        return len([d for d in self._allocated_devices if 'cuda' in d])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'config_file': self.config_file,
            'model_path': self.model_path,
            'device': self.device,
            'window_size': self.window_size,
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'input_format': self.input_format,
            'expected_keypoint_count': self.expected_keypoint_count,
            'coordinate_dimensions': self.coordinate_dimensions,
            'max_persons': self.max_persons,
            'gpu_allocation_strategy': self.gpu_allocation_strategy,
            'use_data_parallel': self.use_data_parallel,
            'use_distributed': self.use_distributed,
            'batch_inference': self.batch_inference,
            'max_batch_size': self.max_batch_size,
            'allocated_devices': self._allocated_devices
        }


@dataclass
class PipelineConfig:
    """통합 파이프라인 설정"""
    pose_config: PoseEstimationConfig
    tracking_config: TrackingConfig
    scoring_config: ScoringConfig
    classification_config: ActionClassificationConfig
    
    # 윈도우 설정
    window_size: int = 100
    window_stride: int = 50
    batch_size: int = 1
    
    # 성능 설정
    enable_gpu: bool = True
    device: Union[str, int, List] = 'cuda:0'
    
    # 멀티 GPU 관련 설정
    gpu_allocation_strategy: str = 'round_robin'  # round_robin, memory_based, first_available
    load_balance_components: bool = True  # 컴포넌트별 로드 밸런싱
    
    # 출력 설정
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None
    
    # 멀티프로세스 설정
    num_workers: int = field(default_factory=lambda: mp.cpu_count())
    enable_multiprocess: bool = False
    multiprocess_batch_size: int = 1
    multiprocess_timeout: float = 300.0  # 5분
    
    def __post_init__(self):
        """초기화 후 GPU 설정 및 검증"""
        # 파이프라인 전체 GPU 설정
        if isinstance(self.device, (str, int, list)):
            self._allocated_devices = setup_multi_gpu(
                self.device, 
                'pipeline', 
                self.gpu_allocation_strategy
            )
        else:
            self._allocated_devices = ['cpu']
        
        # 로드 밸런싱이 활성화된 경우 컴포넌트별 디바이스 재분배
        if self.load_balance_components and self.is_multi_gpu():
            self._redistribute_devices()
    
    def _redistribute_devices(self):
        """컴포넌트별 GPU 재분배"""
        if not self.is_multi_gpu():
            return
        
        gpu_manager = get_gpu_manager()
        available_devices = [d.split(':')[1] for d in self._allocated_devices if 'cuda' in d]
        
        if len(available_devices) >= 2:
            # 포즈 추정과 행동 분류는 계산량이 많으므로 우선적으로 할당
            pose_devices = available_devices[:len(available_devices)//2]
            classification_devices = available_devices[len(available_devices)//2:]
            
            # 설정 업데이트 (강제로 재할당)
            gpu_manager.clear_allocation('pose_estimator')
            gpu_manager.clear_allocation('action_classifier')
            
            setup_multi_gpu(pose_devices, 'pose_estimator', self.gpu_allocation_strategy)
            setup_multi_gpu(classification_devices, 'action_classifier', self.gpu_allocation_strategy)
    
    def get_primary_device(self) -> str:
        """주 디바이스 반환"""
        return self._allocated_devices[0] if self._allocated_devices else 'cpu'
    
    def get_all_devices(self) -> List[str]:
        """모든 할당된 디바이스 반환"""
        return self._allocated_devices.copy()
    
    def is_multi_gpu(self) -> bool:
        """멀티 GPU 사용 여부"""
        return len(self._allocated_devices) > 1 and 'cuda' in self._allocated_devices[0]
    
    def get_device_count(self) -> int:
        """할당된 GPU 수"""
        return len([d for d in self._allocated_devices if 'cuda' in d])
    
    def get_gpu_allocation_summary(self) -> Dict[str, Any]:
        """GPU 할당 요약 반환"""
        gpu_manager = get_gpu_manager()
        return {
            'pipeline_devices': self._allocated_devices,
            'total_gpu_count': self.get_device_count(),
            'is_multi_gpu': self.is_multi_gpu(),
            'load_balance_enabled': self.load_balance_components,
            'component_allocation': gpu_manager.device_allocation.copy(),
            'allocation_strategy': self.gpu_allocation_strategy
        }
    
    def validate(self) -> bool:
        """설정 유효성 검사"""
        if self.window_size <= 0 or self.window_stride <= 0:
            return False
        if self.window_stride > self.window_size:
            import logging
            logging.warning("Window stride is larger than window size")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pose_config': self.pose_config.to_dict(),
            'tracking_config': self.tracking_config.to_dict(),
            'scoring_config': self.scoring_config.to_dict(),
            'classification_config': self.classification_config.to_dict(),
            'window_size': self.window_size,
            'window_stride': self.window_stride,
            'batch_size': self.batch_size,
            'enable_gpu': self.enable_gpu,
            'device': self.device,
            'gpu_allocation_strategy': self.gpu_allocation_strategy,
            'load_balance_components': self.load_balance_components,
            'save_intermediate_results': self.save_intermediate_results,
            'output_dir': self.output_dir,
            'num_workers': self.num_workers,
            'enable_multiprocess': self.enable_multiprocess,
            'multiprocess_batch_size': self.multiprocess_batch_size,
            'multiprocess_timeout': self.multiprocess_timeout,
            'allocated_devices': self._allocated_devices
        }


# 유틸리티 함수들
def convert_poses_to_stgcn_format(frame_poses_list: List[FramePoses], 
                                 max_persons: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """FramePoses 리스트를 ST-GCN++ 형식으로 변환
    
    Args:
        frame_poses_list: 프레임별 포즈 데이터 리스트
        max_persons: 최대 person 수
        
    Returns:
        keypoint: [T, M, V, C] 형태의 키포인트 배열
        keypoint_score: [T, M, V] 형태의 신뢰도 배열
    """
    T = len(frame_poses_list)
    M = max_persons
    V = 17  # COCO 키포인트 수
    C = 3   # x, y, confidence
    
    keypoint = np.zeros((T, M, V, C), dtype=np.float32)
    keypoint_score = np.zeros((T, M, V), dtype=np.float32)
    
    for t, frame_poses in enumerate(frame_poses_list):
        for m, person in enumerate(frame_poses.persons[:max_persons]):
            if isinstance(person.keypoints, np.ndarray) and person.keypoints.shape == (V, C):
                keypoint[t, m, :, :] = person.keypoints
                keypoint_score[t, m, :] = person.keypoints[:, 2]  # confidence
            elif len(person.keypoints) == V * C:
                # 1D 배열인 경우 reshape
                kpts = np.array(person.keypoints).reshape(V, C)
                keypoint[t, m, :, :] = kpts
                keypoint_score[t, m, :] = kpts[:, 2]
    
    return keypoint, keypoint_score


def create_window_annotation(window_poses: List[FramePoses],
                           window_idx: int,
                           start_frame: int,
                           end_frame: int,
                           label: int,
                           video_name: Optional[str] = None,
                           composite_scores: Optional[Dict[int, float]] = None) -> WindowAnnotation:
    """윈도우 어노테이션 생성 도우미 함수"""
    
    # ST-GCN 형식으로 변환
    keypoint, keypoint_score = convert_poses_to_stgcn_format(window_poses)
    
    # 이미지 크기 추정 (첫 번째 유효한 프레임에서)
    img_shape = (640, 640)  # 기본값
    for frame_poses in window_poses:
        if frame_poses.image_shape is not None:
            img_shape = frame_poses.image_shape
            break
    
    return WindowAnnotation(
        window_idx=window_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        frame_dir=f"{video_name}_window_{window_idx}" if video_name else f"window_{window_idx}",
        img_shape=img_shape,
        original_shape=img_shape,
        total_frames=len(window_poses),
        label=label,
        video_name=video_name,
        composite_scores=composite_scores
    )


def validate_data_structure(data: Any, expected_type: type) -> bool:
    """데이터 구조 유효성 검사"""
    try:
        if not isinstance(data, expected_type):
            return False
            
        if isinstance(data, PersonPose):
            return (len(data.bbox) == 4 and 
                   isinstance(data.keypoints, np.ndarray) and 
                   data.keypoints.shape == (17, 3))
                   
        elif isinstance(data, FramePoses):
            return all(validate_data_structure(person, PersonPose) for person in data.persons)
            
        elif isinstance(data, WindowAnnotation):
            T, M, V, C = data.keypoint.shape
            return (T > 0 and M > 0 and V == 17 and C == 3 and 
                   data.keypoint_score.shape == (T, M, V))
                   
        return True
        
    except Exception:
        return False


# 이전한 유틸리티 함수들 가져오기
def convert_poses_to_stgcn_format(annotations):
    """
    이전한 convert_poses_to_stgcn_format 함수 사용
    action_classification.stgcn.data_utils에서 가져옴
    """
    try:
        from ..action_classification.stgcn.data_utils import convert_poses_to_stgcn_format as _convert
        return _convert(annotations)
    except ImportError:
        # Fallback 구현
        return []


def create_window_annotation(persons_data, window_metadata):
    """
    이전한 create_window_annotation 함수 사용
    action_classification.stgcn.data_utils에서 가져옴
    """
    try:
        from ..action_classification.stgcn.data_utils import create_window_annotation as _create
        return _create(persons_data, window_metadata)
    except ImportError:
        # Fallback 구현
        return {
            'persons': persons_data,
            'metadata': window_metadata
        }