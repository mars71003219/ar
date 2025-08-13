"""
포즈 추정 모듈 기본 클래스

모든 포즈 추정 모델이 구현해야 하는 표준 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig


class BasePoseEstimator(ABC):
    """포즈 추정 모듈 기본 클래스"""
    
    def __init__(self, config: PoseEstimationConfig):
        """
        Args:
            config: 포즈 추정 설정
        """
        self.config = config
        self.device = config.device
        self.score_threshold = config.score_threshold
        self.nms_threshold = config.nms_threshold
        self.max_detections = config.max_detections
        
        # 모델 관련 속성
        self.model = None
        self.is_initialized = False
        
    @abstractmethod
    def initialize_model(self) -> bool:
        """모델 초기화
        
        Returns:
            초기화 성공 여부
        """
        pass
        
    @abstractmethod
    def extract_poses(self, frame: np.ndarray, frame_idx: int = 0) -> List[PersonPose]:
        """단일 프레임에서 포즈 추출
        
        Args:
            frame: 입력 이미지 (H, W, C)
            frame_idx: 프레임 인덱스
            
        Returns:
            검출된 person별 포즈 정보 리스트
        """
        pass
        
    @abstractmethod
    def extract_video_poses(self, video_path: str) -> List[FramePoses]:
        """전체 비디오에서 포즈 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            프레임별 포즈 정보 리스트
        """
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        pass
    
    def __enter__(self):
        """Context manager 진입"""
        if not self.is_initialized:
            self.initialize_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'model') and self.model is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """프레임 유효성 검사
        
        Args:
            frame: 입력 프레임
            
        Returns:
            유효성 여부
        """
        if frame is None:
            return False
        
        if len(frame.shape) != 3:
            return False
            
        if frame.shape[2] != 3:  # RGB/BGR 채널 확인
            return False
            
        return True
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 (기본 구현)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            전처리된 프레임
        """
        # 기본적으로는 원본 반환
        return frame
    
    def postprocess_detections(self, raw_detections: Any, frame_shape: Tuple[int, int]) -> List[PersonPose]:
        """검출 결과 후처리 (기본 구현)
        
        Args:
            raw_detections: 원시 검출 결과
            frame_shape: 프레임 크기 (H, W)
            
        Returns:
            후처리된 PersonPose 리스트
        """
        # 하위 클래스에서 구현
        return []
    
    def filter_low_quality_poses(self, poses: List[PersonPose]) -> List[PersonPose]:
        """낮은 품질의 포즈 필터링
        
        Args:
            poses: 원본 포즈 리스트
            
        Returns:
            필터링된 포즈 리스트
        """
        filtered_poses = []
        
        for pose in poses:
            # 전체 신뢰도 점수가 임계값 이상인 경우만 포함
            if pose.score >= self.score_threshold:
                # 키포인트 품질 확인
                if isinstance(pose.keypoints, np.ndarray) and pose.keypoints.shape[0] > 0:
                    # 유효한 키포인트 비율 확인
                    valid_keypoints = np.sum(pose.keypoints[:, 2] > 0.3) / len(pose.keypoints)
                    if valid_keypoints >= 0.3:  # 30% 이상의 키포인트가 유효해야 함
                        filtered_poses.append(pose)
                else:
                    filtered_poses.append(pose)
        
        return filtered_poses
    
    def get_statistics(self) -> Dict[str, Any]:
        """처리 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        return {
            'model_name': self.config.model_name,
            'device': self.device,
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'is_initialized': self.is_initialized
        }
    
    def set_thresholds(self, score_threshold: Optional[float] = None, 
                      nms_threshold: Optional[float] = None):
        """임계값 설정
        
        Args:
            score_threshold: 점수 임계값
            nms_threshold: NMS 임계값
        """
        if score_threshold is not None:
            self.score_threshold = score_threshold
            self.config.score_threshold = score_threshold
            
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
            self.config.nms_threshold = nms_threshold
    
    def benchmark(self, video_path: str, num_frames: int = 100) -> Dict[str, float]:
        """성능 벤치마크
        
        Args:
            video_path: 테스트 비디오 경로
            num_frames: 테스트할 프레임 수
            
        Returns:
            성능 메트릭
        """
        import time
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video'}
        
        times = []
        total_persons = 0
        processed_frames = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            poses = self.extract_poses(frame, i)
            end_time = time.time()
            
            times.append(end_time - start_time)
            total_persons += len(poses)
            processed_frames += 1
        
        cap.release()
        
        if processed_frames == 0:
            return {'error': 'No frames processed'}
        
        return {
            'avg_fps': processed_frames / sum(times),
            'avg_time_per_frame': np.mean(times),
            'std_time_per_frame': np.std(times),
            'avg_persons_per_frame': total_persons / processed_frames,
            'total_frames_processed': processed_frames,
            'total_persons_detected': total_persons
        }