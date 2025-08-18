"""
행동 분류 모듈 기본 클래스

모든 행동 분류 모델이 구현해야 하는 표준 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
try:
    from utils.data_structure import WindowAnnotation, ClassificationResult, ActionClassificationConfig
except ImportError:
    from ..utils.data_structure import WindowAnnotation, ClassificationResult, ActionClassificationConfig


class BaseActionClassifier(ABC):
    """행동 분류 모듈 기본 클래스"""
    
    def __init__(self, config: ActionClassificationConfig):
        """
        Args:
            config: 행동 분류 설정
        """
        self.config = config
        self.model = None
        self.model_path = config.model_path
        self.window_size = config.window_size
        self.input_format = config.input_format
        self.class_names = config.class_names or ['NonFight', 'Fight']
        self.confidence_threshold = config.confidence_threshold
        
        # 초기화 상태
        self.is_initialized = False
        
        # 입력 데이터 검증 관련
        self.expected_keypoint_count = config.expected_keypoint_count or 17
        self.coordinate_dimensions = config.coordinate_dimensions or 3  # x, y, confidence
        
        # 성능 통계
        self.stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'avg_confidence': 0.0,
            'class_distribution': {cls: 0 for cls in self.class_names}
        }
    
    @abstractmethod
    def initialize_model(self) -> bool:
        """모델 초기화
        
        Returns:
            초기화 성공 여부
        """
        pass
    
    def ensure_initialized(self) -> bool:
        """모델이 초기화되었는지 확인하고 필요시 초기화
        
        Returns:
            초기화 성공 여부
        """
        if not self.is_initialized:
            return self.initialize_model()
        return True
    
    @abstractmethod
    def classify_single_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류
        
        Args:
            window_data: 윈도우 포즈 데이터
            
        Returns:
            분류 결과
        """
        pass
    
    def classify_multiple_windows(self, windows: List[WindowAnnotation]) -> List[ClassificationResult]:
        """다중 윈도우 분류
        
        Args:
            windows: 윈도우 데이터 리스트
            
        Returns:
            분류 결과 리스트
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        results = []
        
        for window in windows:
            try:
                result = self.classify_single_window(window)
                results.append(result)
            except Exception as e:
                # 실패한 윈도우에 대해서는 기본값 반환
                default_result = ClassificationResult(
                    window_id=window.window_id,
                    predicted_class='NonFight',
                    confidence=0.0,
                    class_probabilities={'NonFight': 1.0, 'Fight': 0.0},
                    error=str(e)
                )
                results.append(default_result)
        
        # 통계 업데이트
        self.update_statistics(results)
        
        return results
    
    def preprocess_window_data(self, window_data: WindowAnnotation) -> Optional[np.ndarray]:
        """윈도우 데이터 전처리
        
        Args:
            window_data: 윈도우 데이터
            
        Returns:
            전처리된 numpy 배열 또는 None (실패 시)
        """
        try:
            # 윈도우 데이터 검증
            if not self.validate_window_data(window_data):
                return None
            
            # ST-GCN++ 형태로 변환
            if self.input_format == 'stgcn':
                return self._convert_to_stgcn_format(window_data)
            else:
                # 다른 형태 지원 시 추가
                return self._convert_to_default_format(window_data)
                
        except Exception as e:
            print(f"Error preprocessing window data: {str(e)}")
            return None
    
    def validate_window_data(self, window_data: WindowAnnotation) -> bool:
        """MMAction2 표준 어노테이션 유효성 검증
        
        Args:
            window_data: MMAction2 표준 형식의 윈도우 데이터
            
        Returns:
            유효성 여부
        """
        # MMAction2 표준 keypoint 데이터 확인
        if not hasattr(window_data, 'keypoint') or window_data.keypoint is None:
            logging.warning(f"validate_window_data failed: no keypoint data found")
            return False
        
        keypoint = window_data.keypoint
        keypoint_score = getattr(window_data, 'keypoint_score', None)
        
        # 형태 확인: [M, T, V, C]
        if len(keypoint.shape) != 4:
            logging.warning(f"validate_window_data failed: keypoint should be 4D [M, T, V, C], got {keypoint.shape}")
            return False
        
        M, T, V, C = keypoint.shape
        
        # 윈도우 크기 확인
        logging.info(f"validate_window_data: keypoint shape={keypoint.shape} [M, T, V, C], expected_window_size={self.window_size}")
        if T != self.window_size:
            logging.warning(f"validate_window_data failed: window size mismatch - got {T}, expected {self.window_size}")
            return False
        
        # 키포인트 수 확인
        if V != self.expected_keypoint_count:
            logging.warning(f"validate_window_data failed: keypoint count mismatch - got {V}, expected {self.expected_keypoint_count}")
            return False
        
        # 좌표 차원 확인 (2D: x, y)
        if C != 2:
            logging.warning(f"validate_window_data failed: coordinate dimension should be 2 (x, y), got {C}")
            return False
        
        # keypoint_score 형태 확인: [M, T, V]
        if keypoint_score is not None:
            if len(keypoint_score.shape) != 3:
                logging.warning(f"validate_window_data failed: keypoint_score should be 3D [M, T, V], got {keypoint_score.shape}")
                return False
            
            if keypoint_score.shape != (M, T, V):
                logging.warning(f"validate_window_data failed: keypoint_score shape mismatch - got {keypoint_score.shape}, expected {(M, T, V)}")
                return False
        
        # 데이터 품질 확인
        non_zero_ratio = np.count_nonzero(keypoint) / keypoint.size
        logging.info(f"validate_window_data: keypoint non-zero ratio={non_zero_ratio:.4f}")
        
        if non_zero_ratio < 0.01:  # 1% 미만이면 경고
            logging.warning(f"validate_window_data: very low non-zero ratio ({non_zero_ratio:.4f}), data might be empty")
        
        logging.info(f"validate_window_data: validation passed! Shape [M={M}, T={T}, V={V}, C={C}]")
        return True
    
    def _convert_to_stgcn_format(self, window_data: WindowAnnotation) -> Dict[str, Any]:
        """기존 rtmo_gcn_pipeline과 동일한 방식으로 MMAction2 data_sample 형태로 변환
        
        Args:
            window_data: 윈도우 데이터
            
        Returns:
            MMAction2 data_sample 딕셔너리
        """
        # WindowAnnotation에서 이미 준비된 keypoint 데이터 사용
        if hasattr(window_data, 'keypoint') and window_data.keypoint is not None:
            # keypoint 데이터 shape: (T, M, V, C)
            keypoint_data = window_data.keypoint
            keypoint_score_data = getattr(window_data, 'keypoint_score', None)
            
            logging.info(f"Using prepared keypoint data with shape: {keypoint_data.shape}")
            
            # WindowAnnotation이 이미 MMAction2 표준 형식: [M, T, V, C]이므로 변환 불필요
            keypoint = keypoint_data
            
            # WindowProcessor에서 이미 2D 좌표 (x, y)로 저장하므로 변환 불필요
            # keypoint shape은 [M, T, V, 2]이어야 함
            
            # keypoint_score도 이미 MMAction2 표준 형식: [M, T, V]이므로 변환 불필요
            if keypoint_score_data is not None:
                keypoint_score = keypoint_score_data
            else:
                # keypoint_score가 없으면 기본값으로 생성
                M, T, V = keypoint.shape[:3]
                keypoint_score = np.ones((M, T, V), dtype=np.float32)
            
            logging.info(f"Converted keypoint shape: {keypoint.shape}")
            logging.info(f"Converted keypoint_score shape: {keypoint_score.shape}")
            
            # MMAction2 data_sample 형태로 반환
            data_sample = {
                'keypoint': keypoint,  # (M, T, V, C) 형태
                'keypoint_score': keypoint_score,  # (M, T, V) 형태  
                'total_frames': self.window_size,
                'img_shape': (640, 640),  # 기본값
                'start_index': 0,
                'modality': 'Pose',
                'label': -1  # inference 시에는 더미 값
            }
            
            return data_sample
        
        # 대체 방법: frame_data에서 직접 변환 (백업용)
        T = self.window_size
        V = self.expected_keypoint_count
        C = self.coordinate_dimensions
        
        frames = getattr(window_data, 'frame_data', None)
        if not frames:
            logging.warning("No keypoint or frame_data found in window")
            return None
            
        # 최대 person 수 결정
        max_persons = max(len(frame.persons) if hasattr(frame, 'persons') and frame.persons else 0 for frame in frames)
        M = max(max_persons, 1)  # 최소 1명
        
        # 데이터 배열 초기화
        data = np.zeros((C, T, V, M), dtype=np.float32)
        
        # 윈도우 데이터를 배열에 채우기
        for t, frame_poses in enumerate(frames[:T]):  # T개까지만
            if hasattr(frame_poses, 'persons') and frame_poses.persons:
                for m, person in enumerate(frame_poses.persons[:M]):  # M개까지만
                    if hasattr(person, 'keypoints') and person.keypoints is not None:
                        keypoints = person.keypoints
                        if len(keypoints) >= V:  # 충분한 키포인트가 있는 경우
                            for v in range(V):
                                if v < len(keypoints):
                                    keypoint = keypoints[v]
                                    for c in range(min(C, len(keypoint) if hasattr(keypoint, '__len__') else 0)):
                                        data[c, t, v, m] = keypoint[c]
        
        return data
    
    def _convert_to_default_format(self, window_data: WindowAnnotation) -> np.ndarray:
        """기본 형태로 변환 (향후 다른 모델 지원시 사용)
        
        Args:
            window_data: 윈도우 데이터
            
        Returns:
            기본 형태의 numpy 배열
        """
        # 간단한 flatten된 형태로 변환
        flattened_data = []
        
        # 프레임 데이터 가져오기
        frames = getattr(window_data, 'frame_data', None) or getattr(window_data, 'poses', None)
        if not frames:
            return None
            
        for frame_poses in frames:
            frame_data = []
            for person in frame_poses.persons:
                for keypoint in person.keypoints:
                    frame_data.extend(keypoint[:self.coordinate_dimensions])
            
            # 패딩 또는 자르기
            target_size = self.expected_keypoint_count * self.coordinate_dimensions
            if len(frame_data) < target_size:
                frame_data.extend([0.0] * (target_size - len(frame_data)))
            else:
                frame_data = frame_data[:target_size]
            
            flattened_data.extend(frame_data)
        
        return np.array(flattened_data, dtype=np.float32)
    
    def update_statistics(self, results: List[ClassificationResult]):
        """통계 정보 업데이트
        
        Args:
            results: 분류 결과 리스트
        """
        if not results:
            return
        
        valid_results = [r for r in results if r.error is None]
        
        self.stats['total_classifications'] += len(results)
        self.stats['successful_classifications'] += len(valid_results)
        
        if valid_results:
            # 평균 신뢰도 계산
            confidences = [r.confidence for r in valid_results]
            self.stats['avg_confidence'] = np.mean(confidences)
            
            # 클래스 분포 업데이트
            for result in valid_results:
                predicted_class = result.predicted_class
                if predicted_class in self.stats['class_distribution']:
                    self.stats['class_distribution'][predicted_class] += 1
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """분류기 정보 반환
        
        Returns:
            분류기 정보 딕셔너리
        """
        return {
            'model_name': self.config.model_name,
            'model_path': self.model_path,
            'window_size': self.window_size,
            'input_format': self.input_format,
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'is_initialized': self.is_initialized,
            'statistics': self.stats.copy()
        }
    
    def set_confidence_threshold(self, threshold: float):
        """신뢰도 임계값 설정
        
        Args:
            threshold: 새로운 임계값 (0.0 ~ 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.config.confidence_threshold = threshold
    
    def get_supported_input_formats(self) -> List[str]:
        """지원하는 입력 형태 반환
        
        Returns:
            지원하는 입력 형태 리스트
        """
        return ['stgcn', 'default']  # 기본적으로 지원하는 형태들
    
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
        if self.model is not None:
            self.model = None
        self.is_initialized = False