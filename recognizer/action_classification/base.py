"""
행동 분류 모듈 기본 클래스

모든 행동 분류 모델이 구현해야 하는 표준 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
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
        self.coordinate_dimensions = config.coordinate_dimensions or 2
        
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
        """윈도우 데이터 유효성 검증
        
        Args:
            window_data: 윈도우 데이터
            
        Returns:
            유효성 여부
        """
        if not window_data.poses:
            return False
        
        # 윈도우 크기 확인
        if len(window_data.poses) != self.window_size:
            return False
        
        # 각 프레임의 포즈 데이터 확인
        for frame_poses in window_data.poses:
            if not frame_poses.persons:
                continue  # 빈 프레임은 허용
                
            for person in frame_poses.persons:
                # 키포인트 수 확인
                if len(person.keypoints) != self.expected_keypoint_count:
                    return False
                
                # 좌표 차원 확인
                for kpt in person.keypoints:
                    if len(kpt) != self.coordinate_dimensions:
                        return False
        
        return True
    
    def _convert_to_stgcn_format(self, window_data: WindowAnnotation) -> np.ndarray:
        """ST-GCN++ 입력 형태로 변환
        
        Args:
            window_data: 윈도우 데이터
            
        Returns:
            ST-GCN++ 입력 형태의 numpy 배열 (C, T, V, M)
            C: 좌표 차원 (2 또는 3)
            T: 시간 프레임 수
            V: 키포인트 수 (17)
            M: 최대 person 수
        """
        T = self.window_size
        V = self.expected_keypoint_count
        C = self.coordinate_dimensions
        
        # 최대 person 수 결정
        max_persons = max(len(frame.persons) for frame in window_data.poses)
        M = max(max_persons, 1)  # 최소 1명
        
        # 데이터 배열 초기화
        data = np.zeros((C, T, V, M), dtype=np.float32)
        
        # 윈도우 데이터를 배열에 채우기
        for t, frame_poses in enumerate(window_data.poses):
            for m, person in enumerate(frame_poses.persons[:M]):  # M개까지만
                for v, keypoint in enumerate(person.keypoints):
                    for c in range(min(C, len(keypoint))):
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
        
        for frame_poses in window_data.poses:
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