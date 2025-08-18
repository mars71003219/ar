"""
ST-GCN++ 기반 행동 분류기 구현

기존 rtmo_gcn_pipeline의 ST-GCN++ 모델을 새로운 표준 인터페이스에 맞게 재구성한 버전입니다.
MMAction2의 inference API를 활용하여 구현되었습니다.
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import logging

# MMAction2 경로 추가
mmaction_path = "/home/gaonpf/hsnam/mmlabs/mmaction2"
if mmaction_path not in sys.path:
    sys.path.insert(0, mmaction_path)

try:
    from mmaction.apis import init_recognizer, inference_recognizer
    from mmaction.structures import ActionDataSample
    import mmengine
    MMACTION_AVAILABLE = True
except ImportError as e:
    MMACTION_AVAILABLE = False
    logging.warning(f"MMAction2 not available: {e}")

# 이전한 data_utils 모듈 사용
try:
    from .data_utils import convert_to_stgcn_format, convert_poses_to_stgcn_format
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    DATA_UTILS_AVAILABLE = False
    logging.warning(f"Data utils not available: {e}")

try:
    from action_classification.base import BaseActionClassifier
    from utils.data_structure import WindowAnnotation, ClassificationResult, ActionClassificationConfig
except ImportError:
    from ..base import BaseActionClassifier
    from ...utils.data_structure import WindowAnnotation, ClassificationResult, ActionClassificationConfig


class STGCNActionClassifier(BaseActionClassifier):
    """ST-GCN++ 기반 행동 분류기"""
    
    def __init__(self, config: ActionClassificationConfig):
        """
        Args:
            config: ST-GCN++ 분류 설정
        """
        super().__init__(config)
        
        # ST-GCN++ 관련 설정
        self.device = config.device or 'cuda:0'
        self.config_file = config.config_file
        self.num_classes = len(self.class_names)
        
        # 전처리 관련
        self.clip_len = config.window_size
        self.max_persons = config.max_persons or 2
        self.coordinate_format = config.coordinate_format or 'xy'
        
        # 성능 최적화 관련
        self.batch_inference = getattr(config, 'batch_inference', False)
        self.max_batch_size = getattr(config, 'max_batch_size', 8)
        
        # MMAction2 모델
        self.recognizer = None
        
        # 윈도우 카운터 (STGCN 내부에서 관리)
        self.window_counter = 0
    
    def initialize_model(self) -> bool:
        """ST-GCN++ 모델 초기화"""
        try:
            if self.is_initialized:
                return True
            
            if not MMACTION_AVAILABLE:
                logging.error("MMAction2 not available for ST-GCN++ classifier")
                return False
            
            if not os.path.exists(self.config_file):
                logging.error(f"Config file not found: {self.config_file}")
                return False
            
            if not os.path.exists(self.model_path):
                logging.error(f"Model checkpoint not found: {self.model_path}")
                return False
            
            # MMAction2 모델 초기화
            logging.info(f"Loading ST-GCN++ model from {self.model_path}")
            self.recognizer = init_recognizer(
                config=self.config_file,
                checkpoint=self.model_path,
                device=self.device
            )
            
            self.is_initialized = True
            logging.info("ST-GCN++ classifier initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize ST-GCN++ classifier: {str(e)}")
            self.is_initialized = False
            return False
    
    def classify_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류 - 파이프라인 호환용 메서드명"""
        logging.info(f"STGCN classify_window called for window {window_data.window_idx}")
        return self.classify_single_window(window_data)
    
    def classify_single_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류"""
        # STGCN이 처리할 때마다 윈도우 카운터 증가
        self.window_counter += 1
        
        logging.info(f"STGCN processing window {self.window_counter} (window_idx: {window_data.window_idx})")
        # 자동 초기화
        if not self.ensure_initialized():
            raise RuntimeError("Failed to initialize STGCN model")
        
        try:
            # 윈도우 데이터 전처리
            preprocessed_data = self.preprocess_window_data(window_data)
            if preprocessed_data is None:
                logging.warning(f"Preprocessing failed for STGCN window {self.window_counter}")
                return self._create_error_result(self.window_counter, "Preprocessing failed")
            
            # ST-GCN++ 입력 형태로 변환
            stgcn_input = self._prepare_stgcn_input(preprocessed_data)
            if stgcn_input is None:
                return self._create_error_result(self.window_counter, "ST-GCN++ input preparation failed")
            
            # MMAction2 추론 실행
            with torch.no_grad():
                result = inference_recognizer(self.recognizer, stgcn_input)
                
            # 결과 후처리 (윈도우 번호 포함)
            return self._process_inference_result(self.window_counter, result)
            
        except Exception as e:
            import traceback
            logging.error(f"Error in ST-GCN++ classification for window {self.window_counter}: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_result(self.window_counter, str(e))
    
    def classify_multiple_windows(self, windows: List[WindowAnnotation]) -> List[ClassificationResult]:
        """다중 윈도우 분류 (배치 처리 지원)"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        if not windows:
            return []
        
        results = []
        
        if self.batch_inference and len(windows) > 1:
            # 배치 처리
            results = self._classify_batch_windows(windows)
        else:
            # 개별 처리
            for window in windows:
                result = self.classify_single_window(window)
                results.append(result)
        
        # 통계 업데이트
        self.update_statistics(results)
        
        return results
    
    def _classify_batch_windows(self, windows: List[WindowAnnotation]) -> List[ClassificationResult]:
        """배치 윈도우 분류"""
        results = []
        
        # 배치 크기로 나누어 처리
        for i in range(0, len(windows), self.max_batch_size):
            batch_windows = windows[i:i + self.max_batch_size]
            batch_results = self._process_window_batch(batch_windows)
            results.extend(batch_results)
        
        return results
    
    def _process_window_batch(self, batch_windows: List[WindowAnnotation]) -> List[ClassificationResult]:
        """윈도우 배치 처리"""
        batch_results = []
        
        # 각 윈도우 개별 처리 (MMAction2는 기본적으로 단일 입력 처리)
        for window in batch_windows:
            result = self.classify_single_window(window)
            batch_results.append(result)
        
        return batch_results
    
    def _prepare_stgcn_input(self, preprocessed_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ST-GCN++ 입력 형태로 변환 (이미 MMAction2 data_sample 형태)"""
        try:
            # 전처리에서 이미 MMAction2 data_sample 형태로 변환되었으므로 그대로 사용
            if isinstance(preprocessed_data, dict) and 'keypoint' in preprocessed_data:
                logging.info(f"Using preprocessed data_sample with keypoint shape: {preprocessed_data['keypoint'].shape}")
                return preprocessed_data
            else:
                logging.error(f"Invalid preprocessed_data type or format: {type(preprocessed_data)}")
                return None
            
        except Exception as e:
            import traceback
            logging.error(f"Error preparing ST-GCN++ input: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _convert_to_stgcn_basic(self, data: np.ndarray) -> Dict[str, Any]:
        """기본 ST-GCN++ 변환 구현"""
        try:
            logging.info(f"ST-GCN input data shape: {data.shape}, ndim: {data.ndim}, dtype: {data.dtype}")
            
            # data shape: (C, T, V, M)
            if data.ndim != 4:
                raise ValueError(f"Expected 4D array, got {data.ndim}D with shape {data.shape}")
            
            C, T, V, M = data.shape
            logging.info(f"Unpacked dimensions: C={C}, T={T}, V={V}, M={M}")
            
            # 최소 차원 검증
            if C == 0 or T == 0 or V == 0 or M == 0:
                raise ValueError(f"Invalid dimensions: C={C}, T={T}, V={V}, M={M}")
            
            # 배치 차원 추가: (C, T, V, M) -> (1, C, T, V, M)
            stgcn_data = np.expand_dims(data, axis=0)
        except (ValueError, IndexError) as e:
            logging.error(f"Error in basic ST-GCN conversion: {str(e)}")
            # 기본값으로 더미 데이터 생성
            stgcn_data = np.zeros((1, 3, 100, 17, 2))
        
        return {
            'keypoint': stgcn_data,  # 이미 (1, C, T, V, M) 형태
            'total_frames': T,
            'img_shape': (640, 640),  # 기본값
            'original_shape': (640, 640),
            'label': 0
        }
    
    def _process_inference_result(self, window_id: int, result: ActionDataSample) -> ClassificationResult:
        """MMAction2 추론 결과 후처리 (기존 rtmo_gcn_pipeline과 동일한 방식)"""
        try:
            # 예측 점수 추출
            pred_scores = result.pred_score
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            
            logging.info(f"Processing inference result for window {window_id}: scores={pred_scores}")
            
            # 기존 rtmo_gcn_pipeline과 동일한 방식으로 처리
            # scores[1]이 Fight 점수, scores[0]이 NonFight 점수
            if len(pred_scores) > 1:
                fight_score = float(pred_scores[1])
                nonfight_score = float(pred_scores[0])
                predicted_class_idx = 1 if fight_score > nonfight_score else 0
                confidence = fight_score if predicted_class_idx == 1 else nonfight_score
            else:
                # 점수가 1개만 있는 경우 (이상한 상황)
                fight_score = float(pred_scores[0])
                nonfight_score = 1.0 - fight_score
                predicted_class_idx = 1 if fight_score > 0.5 else 0
                confidence = fight_score if predicted_class_idx == 1 else nonfight_score
            
            # 클래스 이름 결정
            predicted_class = self.class_names[predicted_class_idx] if predicted_class_idx < len(self.class_names) else 'Unknown'
            
            logging.info(f"STGCN RESULT - Window {window_id}: {predicted_class} ({confidence:.3f})")
            
            # ClassificationResult 생성 (윈도우 번호 메타데이터에 추가)
            result = ClassificationResult(
                prediction=predicted_class_idx,
                confidence=confidence,
                probabilities=pred_scores.tolist(),
                model_name='stgcn'
            )
            
            # 윈도우 번호를 결과에 저장 (메타데이터로)
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata = {'display_id': window_id}
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing inference result for window {window_id}: {str(e)}")
            return self._create_error_result(window_id, f"Result processing error: {str(e)}")
    
    def _create_error_result(self, window_id: int, error_msg: str) -> ClassificationResult:
        """에러 결과 생성"""
        return ClassificationResult(
            prediction=0,  # 기본값으로 NonFight
            confidence=0.0,
            probabilities=[1.0, 0.0],  # [NonFight_prob, Fight_prob]
            model_name='stgcn'
        )
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """분류기 정보 반환"""
        base_info = super().get_classifier_info()
        
        base_info.update({
            'classifier_type': 'stgcn_plus_plus',
            'config_file': self.config_file,
            'device': self.device,
            'max_persons': self.max_persons,
            'coordinate_format': self.coordinate_format,
            'batch_inference': self.batch_inference,
            'max_batch_size': self.max_batch_size,
            'mmaction_available': MMACTION_AVAILABLE,
            'data_utils_available': DATA_UTILS_AVAILABLE
        })
        
        return base_info
    
    def set_device(self, device: str):
        """디바이스 설정
        
        Args:
            device: 새로운 디바이스 ('cuda:0', 'cpu' 등)
        """
        self.device = device
        if self.recognizer is not None:
            self.recognizer.to(device)
    
    def set_batch_inference(self, enable: bool, batch_size: Optional[int] = None):
        """배치 추론 설정
        
        Args:
            enable: 배치 추론 활성화 여부
            batch_size: 배치 크기 (None이면 기존 값 유지)
        """
        self.batch_inference = enable
        if batch_size is not None:
            self.max_batch_size = max(1, batch_size)
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        if self.recognizer is not None:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.recognizer = None