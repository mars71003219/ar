#!/usr/bin/env python3
"""
STGCN ONNX 행동 분류기 구현

ONNX 공통 베이스 클래스를 상속받아 STGCN 전용 기능을 구현
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
import time

try:
    from action_classification.base import BaseActionClassifier
    from utils.data_structure import WindowAnnotation, ClassificationResult, ActionClassificationConfig
    from utils.onnx_base import ONNXInferenceBase
except ImportError:
    from ..base import BaseActionClassifier
    from ...utils.data_structure import WindowAnnotation, ClassificationResult, ActionClassificationConfig
    from ...utils.onnx_base import ONNXInferenceBase

# data_utils 모듈 사용
try:
    from .data_utils import convert_to_stgcn_format, convert_poses_to_stgcn_format
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    DATA_UTILS_AVAILABLE = False
    logging.warning(f"Data utils not available: {e}")


class STGCNONNXClassifier(BaseActionClassifier):
    """STGCN ONNX 행동 분류기"""
    
    def __init__(self, config: ActionClassificationConfig):
        """
        Args:
            config: STGCN ONNX 분류 설정
        """
        super().__init__(config)
        
        # STGCN 관련 설정
        self.device = config.device or 'cuda:0'
        self.num_classes = len(self.class_names)
        
        # 전처리 관련
        self.clip_len = config.window_size
        self.max_persons = config.max_persons or 4
        self.coordinate_format = config.coordinate_format or 'xyz'  # ONNX 모델은 3D 좌표 기대
        
        # ONNX 추론 베이스 클래스
        self.onnx_inferencer = ONNXInferenceBase(
            model_path=self.checkpoint_path,
            device=self.device
        )
        
        # 윈도우 카운터
        self.window_counter = 0
        
        # 워밍업 관련
        self.is_warmed_up = False
        self.warmup_runs = 3
    
    def initialize_model(self) -> bool:
        """STGCN ONNX 모델 초기화"""
        try:
            if self.is_initialized:
                return True
            
            # ONNX 세션 초기화
            if not self.onnx_inferencer.initialize_session():
                logging.error("Failed to initialize ONNX session")
                return False
            
            # 모델 워밍업
            logging.info("Starting STGCN ONNX model warmup...")
            self._warmup_model()
            
            self.is_initialized = True
            logging.info("STGCN ONNX classifier initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize STGCN ONNX classifier: {str(e)}")
            self.is_initialized = False
            return False
    
    def _warmup_model(self):
        """모델 워밍업"""
        try:
            # ONNX 모델의 예상 입력 형태: [batch_size, num_person, num_frames, 17, 3]
            input_shape = (1, self.max_persons, self.clip_len, 17, 3)
            
            success = self.onnx_inferencer.warmup(
                input_shape=input_shape,
                num_runs=self.warmup_runs
            )
            
            if success:
                self.is_warmed_up = True
                logging.info("STGCN ONNX warmup completed successfully")
            else:
                logging.warning("STGCN ONNX warmup failed, but continuing")
                self.is_warmed_up = False
                
        except Exception as e:
            logging.warning(f"Model warmup failed: {str(e)}")
            self.is_warmed_up = False
    
    def classify_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류 - 파이프라인 호환용"""
        logging.info(f"STGCN ONNX classify_window called for window {window_data.window_idx}")
        return self.classify_single_window(window_data)
    
    def classify_single_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류"""
        total_start = time.time()
        
        # 윈도우 카운터 증가
        self.window_counter += 1
        
        logging.info(f"STGCN ONNX processing window {self.window_counter} (window_idx: {window_data.window_idx})")
        
        # 자동 초기화
        if not self.ensure_initialized():
            raise RuntimeError("Failed to initialize STGCN ONNX model")
        
        try:
            # 1. 윈도우 데이터 전처리
            preprocess_start = time.time()
            preprocessed_data = self.preprocess_window_data(window_data)
            preprocess_time = time.time() - preprocess_start
            
            if preprocessed_data is None:
                logging.warning(f"Preprocessing failed for STGCN ONNX window {self.window_counter}")
                return self._create_error_result(self.window_counter, "Preprocessing failed")
            
            # 2. STGCN ONNX 입력 형태로 변환
            prepare_start = time.time()
            stgcn_input = self._prepare_stgcn_onnx_input(preprocessed_data)
            prepare_time = time.time() - prepare_start
            
            if stgcn_input is None:
                return self._create_error_result(self.window_counter, "STGCN ONNX input preparation failed")
            
            # 3. ONNX 추론 실행
            inference_start = time.time()
            outputs = self.onnx_inferencer.inference({'input_tensor': stgcn_input})
            inference_time = time.time() - inference_start
            
            if outputs is None:
                return self._create_error_result(self.window_counter, "ONNX inference failed")
            
            # 4. 결과 후처리
            postprocess_start = time.time()
            final_result = self._process_onnx_result(self.window_counter, outputs)
            postprocess_time = time.time() - postprocess_start
            
            total_time = time.time() - total_start
            
            # 성능 로깅
            logging.info(f"STGCN ONNX Window {self.window_counter} Performance:")
            logging.info(f"  - Preprocess: {preprocess_time*1000:.2f}ms")
            logging.info(f"  - Input prep: {prepare_time*1000:.2f}ms")
            logging.info(f"  - ONNX inference: {inference_time*1000:.2f}ms")
            logging.info(f"  - Postprocess: {postprocess_time*1000:.2f}ms")
            logging.info(f"  - Total: {total_time*1000:.2f}ms ({1.0/total_time:.1f} FPS)")
            
            return final_result
            
        except Exception as e:
            import traceback
            logging.error(f"Error in STGCN ONNX classification for window {self.window_counter}: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return self._create_error_result(self.window_counter, str(e))
    
    def _prepare_stgcn_onnx_input(self, preprocessed_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """STGCN ONNX 입력 형태로 변환"""
        try:
            if 'keypoint' not in preprocessed_data:
                logging.error("No keypoint data in preprocessed_data")
                return None
            
            keypoint_data = preprocessed_data['keypoint']
            logging.info(f"Converting keypoint data shape: {keypoint_data.shape}")
            
            # keypoint_data가 이미 올바른 형태인지 확인
            # 예상 형태: [M, T, V, C] 또는 [1, M, T, V, C]
            if keypoint_data.ndim == 4:
                # [M, T, V, C] -> [1, M, T, V, C] (배치 차원 추가)
                keypoint_data = np.expand_dims(keypoint_data, axis=0)
            elif keypoint_data.ndim == 5:
                # 이미 배치 차원이 있음
                pass
            else:
                logging.error(f"Unexpected keypoint data shape: {keypoint_data.shape}")
                return None
            
            # 차원 확인: [batch_size, num_person, num_frames, num_joints, coords]
            batch_size, num_person, num_frames, num_joints, coords = keypoint_data.shape
            
            # ONNX 모델이 기대하는 형태 확인
            expected_joints = 17  # COCO 형식
            expected_coords = 3   # x, y, z
            
            if num_joints != expected_joints:
                logging.warning(f"Joint count mismatch: got {num_joints}, expected {expected_joints}")
            
            if coords == 2:
                # 2D 좌표를 3D로 확장 (z=0 추가)
                logging.info("Converting 2D coordinates to 3D by adding z=0")
                z_coords = np.zeros((batch_size, num_person, num_frames, num_joints, 1))
                keypoint_data = np.concatenate([keypoint_data, z_coords], axis=-1)
            elif coords != expected_coords:
                logging.warning(f"Coordinate count mismatch: got {coords}, expected {expected_coords}")
            
            # float32로 변환
            keypoint_data = keypoint_data.astype(np.float32)
            
            logging.info(f"Final ONNX input shape: {keypoint_data.shape}")
            return keypoint_data
            
        except Exception as e:
            import traceback
            logging.error(f"Error preparing STGCN ONNX input: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _process_onnx_result(self, window_id: int, outputs: List[np.ndarray]) -> ClassificationResult:
        """ONNX 추론 결과 후처리"""
        try:
            if not outputs or len(outputs) == 0:
                logging.error("Empty ONNX outputs")
                return self._create_error_result(window_id, "Empty ONNX outputs")
            
            # 첫 번째 출력이 클래스 점수
            pred_scores = outputs[0]
            
            # 배치 차원 제거 (배치 크기가 1인 경우)
            if pred_scores.ndim > 1 and pred_scores.shape[0] == 1:
                pred_scores = pred_scores[0]
            
            logging.info(f"Processing ONNX result for window {window_id}: raw_scores={pred_scores}")
            
            # Softmax 적용하여 확률로 변환
            exp_scores = np.exp(pred_scores - np.max(pred_scores))  # 수치 안정성을 위해 최대값 빼기
            probabilities = exp_scores / np.sum(exp_scores)
            
            logging.info(f"Probabilities after softmax for window {window_id}: {probabilities}")
            
            # 클래스 예측
            if len(probabilities) >= 2:
                fight_prob = float(probabilities[1])
                nonfight_prob = float(probabilities[0])
                predicted_class_idx = 1 if fight_prob > nonfight_prob else 0
                confidence = fight_prob if predicted_class_idx == 1 else nonfight_prob
            else:
                # 단일 점수인 경우
                fight_prob = float(probabilities[0])
                nonfight_prob = 1.0 - fight_prob
                predicted_class_idx = 1 if fight_prob > 0.5 else 0
                confidence = fight_prob if predicted_class_idx == 1 else nonfight_prob
            
            # 클래스 이름
            predicted_class = self.class_names[predicted_class_idx] if predicted_class_idx < len(self.class_names) else 'Unknown'
            
            logging.info(f"STGCN ONNX RESULT - Window {window_id}: {predicted_class} ({confidence:.3f})")
            
            # ClassificationResult 생성
            result = ClassificationResult(
                prediction=predicted_class_idx,
                confidence=confidence,
                probabilities=probabilities.tolist(),
                model_name='stgcn_onnx'
            )
            
            # 메타데이터 추가
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata = {'display_id': window_id}
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing ONNX result for window {window_id}: {str(e)}")
            return self._create_error_result(window_id, f"Result processing error: {str(e)}")
    
    def classify_multiple_windows(self, windows: List[WindowAnnotation]) -> List[ClassificationResult]:
        """다중 윈도우 분류"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        if not windows:
            return []
        
        results = []
        
        # 개별 처리 (ONNX는 일반적으로 단일 입력 처리에 최적화)
        for window in windows:
            result = self.classify_single_window(window)
            results.append(result)
        
        # 통계 업데이트
        self.update_statistics(results)
        
        return results
    
    def _create_error_result(self, window_id: int, error_msg: str) -> ClassificationResult:
        """에러 결과 생성"""
        result = ClassificationResult(
            prediction=0,  # 기본값으로 NonFight
            confidence=0.0,
            probabilities=[1.0, 0.0],  # [NonFight_prob, Fight_prob]
            model_name='stgcn_onnx'
        )
        
        # 에러 메타데이터 추가
        if not hasattr(result, 'metadata'):
            result.metadata = {}
        result.metadata = {'display_id': window_id, 'error': error_msg}
        
        return result
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """분류기 정보 반환"""
        base_info = super().get_classifier_info()
        
        base_info.update({
            'classifier_type': 'stgcn_onnx',
            'device': self.device,
            'max_persons': self.max_persons,
            'coordinate_format': self.coordinate_format,
            'data_utils_available': DATA_UTILS_AVAILABLE,
            'onnx_model_info': self.onnx_inferencer.get_model_info(),
            'onnx_stats': self.onnx_inferencer.get_processing_stats()
        })
        
        return base_info
    
    def set_device(self, device: str):
        """디바이스 설정"""
        self.device = device
        # ONNX 추론기의 디바이스도 업데이트
        self.onnx_inferencer.device = device
        # 세션 재초기화 필요
        self.onnx_inferencer.cleanup()
        self.is_initialized = False
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        if self.onnx_inferencer:
            self.onnx_inferencer.cleanup()
    
    def reset_stats(self):
        """통계 초기화"""
        super().reset_stats()
        if self.onnx_inferencer:
            self.onnx_inferencer.reset_stats()