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
        
        # 워밍업 관련
        self.is_warmed_up = False
        self.warmup_runs = 1  # 워밍업 실행 횟수 (에러 방지를 위해 1번만)
    
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
            
            # 모델 워밍업 실행
            logging.info("Starting STGCN model warmup...")
            self._warmup_model()
            
            # GPU 메모리 최적화 
            self._optimize_gpu_performance()
            
            self.is_initialized = True
            logging.info("ST-GCN++ classifier initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize ST-GCN++ classifier: {str(e)}")
            self.is_initialized = False
            return False
    
    def _warmup_model(self):
        """모델 워밍업을 통한 Cold Start 해결"""
        try:
            import time
            logging.info(f"Performing {self.warmup_runs} warmup iterations...")
            
            # 더미 데이터 생성 (실제 입력과 동일한 형태)
            # 각 warmup마다 새로운 더미 데이터 생성하여 메모리 충돌 방지
            
            warmup_times = []
            
            with torch.no_grad():
                for i in range(self.warmup_runs):
                    start_time = time.time()
                    
                    # 각 warmup마다 새로운 더미 데이터 생성
                    dummy_data = {
                        'keypoint': np.random.randn(4, 100, 17, 2).astype(np.float32),  # (M, T, V, C)
                        'keypoint_score': np.random.rand(4, 100, 17).astype(np.float32),  # (M, T, V)
                        'total_frames': 100,
                        'img_shape': (640, 640),
                        'original_shape': (640, 640),
                        'label': 0
                    }
                    
                    try:
                        # MMAction2 추론 실행
                        logging.debug(f"Warmup {i+1}: Starting inference_recognizer call")
                        result = inference_recognizer(self.recognizer, dummy_data)
                        logging.debug(f"Warmup {i+1}: inference_recognizer completed successfully")
                        
                        warmup_time = time.time() - start_time
                        warmup_times.append(warmup_time)
                        logging.info(f"  Warmup {i+1}/{self.warmup_runs}: {warmup_time*1000:.2f}ms")
                        
                        # 메모리 정리 - 각 워밍업 후 즉시 정리
                        del result
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        warmup_time = time.time() - start_time
                        warmup_times.append(warmup_time)
                        import traceback
                        full_traceback = traceback.format_exc()
                        logging.warning(f"  Warmup {i+1}/{self.warmup_runs}: {warmup_time*1000:.2f}ms")
                        logging.warning(f"  Error: {str(e)}")
                        logging.debug(f"  Full traceback:\n{full_traceback}")
                        
                        # 에러 발생해도 메모리 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 각 warmup 후 더미 데이터 정리
                    del dummy_data
            
            # warmup 완료 후 최종 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 최종 GPU 메모리 동기화 및 정리
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            avg_warmup_time = sum(warmup_times) / len(warmup_times)
            logging.info(f"Warmup completed! Average time: {avg_warmup_time*1000:.2f}ms")
            logging.info(f"Final warmup time: {warmup_times[-1]*1000:.2f}ms ({1.0/warmup_times[-1]:.1f} FPS)")
            
            self.is_warmed_up = True
            
        except Exception as e:
            logging.warning(f"Model warmup failed, but continuing: {str(e)}")
            self.is_warmed_up = False
    
    def _optimize_gpu_performance(self):
        """GPU 성능 최적화"""
        try:
            import time
            if torch.cuda.is_available() and 'cuda' in str(self.device):
                logging.info("Optimizing GPU performance...")
                
                # CUDA 컨텍스트 최적화
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 모델을 evaluation mode로 설정하여 최적화 활성화
                self.recognizer.eval()
                
                # PyTorch 성능 최적화 설정
                torch.backends.cudnn.benchmark = True  # cuDNN 자동 튜닝
                torch.backends.cudnn.deterministic = False  # 성능 우선
                
                # GPU 메모리 풀 설정
                torch.cuda.set_per_process_memory_fraction(0.8)  # 메모리 사용량 제한
                
                # 추가 워밍업 (GPU 커널 최적화 완료까지)
                extended_warmup_times = []
                dummy_data = {
                    'keypoint': np.random.randn(4, 100, 17, 2).astype(np.float32),
                    'keypoint_score': np.random.rand(4, 100, 17).astype(np.float32),
                    'total_frames': 100,
                    'img_shape': (640, 640),
                    'original_shape': (640, 640),
                    'label': 0
                }
                
                with torch.no_grad():
                    # 안전한 연속 실행으로 커널 최적화
                    successful_runs = 0
                    target_runs = 5
                    
                    for i in range(target_runs):
                        start_time = time.time()
                        try:
                            # 첫 번째 실행 후에는 다른 데이터로 테스트
                            if i == 0:
                                result = inference_recognizer(self.recognizer, dummy_data)
                            else:
                                # 약간 다른 dummy 데이터 생성 (랜덤 시드 변경)
                                np.random.seed(42 + i)
                                test_data = {
                                    'keypoint': np.random.randn(4, 100, 17, 2).astype(np.float32),
                                    'keypoint_score': np.random.rand(4, 100, 17).astype(np.float32),
                                    'total_frames': 100,
                                    'img_shape': (640, 640),
                                    'original_shape': (640, 640),
                                    'label': 0
                                }
                                result = inference_recognizer(self.recognizer, test_data)
                            
                            warmup_time = time.time() - start_time
                            extended_warmup_times.append(warmup_time)
                            successful_runs += 1
                            logging.info(f"  Extended warmup {i+1}/{target_runs}: {warmup_time*1000:.2f}ms - Success")
                            
                        except Exception as e:
                            warmup_time = time.time() - start_time
                            extended_warmup_times.append(warmup_time)
                            logging.warning(f"  Extended warmup {i+1}/{target_runs}: {warmup_time*1000:.2f}ms (with error: {str(e)})")
                            
                            # 에러가 발생한 경우 모델 상태 초기화 시도
                            try:
                                if hasattr(self.recognizer, 'backbone'):
                                    if hasattr(self.recognizer.backbone, 'eval'):
                                        self.recognizer.backbone.eval()
                                if hasattr(self.recognizer, 'cls_head'):
                                    if hasattr(self.recognizer.cls_head, 'eval'):
                                        self.recognizer.cls_head.eval()
                                # GPU 메모리 정리
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except:
                                pass
                
                # 성능 안정화 확인 (성공한 실행만 고려)
                if extended_warmup_times:
                    recent_avg = sum(extended_warmup_times[-min(3, len(extended_warmup_times)):]) / min(3, len(extended_warmup_times))
                    logging.info(f"GPU optimization completed! Successful runs: {successful_runs}/{target_runs}, Stable performance: {recent_avg*1000:.2f}ms ({1.0/recent_avg:.1f} FPS)")
                else:
                    logging.warning("GPU optimization failed - no successful runs")
                
                # 메모리 최종 정리
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logging.warning(f"GPU performance optimization failed: {str(e)}")
    
    def classify_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류 - 파이프라인 호환용 메서드명"""
        logging.info(f"STGCN classify_window called for window {window_data.window_idx}")
        return self.classify_single_window(window_data)
    
    def classify_single_window(self, window_data: WindowAnnotation) -> ClassificationResult:
        """단일 윈도우 분류"""
        import time
        total_start = time.time()
        
        # STGCN이 처리할 때마다 윈도우 카운터 증가
        self.window_counter += 1
        
        logging.info(f"STGCN processing window {self.window_counter} (window_idx: {window_data.window_idx})")
        # 자동 초기화
        if not self.ensure_initialized():
            raise RuntimeError("Failed to initialize STGCN model")
        
        try:
            # 1. 윈도우 데이터 전처리
            preprocess_start = time.time()
            preprocessed_data = self.preprocess_window_data(window_data)
            preprocess_time = time.time() - preprocess_start
            
            if preprocessed_data is None:
                logging.warning(f"Preprocessing failed for STGCN window {self.window_counter}")
                return self._create_error_result(self.window_counter, "Preprocessing failed")
            
            # 2. ST-GCN++ 입력 형태로 변환
            prepare_start = time.time()
            stgcn_input = self._prepare_stgcn_input(preprocessed_data)
            prepare_time = time.time() - prepare_start
            
            if stgcn_input is None:
                return self._create_error_result(self.window_counter, "ST-GCN++ input preparation failed")
            
            # 3. MMAction2 추론 실행
            inference_start = time.time()
            with torch.no_grad():
                result = inference_recognizer(self.recognizer, stgcn_input)
            inference_time = time.time() - inference_start
            
            # 중간 데이터 정리
            del stgcn_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 4. 결과 후처리
            postprocess_start = time.time()
            final_result = self._process_inference_result(self.window_counter, result)
            postprocess_time = time.time() - postprocess_start
            
            # 추론 결과 정리
            del result
            
            total_time = time.time() - total_start
            
            # 상세한 성능 로깅
            logging.info(f"STGCN Window {self.window_counter} Performance Breakdown:")
            logging.info(f"  - Preprocess: {preprocess_time*1000:.2f}ms")
            logging.info(f"  - Input prep: {prepare_time*1000:.2f}ms")
            logging.info(f"  - Inference:  {inference_time*1000:.2f}ms")
            logging.info(f"  - Postprocess: {postprocess_time*1000:.2f}ms")
            logging.info(f"  - Total:      {total_time*1000:.2f}ms ({1.0/total_time:.1f} FPS)")
            
            return final_result
            
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
            
            logging.info(f"Attempting to unpack data.shape: {data.shape}")
            try:
                C, T, V, M = data.shape
                logging.info(f"Successfully unpacked dimensions: C={C}, T={T}, V={V}, M={M}")
            except ValueError as unpack_err:
                logging.error(f"Failed to unpack data.shape {data.shape}: {unpack_err}")
                raise
            
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
        result = ClassificationResult(
            prediction=0,  # 기본값으로 NonFight
            confidence=0.0,
            probabilities=[1.0, 0.0],  # [NonFight_prob, Fight_prob]
            model_name='stgcn'
        )
        
        # 에러 결과에도 메타데이터 추가
        if not hasattr(result, 'metadata'):
            result.metadata = {}
        result.metadata = {'display_id': window_id, 'error': error_msg}
        
        return result
    
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