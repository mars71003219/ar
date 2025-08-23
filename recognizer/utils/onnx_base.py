#!/usr/bin/env python3
"""
ONNX 추론 공통 기능 베이스 클래스

RTMO와 STGCN 등 다양한 ONNX 모델에서 공통으로 사용되는 기능들을 모듈화
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logging.warning("ONNXRuntime not available")


class ONNXInferenceBase:
    """ONNX 추론 공통 기능 베이스 클래스"""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        Args:
            model_path: ONNX 모델 파일 경로
            device: 사용할 디바이스 ('cuda:0', 'cpu' 등)
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNXRuntime is required for ONNX inference")
            
        self.model_path = model_path
        self.device = device
        self.session = None
        self.is_initialized = False
        
        # 성능 최적화용 캐시
        self._input_name = None
        self._output_names = None
        
        # 통계
        self.stats = {
            'total_inferences': 0,
            'processing_times': [],
            'errors': 0
        }
    
    def initialize_session(self) -> bool:
        """ONNX 세션 초기화"""
        try:
            if self.is_initialized:
                return True
                
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
            # CUDA 12 환경변수 설정
            self._setup_cuda_environment()
            
            # 실행 제공자 설정
            providers = self._get_providers()
            
            # 세션 옵션 설정
            session_options = self._create_session_options()
            
            try:
                self.session = ort.InferenceSession(
                    path_or_bytes=self.model_path,
                    providers=providers,
                    sess_options=session_options
                )
                
                # 실제 사용 중인 제공자 확인
                self._validate_providers()
                
            except Exception as e:
                logging.warning(f"지정된 제공자로 세션 생성 실패: {e}")
                logging.info("CPU 제공자로 재시도합니다.")
                
                # CPU 제공자로 다시 시도
                self.session = ort.InferenceSession(
                    path_or_bytes=self.model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=session_options
                )
            
            self.is_initialized = True
            logging.info(f"ONNX model initialized successfully: {self.model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize ONNX model: {str(e)}")
            self.is_initialized = False
            return False
    
    def _setup_cuda_environment(self):
        """CUDA 환경 변수 설정"""
        cuda_lib_path = '/usr/local/cuda-12.1/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64'
        current_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f'{cuda_lib_path}:{current_path}'
        os.environ['CUDA_PATH'] = '/usr/local/cuda-12.1'
    
    def _create_session_options(self) -> ort.SessionOptions:
        """최적화된 세션 옵션 생성"""
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # CUDA 호환성을 위해 그래프 최적화 비활성화
        if self.device.startswith('cuda'):
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.log_severity_level = 3  # 경고 메시지 줄이기
        return session_options
    
    def _get_providers(self) -> List[str]:
        """사용 가능한 실행 공급자 반환"""
        available_providers = ort.get_available_providers()
        
        # 디바이스에 따라 우선순위 설정
        if self.device.startswith('cuda') and 'CUDAExecutionProvider' in available_providers:
            return self._get_cuda_providers()
        elif self.device == 'cpu':
            return ['CPUExecutionProvider']
        else:
            # 기본값: CUDA 사용 가능하면 CUDA, 아니면 CPU
            if 'CUDAExecutionProvider' in available_providers:
                return self._get_cuda_providers()
            else:
                return ['CPUExecutionProvider']
    
    def _get_cuda_providers(self) -> List[str]:
        """CUDA 제공자 설정 반환"""
        # CUDA 디바이스 ID 추출
        device_id = 0
        if self.device.startswith('cuda:'):
            try:
                device_id = int(self.device.split(':')[1])
            except (IndexError, ValueError):
                device_id = 0
        
        # CUDA 제공자 옵션 설정 (호환성 우선)
        cuda_provider_options = {
            'device_id': device_id,
            'arena_extend_strategy': 'kSameAsRequested',  # 메모리 전략 안정화
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB로 제한
            'cudnn_conv_algo_search': 'DEFAULT',  # 기본 알고리즘 사용
            'do_copy_in_default_stream': True,  # 기본 스트림 사용
            'enable_cuda_graph': False  # CUDA 그래프 비활성화
        }
        
        return [
            ('CUDAExecutionProvider', cuda_provider_options),
            'CPUExecutionProvider'
        ]
    
    def _validate_providers(self):
        """실제 사용 중인 제공자 검증"""
        actual_providers = self.session.get_providers()
        if self.device.startswith('cuda') and 'CUDAExecutionProvider' not in actual_providers:
            logging.warning(f"CUDA 제공자를 요청했지만 사용할 수 없습니다. CPU로 실행됩니다.")
            logging.warning(f"실제 사용 제공자: {actual_providers}")
    
    def inference(self, input_data: Dict[str, np.ndarray]) -> Optional[List[np.ndarray]]:
        """ONNX 모델 추론 실행
        
        Args:
            input_data: 입력 데이터 딕셔너리 {input_name: input_tensor}
            
        Returns:
            모델 출력 리스트 또는 None (실패시)
        """
        if not self.is_initialized:
            if not self.initialize_session():
                return None
        
        try:
            import time
            start_time = time.time()
            
            # 입/출력 이름 캐시 (첫 실행시만 계산)
            if self._input_name is None:
                self._input_name = self.session.get_inputs()[0].name
                self._output_names = [output.name for output in self.session.get_outputs()]
            
            # 단일 입력인 경우 자동 변환
            if len(input_data) == 1 and self._input_name not in input_data:
                input_tensor = list(input_data.values())[0]
                input_data = {self._input_name: input_tensor}
            
            # 추론 실행
            outputs = self.session.run(self._output_names, input_data)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['total_inferences'] += 1
            
            return outputs
            
        except Exception as e:
            logging.error(f"ONNX inference failed: {str(e)}")
            self.stats['errors'] += 1
            return None
    
    def warmup(self, input_shape: Tuple[int, ...], num_runs: int = 3) -> bool:
        """모델 워밍업
        
        Args:
            input_shape: 입력 텐서 형태
            num_runs: 워밍업 실행 횟수
            
        Returns:
            워밍업 성공 여부
        """
        if not self.is_initialized:
            if not self.initialize_session():
                return False
        
        try:
            import time
            logging.info(f"ONNX model warmup 시작 ({num_runs}회)")
            
            warmup_times = []
            
            for i in range(num_runs):
                start_time = time.time()
                
                # 더미 데이터 생성
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                input_data = {self._input_name: dummy_input} if self._input_name else {'input': dummy_input}
                
                # 추론 실행
                outputs = self.inference(input_data)
                
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                
                if outputs is not None:
                    logging.info(f"  Warmup {i+1}/{num_runs}: {warmup_time*1000:.2f}ms")
                else:
                    logging.warning(f"  Warmup {i+1}/{num_runs}: {warmup_time*1000:.2f}ms (failed)")
            
            avg_time = sum(warmup_times) / len(warmup_times)
            logging.info(f"Warmup 완료! 평균 시간: {avg_time*1000:.2f}ms")
            return True
            
        except Exception as e:
            logging.error(f"Warmup 실패: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'model_path': self.model_path,
            'device': self.device,
            'is_initialized': self.is_initialized,
            'onnxruntime_available': ONNXRUNTIME_AVAILABLE,
            'statistics': self.stats.copy()
        }
        
        if self.session:
            try:
                info['input_info'] = [
                    {'name': inp.name, 'shape': inp.shape, 'type': inp.type}
                    for inp in self.session.get_inputs()
                ]
                info['output_info'] = [
                    {'name': out.name, 'shape': out.shape, 'type': out.type}
                    for out in self.session.get_outputs()
                ]
                info['providers'] = self.session.get_providers()
            except Exception as e:
                logging.warning(f"모델 정보 조회 실패: {str(e)}")
        
        return info
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['std_processing_time'] = np.std(stats['processing_times'])
            stats['fps_estimate'] = 1.0 / np.mean(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['std_processing_time'] = 0.0
            stats['fps_estimate'] = 0.0
            stats['min_processing_time'] = 0.0
            stats['max_processing_time'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_inferences': 0,
            'processing_times': [],
            'errors': 0
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self.session:
            del self.session
            self.session = None
        
        self.is_initialized = False
        self._input_name = None
        self._output_names = None