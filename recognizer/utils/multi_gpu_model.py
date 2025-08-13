"""
멀티 GPU 모델 로딩 및 처리 유틸리티

PyTorch DataParallel, DistributedDataParallel을 활용한 멀티 GPU 지원을 제공합니다.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import List, Optional, Union, Dict, Any
import logging
import os
from contextlib import contextmanager

from .gpu_manager import get_gpu_manager


class MultiGPUModelWrapper:
    """멀티 GPU 모델 래퍼 클래스"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Args:
            model: 원본 PyTorch 모델
            config: GPU 설정 딕셔너리
                - device: 디바이스 설정
                - use_data_parallel: DataParallel 사용 여부
                - use_distributed: DistributedDataParallel 사용 여부
                - gpu_allocation_strategy: GPU 할당 전략
        """
        self.original_model = model
        self.config = config
        self.wrapped_model = None
        self.device_ids = []
        self.primary_device = 'cpu'
        
        self.gpu_manager = get_gpu_manager()
        self._setup_devices()
        self._wrap_model()
    
    def _setup_devices(self):
        """디바이스 설정"""
        device_config = self.config.get('device', 'cpu')
        component_name = self.config.get('component_name', 'model')
        strategy = self.config.get('gpu_allocation_strategy', 'round_robin')
        
        # GPU 매니저를 통해 디바이스 할당
        allocated_device_ids = self.gpu_manager.allocate_devices(
            component_name, device_config, strategy
        )
        
        if allocated_device_ids:
            self.device_ids = allocated_device_ids
            self.primary_device = f'cuda:{allocated_device_ids[0]}'
        else:
            self.device_ids = []
            self.primary_device = 'cpu'
        
        logging.info(f"Model devices: {self.device_ids}, primary: {self.primary_device}")
    
    def _wrap_model(self):
        """모델 래핑"""
        # 먼저 모델을 주 디바이스로 이동
        self.original_model = self.original_model.to(self.primary_device)
        
        use_data_parallel = self.config.get('use_data_parallel', False)
        use_distributed = self.config.get('use_distributed', False)
        
        if len(self.device_ids) <= 1:
            # 단일 GPU 또는 CPU
            self.wrapped_model = self.original_model
            logging.info(f"Using single device: {self.primary_device}")
            
        elif use_distributed and dist.is_available() and dist.is_initialized():
            # DistributedDataParallel 사용
            self.wrapped_model = DistributedDataParallel(
                self.original_model,
                device_ids=[self.device_ids[0]],
                output_device=self.device_ids[0]
            )
            logging.info(f"Using DistributedDataParallel with devices: {self.device_ids}")
            
        elif use_data_parallel:
            # DataParallel 사용
            self.wrapped_model = DataParallel(
                self.original_model,
                device_ids=self.device_ids,
                output_device=self.device_ids[0]
            )
            logging.info(f"Using DataParallel with devices: {self.device_ids}")
            
        else:
            # 멀티 GPU이지만 병렬 처리 비활성화
            self.wrapped_model = self.original_model
            logging.warning(f"Multi-GPU available {self.device_ids} but parallel processing disabled")
    
    def __call__(self, *args, **kwargs):
        """모델 호출"""
        return self.wrapped_model(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.wrapped_model(*args, **kwargs)
    
    def eval(self):
        """평가 모드 설정"""
        self.wrapped_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """훈련 모드 설정"""
        self.wrapped_model.train(mode)
        return self
    
    def to(self, device):
        """디바이스 이동 (래핑된 모델에서는 제한적)"""
        if isinstance(device, str) and device != self.primary_device:
            logging.warning(f"Model is wrapped for multi-GPU. Cannot move to {device}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'device_ids': self.device_ids,
            'primary_device': self.primary_device,
            'is_multi_gpu': len(self.device_ids) > 1,
            'parallel_type': self._get_parallel_type(),
            'model_type': type(self.original_model).__name__,
            'wrapped_model_type': type(self.wrapped_model).__name__
        }
    
    def _get_parallel_type(self) -> str:
        """병렬 처리 타입 반환"""
        if isinstance(self.wrapped_model, DistributedDataParallel):
            return 'DistributedDataParallel'
        elif isinstance(self.wrapped_model, DataParallel):
            return 'DataParallel'
        else:
            return 'Single'
    
    def get_memory_usage(self) -> Dict[str, float]:
        """GPU 메모리 사용량 반환 (GB 단위)"""
        memory_info = {}
        
        if torch.cuda.is_available():
            for device_id in self.device_ids:
                torch.cuda.set_device(device_id)
                allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                cached = torch.cuda.memory_reserved(device_id) / (1024**3)
                memory_info[f'cuda:{device_id}'] = {
                    'allocated': allocated,
                    'cached': cached
                }
        
        return memory_info
    
    def optimize_memory(self):
        """메모리 최적화"""
        if torch.cuda.is_available():
            for device_id in self.device_ids:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
        
        logging.info("GPU memory cache cleared")
    
    def state_dict(self):
        """상태 딕셔너리 반환"""
        if hasattr(self.wrapped_model, 'module'):
            # DataParallel 또는 DistributedDataParallel의 경우
            return self.wrapped_model.module.state_dict()
        else:
            return self.wrapped_model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """상태 딕셔너리 로드"""
        if hasattr(self.wrapped_model, 'module'):
            # DataParallel 또는 DistributedDataParallel의 경우
            return self.wrapped_model.module.load_state_dict(state_dict, strict=strict)
        else:
            return self.wrapped_model.load_state_dict(state_dict, strict=strict)


class MultiGPUBatchProcessor:
    """멀티 GPU 배치 처리기"""
    
    def __init__(self, models: List[MultiGPUModelWrapper], max_batch_size: int = 8):
        """
        Args:
            models: 멀티 GPU 모델 래퍼 리스트
            max_batch_size: 최대 배치 크기
        """
        self.models = models
        self.max_batch_size = max_batch_size
        self.device_count = sum(len(model.device_ids) for model in models if model.device_ids)
    
    def process_batch(self, data_batch: List[Any], 
                     model_index: int = 0) -> List[Any]:
        """배치 데이터 처리
        
        Args:
            data_batch: 처리할 데이터 배치
            model_index: 사용할 모델 인덱스
            
        Returns:
            처리 결과 리스트
        """
        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} out of range")
        
        model = self.models[model_index]
        
        # 배치 크기 조정
        if len(data_batch) > self.max_batch_size:
            # 큰 배치를 작은 배치로 분할
            results = []
            for i in range(0, len(data_batch), self.max_batch_size):
                sub_batch = data_batch[i:i + self.max_batch_size]
                sub_results = self._process_single_batch(sub_batch, model)
                results.extend(sub_results)
            return results
        else:
            return self._process_single_batch(data_batch, model)
    
    def _process_single_batch(self, data_batch: List[Any], 
                             model: MultiGPUModelWrapper) -> List[Any]:
        """단일 배치 처리"""
        # 데이터를 적절한 디바이스로 이동
        device_data = self._move_data_to_device(data_batch, model.primary_device)
        
        # 모델 추론
        with torch.no_grad():
            results = model(device_data)
        
        # 결과를 CPU로 이동 (필요한 경우)
        cpu_results = self._move_results_to_cpu(results)
        
        return cpu_results
    
    def _move_data_to_device(self, data: Any, device: str) -> Any:
        """데이터를 특정 디바이스로 이동"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, list):
            return [self._move_data_to_device(item, device) for item in data]
        elif isinstance(data, dict):
            return {key: self._move_data_to_device(value, device) for key, value in data.items()}
        else:
            return data
    
    def _move_results_to_cpu(self, results: Any) -> Any:
        """결과를 CPU로 이동"""
        if isinstance(results, torch.Tensor):
            return results.cpu()
        elif isinstance(results, list):
            return [self._move_results_to_cpu(item) for item in results]
        elif isinstance(results, dict):
            return {key: self._move_results_to_cpu(value) for key, value in results.items()}
        else:
            return results


def setup_distributed_training(rank: int, world_size: int, backend: str = 'nccl'):
    """분산 훈련 설정
    
    Args:
        rank: 현재 프로세스 랭크
        world_size: 전체 프로세스 수
        backend: 통신 백엔드
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if backend == 'nccl' and not torch.cuda.is_available():
        backend = 'gloo'  # CPU 전용 환경에서는 gloo 사용
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    logging.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")


def cleanup_distributed_training():
    """분산 훈련 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Distributed training cleaned up")


@contextmanager
def multi_gpu_context(device_ids: List[int]):
    """멀티 GPU 컨텍스트 매니저
    
    Args:
        device_ids: 사용할 GPU ID 리스트
    """
    if not torch.cuda.is_available() or not device_ids:
        yield
        return
    
    # 원래 디바이스 저장
    original_device = torch.cuda.current_device()
    
    try:
        # 주 디바이스 설정
        torch.cuda.set_device(device_ids[0])
        
        # PyTorch 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        yield
        
    finally:
        # 원래 디바이스 복원
        torch.cuda.set_device(original_device)
        
        # 메모리 정리
        for device_id in device_ids:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()


def create_multi_gpu_model(model: nn.Module, config: Dict[str, Any]) -> MultiGPUModelWrapper:
    """멀티 GPU 모델 생성 편의 함수
    
    Args:
        model: 원본 PyTorch 모델
        config: GPU 설정
        
    Returns:
        멀티 GPU 모델 래퍼
    """
    return MultiGPUModelWrapper(model, config)


def get_optimal_batch_size(model: nn.Module, input_shape: tuple, 
                          device_ids: List[int], max_memory_ratio: float = 0.8) -> int:
    """최적 배치 크기 계산
    
    Args:
        model: PyTorch 모델
        input_shape: 입력 데이터 형태
        device_ids: GPU ID 리스트
        max_memory_ratio: 최대 메모리 사용 비율
        
    Returns:
        권장 배치 크기
    """
    if not torch.cuda.is_available() or not device_ids:
        return 1
    
    # 가장 메모리가 적은 GPU를 기준으로 계산
    min_memory = float('inf')
    for device_id in device_ids:
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / (1024**3)  # GB
        min_memory = min(min_memory, total_memory)
    
    # 추정 메모리 사용량 (매우 단순한 추정)
    # 실제로는 더 정교한 계산이 필요
    input_size = 1
    for dim in input_shape:
        input_size *= dim
    
    # 4바이트 float32 기준, 포워드/백워드 고려
    estimated_memory_per_sample = (input_size * 4 * 3) / (1024**3)  # GB
    
    available_memory = min_memory * max_memory_ratio
    optimal_batch_size = max(1, int(available_memory / estimated_memory_per_sample))
    
    return min(optimal_batch_size, 32)  # 최대 32로 제한