"""
GPU 관리 유틸리티

멀티 GPU 환경에서 디바이스 할당 및 로드 밸런싱을 담당합니다.
"""

import torch
import logging
import re
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU 정보"""
    device_id: int
    device_name: str
    memory_total: float  # GB
    memory_available: float  # GB
    utilization: float  # 0-100%
    is_available: bool = True


class GPUManager:
    """GPU 관리자"""
    
    def __init__(self):
        self.available_devices: List[int] = []
        self.device_info: Dict[int, GPUInfo] = {}
        self.device_allocation: Dict[str, List[int]] = {}  # component_name -> device_ids
        self.current_allocation_index = 0
        
        self.initialize_devices()
    
    def initialize_devices(self):
        """사용 가능한 GPU 디바이스 초기화"""
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. Using CPU mode.")
            return
        
        device_count = torch.cuda.device_count()
        logging.info(f"Found {device_count} CUDA devices")
        
        for device_id in range(device_count):
            try:
                device_properties = torch.cuda.get_device_properties(device_id)
                memory_total = device_properties.total_memory / (1024**3)  # GB
                
                # 메모리 사용량 확인
                torch.cuda.set_device(device_id)
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                memory_available = memory_total - memory_allocated
                
                # GPU 활용률 확인 (간단한 방법)
                utilization = (memory_allocated / memory_total) * 100
                
                gpu_info = GPUInfo(
                    device_id=device_id,
                    device_name=device_properties.name,
                    memory_total=memory_total,
                    memory_available=memory_available,
                    utilization=utilization,
                    is_available=True
                )
                
                self.device_info[device_id] = gpu_info
                self.available_devices.append(device_id)
                
                logging.info(f"GPU {device_id}: {gpu_info.device_name} "
                           f"({memory_total:.1f}GB total, {memory_available:.1f}GB available)")
                
            except Exception as e:
                logging.warning(f"Failed to initialize GPU {device_id}: {str(e)}")
    
    def parse_device_config(self, device_config: Union[str, int, List]) -> List[int]:
        """디바이스 설정 파싱
        
        Args:
            device_config: 디바이스 설정
                - "0,1,2" (문자열)
                - [0, 1, 2] (리스트)
                - 0 (단일 정수)
                - "cuda:0,cuda:1" (CUDA 형식)
        
        Returns:
            디바이스 ID 리스트
        """
        device_ids = []
        
        if isinstance(device_config, int):
            # 단일 정수
            device_ids = [device_config]
        elif isinstance(device_config, list):
            # 리스트
            device_ids = [int(d) for d in device_config]
        elif isinstance(device_config, str):
            if device_config.lower() == 'cpu':
                return []  # CPU 모드
            
            # 문자열 파싱
            if ',' in device_config:
                # "0,1,2" 또는 "cuda:0,cuda:1,cuda:2" 형식
                parts = device_config.split(',')
                for part in parts:
                    part = part.strip()
                    if part.startswith('cuda:'):
                        device_id = int(part.split(':')[1])
                    else:
                        device_id = int(part)
                    device_ids.append(device_id)
            else:
                # 단일 문자열 "0" 또는 "cuda:0"
                if device_config.startswith('cuda:'):
                    device_id = int(device_config.split(':')[1])
                else:
                    device_id = int(device_config)
                device_ids = [device_id]
        
        # 유효성 검사
        valid_device_ids = []
        for device_id in device_ids:
            if device_id in self.available_devices:
                valid_device_ids.append(device_id)
            else:
                logging.warning(f"Device {device_id} is not available. Skipping.")
        
        return valid_device_ids
    
    def allocate_devices(self, component_name: str, device_config: Union[str, int, List], 
                        strategy: str = 'round_robin') -> List[int]:
        """컴포넌트에 GPU 디바이스 할당
        
        Args:
            component_name: 컴포넌트 이름 (pose_estimator, tracker 등)
            device_config: 디바이스 설정
            strategy: 할당 전략 ('round_robin', 'memory_based', 'first_available')
        
        Returns:
            할당된 디바이스 ID 리스트
        """
        requested_devices = self.parse_device_config(device_config)
        
        if not requested_devices:
            logging.info(f"Using CPU for {component_name}")
            return []
        
        # 전략에 따른 디바이스 선택
        if strategy == 'round_robin':
            allocated_devices = self._allocate_round_robin(requested_devices)
        elif strategy == 'memory_based':
            allocated_devices = self._allocate_memory_based(requested_devices)
        elif strategy == 'first_available':
            allocated_devices = self._allocate_first_available(requested_devices)
        else:
            allocated_devices = requested_devices  # 기본값: 요청된 디바이스 그대로
        
        self.device_allocation[component_name] = allocated_devices
        
        logging.info(f"Allocated devices {allocated_devices} to {component_name}")
        return allocated_devices
    
    def _allocate_round_robin(self, requested_devices: List[int]) -> List[int]:
        """라운드 로빈 방식 할당"""
        if not requested_devices:
            return []
        
        # 현재 할당 인덱스 기준으로 순환 선택
        allocated = []
        for i in range(len(requested_devices)):
            device_idx = (self.current_allocation_index + i) % len(self.available_devices)
            device_id = self.available_devices[device_idx]
            if device_id in requested_devices:
                allocated.append(device_id)
        
        self.current_allocation_index = (self.current_allocation_index + len(allocated)) % len(self.available_devices)
        return allocated or requested_devices
    
    def _allocate_memory_based(self, requested_devices: List[int]) -> List[int]:
        """메모리 사용량 기준 할당"""
        if not requested_devices:
            return []
        
        # 메모리 사용량이 적은 순서로 정렬
        available_devices = [(device_id, self.device_info[device_id].memory_available) 
                           for device_id in requested_devices if device_id in self.device_info]
        available_devices.sort(key=lambda x: x[1], reverse=True)  # 사용 가능한 메모리 많은 순
        
        return [device_id for device_id, _ in available_devices]
    
    def _allocate_first_available(self, requested_devices: List[int]) -> List[int]:
        """첫 번째 사용 가능한 디바이스 할당"""
        return requested_devices
    
    def get_device_string(self, component_name: str, index: int = 0) -> str:
        """컴포넌트의 주 디바이스 문자열 반환
        
        Args:
            component_name: 컴포넌트 이름
            index: 디바이스 인덱스 (멀티 GPU 사용시)
        
        Returns:
            디바이스 문자열 (예: 'cuda:0', 'cpu')
        """
        if component_name not in self.device_allocation:
            return 'cpu'
        
        allocated_devices = self.device_allocation[component_name]
        if not allocated_devices:
            return 'cpu'
        
        if index >= len(allocated_devices):
            index = 0  # 인덱스가 범위를 벗어나면 첫 번째 디바이스 사용
        
        return f'cuda:{allocated_devices[index]}'
    
    def get_all_devices(self, component_name: str) -> List[str]:
        """컴포넌트에 할당된 모든 디바이스 문자열 리스트 반환
        
        Args:
            component_name: 컴포넌트 이름
        
        Returns:
            디바이스 문자열 리스트
        """
        if component_name not in self.device_allocation:
            return ['cpu']
        
        allocated_devices = self.device_allocation[component_name]
        if not allocated_devices:
            return ['cpu']
        
        return [f'cuda:{device_id}' for device_id in allocated_devices]
    
    def is_multi_gpu(self, component_name: str) -> bool:
        """컴포넌트가 멀티 GPU를 사용하는지 확인
        
        Args:
            component_name: 컴포넌트 이름
        
        Returns:
            멀티 GPU 사용 여부
        """
        if component_name not in self.device_allocation:
            return False
        
        return len(self.device_allocation[component_name]) > 1
    
    def get_gpu_count(self, component_name: str) -> int:
        """컴포넌트에 할당된 GPU 수 반환
        
        Args:
            component_name: 컴포넌트 이름
        
        Returns:
            할당된 GPU 수
        """
        if component_name not in self.device_allocation:
            return 0
        
        return len(self.device_allocation[component_name])
    
    def update_device_status(self):
        """디바이스 상태 업데이트"""
        for device_id in self.available_devices:
            try:
                torch.cuda.set_device(device_id)
                properties = torch.cuda.get_device_properties(device_id)
                memory_total = properties.total_memory / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                memory_available = memory_total - memory_allocated
                utilization = (memory_allocated / memory_total) * 100
                
                if device_id in self.device_info:
                    self.device_info[device_id].memory_available = memory_available
                    self.device_info[device_id].utilization = utilization
                    
            except Exception as e:
                logging.warning(f"Failed to update status for GPU {device_id}: {str(e)}")
                if device_id in self.device_info:
                    self.device_info[device_id].is_available = False
    
    def get_device_info_summary(self) -> Dict:
        """디바이스 정보 요약 반환"""
        summary = {
            'total_devices': len(self.available_devices),
            'available_devices': self.available_devices.copy(),
            'device_allocation': self.device_allocation.copy(),
            'device_details': {}
        }
        
        for device_id, info in self.device_info.items():
            summary['device_details'][device_id] = {
                'name': info.device_name,
                'memory_total': f"{info.memory_total:.1f}GB",
                'memory_available': f"{info.memory_available:.1f}GB",
                'utilization': f"{info.utilization:.1f}%",
                'is_available': info.is_available
            }
        
        return summary
    
    def optimize_allocation(self) -> Dict[str, List[int]]:
        """현재 할당을 최적화하여 새로운 할당 제안
        
        Returns:
            최적화된 할당 딕셔너리
        """
        self.update_device_status()
        
        optimized_allocation = {}
        
        # 메모리 사용량 기준으로 재할당
        for component_name, current_devices in self.device_allocation.items():
            if current_devices:  # GPU 사용 중인 경우만
                optimized_devices = self._allocate_memory_based(current_devices)
                optimized_allocation[component_name] = optimized_devices
            else:
                optimized_allocation[component_name] = current_devices
        
        return optimized_allocation
    
    def clear_allocation(self, component_name: str):
        """컴포넌트의 디바이스 할당 해제
        
        Args:
            component_name: 컴포넌트 이름
        """
        if component_name in self.device_allocation:
            logging.info(f"Clearing device allocation for {component_name}")
            del self.device_allocation[component_name]
    
    def clear_all_allocations(self):
        """모든 할당 해제"""
        logging.info("Clearing all device allocations")
        self.device_allocation.clear()
        self.current_allocation_index = 0


# 전역 GPU 매니저 인스턴스
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """전역 GPU 매니저 인스턴스 반환"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def setup_multi_gpu(device_config: Union[str, int, List], component_name: str, 
                   strategy: str = 'round_robin') -> List[str]:
    """멀티 GPU 설정 편의 함수
    
    Args:
        device_config: 디바이스 설정
        component_name: 컴포넌트 이름
        strategy: 할당 전략
    
    Returns:
        할당된 디바이스 문자열 리스트
    """
    gpu_manager = get_gpu_manager()
    device_ids = gpu_manager.allocate_devices(component_name, device_config, strategy)
    return [f'cuda:{device_id}' for device_id in device_ids] if device_ids else ['cpu']


def get_primary_device(component_name: str) -> str:
    """컴포넌트의 주 디바이스 반환
    
    Args:
        component_name: 컴포넌트 이름
    
    Returns:
        주 디바이스 문자열
    """
    gpu_manager = get_gpu_manager()
    return gpu_manager.get_device_string(component_name)


def is_multi_gpu_enabled(component_name: str) -> bool:
    """멀티 GPU 사용 여부 확인
    
    Args:
        component_name: 컴포넌트 이름
    
    Returns:
        멀티 GPU 사용 여부
    """
    gpu_manager = get_gpu_manager()
    return gpu_manager.is_multi_gpu(component_name)