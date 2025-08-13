#!/usr/bin/env python3
"""
Model Manager - 포즈 모델 초기화 및 관리
"""

import os
import torch


class ModelManager:
    """포즈 모델 초기화 및 관리 클래스"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.pose_model = None
        self._setup_gpu_environment()
    
    def _setup_gpu_environment(self):
        """GPU 환경 설정"""
        try:
            if torch.cuda.is_available() and 'cuda' in self.device:
                gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
                if gpu_id < torch.cuda.device_count():
                    torch.cuda.set_device(gpu_id)
                else:
                    self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        except Exception as e:
            self.device = 'cpu'
    
    def initialize_pose_model(self, config_file, checkpoint_file):
        """포즈 모델 초기화"""
        if self.pose_model is None:
            try:
                from mmpose.apis import init_model
                print(f"Initializing pose model: {config_file}")
                self.pose_model = init_model(config_file, checkpoint_file, device=self.device)
                print("Pose model initialized successfully")
            except Exception as e:
                print(f"Failed to initialize pose model: {str(e)}")
                raise e
        return self.pose_model
    
    def get_pose_model(self):
        """포즈 모델 반환"""
        return self.pose_model
    
    def clear_gpu_cache(self):
        """GPU 메모리 캐시 정리"""
        try:
            if torch.cuda.is_available() and 'cuda' in self.device:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            pass
    
    def cleanup(self):
        """모델 정리"""
        if self.pose_model is not None:
            try:
                if hasattr(self.pose_model, 'cpu'):
                    self.pose_model.cpu()
                del self.pose_model
                self.pose_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Pose model resources cleaned up")
            except Exception as e:
                print(f"Warning: Failed to cleanup pose model: {str(e)}")