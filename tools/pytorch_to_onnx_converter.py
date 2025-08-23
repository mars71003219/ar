#!/usr/bin/env python3
"""
PyTorch to ONNX Converter
STGCN++ 모델을 ONNX 형식으로 변환하는 도구

Features:
- 동적/정적 입력 크기 지원
- 다양한 최적화 옵션
- 검증 및 벤치마크 기능
- 상세한 로깅 및 오류 처리
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from mmaction.apis import init_recognizer
from mmengine.logging import MMLogger
from mmengine import Config


class STGCNONNXConverter:
    """STGCN 모델을 ONNX로 변환하는 클래스"""
    
    def __init__(self, 
                 config_path: str,
                 checkpoint_path: str,
                 output_path: str,
                 device: str = 'cuda:0'):
        """
        Args:
            config_path: MMAction2 설정 파일 경로
            checkpoint_path: PyTorch 체크포인트 파일 경로
            output_path: 출력 ONNX 파일 경로
            device: 추론 디바이스
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.device = device
        
        # 로깅 설정
        self.logger = MMLogger.get_instance('pytorch_to_onnx', log_level='INFO')
        
        # 모델 초기화
        self.model = None
        self.config = None
        
    def load_model(self) -> nn.Module:
        """PyTorch 모델 로드"""
        try:
            self.logger.info(f"Loading model from {self.checkpoint_path}")
            self.logger.info(f"Using config: {self.config_path}")
            
            # MMAction2 모델 초기화
            self.model = init_recognizer(
                config=self.config_path,
                checkpoint=self.checkpoint_path,
                device=self.device
            )
            
            # 평가 모드로 설정
            self.model.eval()
            
            # 설정 로드
            self.config = Config.fromfile(self.config_path)
            
            self.logger.info("Model loaded successfully")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def create_dummy_input(self, 
                          batch_size: int = 1,
                          num_frames: int = 100,
                          num_persons: int = 4,
                          num_keypoints: int = 17,
                          keypoint_dim: int = 2) -> torch.Tensor:
        """더미 입력 데이터 생성
        
        Args:
            batch_size: 배치 크기
            num_frames: 프레임 수
            num_persons: 사람 수
            num_keypoints: 키포인트 수
            keypoint_dim: 키포인트 차원 (2D/3D)
            
        Returns:
            torch.Tensor: 더미 입력 텐서 [N, M, T, V, C]
        """
        # STGCN 입력 형식: [N, M, T, V, C]
        # N: batch_size, M: num_persons, T: num_frames, V: num_keypoints, C: keypoint_dim
        dummy_input = torch.randn(
            batch_size, num_persons, num_frames, num_keypoints, keypoint_dim,
            device=self.device, dtype=torch.float32
        )
        
        self.logger.info(f"Created dummy input shape: {dummy_input.shape}")
        return dummy_input
    
    def convert_to_onnx(self,
                       dummy_input: torch.Tensor,
                       dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                       opset_version: int = 11,
                       optimize: bool = True,
                       verbose: bool = False) -> None:
        """ONNX 변환 수행
        
        Args:
            dummy_input: 더미 입력 데이터
            dynamic_axes: 동적 축 설정
            opset_version: ONNX opset 버전
            optimize: 최적화 여부
            verbose: 상세 로깅 여부
        """
        try:
            self.logger.info("Starting ONNX conversion...")
            self.logger.info(f"Input shape: {dummy_input.shape}")
            self.logger.info(f"Output path: {self.output_path}")
            self.logger.info(f"Dynamic axes: {dynamic_axes}")
            self.logger.info(f"Opset version: {opset_version}")
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            # ONNX 변환
            with torch.no_grad():
                torch.onnx.export(
                    model=self.model,
                    args=dummy_input,
                    f=self.output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=optimize,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    verbose=verbose
                )
            
            self.logger.info(f"ONNX model saved to: {self.output_path}")
            
        except Exception as e:
            self.logger.error(f"ONNX conversion failed: {str(e)}")
            raise
    
    def verify_onnx_model(self, dummy_input: torch.Tensor, rtol: float = 1e-3) -> bool:
        """ONNX 모델 검증
        
        Args:
            dummy_input: 검증용 입력 데이터
            rtol: 상대 허용 오차
            
        Returns:
            bool: 검증 성공 여부
        """
        try:
            self.logger.info("Verifying ONNX model...")
            
            # ONNX 모델 로드 및 검증
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            self.logger.info("ONNX model structure verification passed")
            
            # ONNX Runtime 세션 생성
            ort_session = ort.InferenceSession(
                self.output_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # PyTorch 모델 추론
            with torch.no_grad():
                pytorch_output = self.model(dummy_input)
                if isinstance(pytorch_output, (list, tuple)):
                    pytorch_output = pytorch_output[0]
                pytorch_result = pytorch_output.cpu().numpy()
            
            # ONNX 모델 추론
            onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_result = ort_session.run(None, onnx_input)[0]
            
            # 결과 비교
            if np.allclose(pytorch_result, onnx_result, rtol=rtol):
                self.logger.info(f"✅ Verification passed! Max difference: {np.max(np.abs(pytorch_result - onnx_result)):.6f}")
                return True
            else:
                self.logger.error(f"❌ Verification failed! Max difference: {np.max(np.abs(pytorch_result - onnx_result)):.6f}")
                return False
                
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False
    
    def benchmark_models(self, 
                        dummy_input: torch.Tensor,
                        num_runs: int = 100,
                        warmup_runs: int = 10) -> Dict[str, float]:
        """모델 성능 벤치마크
        
        Args:
            dummy_input: 벤치마크용 입력 데이터
            num_runs: 측정 횟수
            warmup_runs: 워밍업 횟수
            
        Returns:
            Dict[str, float]: 성능 측정 결과
        """
        try:
            self.logger.info(f"Benchmarking models (warmup: {warmup_runs}, runs: {num_runs})")
            
            # ONNX Runtime 세션
            ort_session = ort.InferenceSession(
                self.output_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            
            # PyTorch 벤치마크
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            pytorch_start = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
                    
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            pytorch_time = (time.time() - pytorch_start) / num_runs
            
            # ONNX 벤치마크  
            # Warmup
            for _ in range(warmup_runs):
                _ = ort_session.run(None, onnx_input)
            
            onnx_start = time.time()
            for _ in range(num_runs):
                _ = ort_session.run(None, onnx_input)
            onnx_time = (time.time() - onnx_start) / num_runs
            
            results = {
                'pytorch_time_ms': pytorch_time * 1000,
                'onnx_time_ms': onnx_time * 1000,
                'speedup': pytorch_time / onnx_time if onnx_time > 0 else 0
            }
            
            self.logger.info(f"PyTorch inference time: {results['pytorch_time_ms']:.2f}ms")
            self.logger.info(f"ONNX inference time: {results['onnx_time_ms']:.2f}ms")
            self.logger.info(f"Speedup: {results['speedup']:.2f}x")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {str(e)}")
            return {}


def create_dynamic_axes_config(enable_dynamic: bool,
                              dynamic_batch: bool = True,
                              dynamic_frames: bool = True,
                              dynamic_persons: bool = False) -> Optional[Dict[str, Dict[int, str]]]:
    """동적 축 설정 생성
    
    Args:
        enable_dynamic: 동적 축 활성화 여부
        dynamic_batch: 배치 크기 동적 여부
        dynamic_frames: 프레임 수 동적 여부  
        dynamic_persons: 사람 수 동적 여부
        
    Returns:
        Optional[Dict]: 동적 축 설정
    """
    if not enable_dynamic:
        return None
    
    dynamic_axes = {
        'input': {},
        'output': {}
    }
    
    # STGCN 입력 형태: [N, M, T, V, C]
    if dynamic_batch:
        dynamic_axes['input'][0] = 'batch_size'
        dynamic_axes['output'][0] = 'batch_size'
    
    if dynamic_persons:
        dynamic_axes['input'][1] = 'num_persons'
    
    if dynamic_frames:
        dynamic_axes['input'][2] = 'num_frames'
    
    return dynamic_axes if dynamic_axes['input'] else None


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='PyTorch to ONNX Converter for STGCN')
    
    # 필수 인자
    parser.add_argument('--config', type=str, required=True,
                       help='MMAction2 config file path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='PyTorch checkpoint file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output ONNX file path')
    
    # 입력 크기 설정
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--num-frames', type=int, default=100,
                       help='Number of frames (default: 100)')
    parser.add_argument('--num-persons', type=int, default=4,
                       help='Number of persons (default: 4)')
    parser.add_argument('--num-keypoints', type=int, default=17,
                       help='Number of keypoints (default: 17)')
    parser.add_argument('--keypoint-dim', type=int, default=2,
                       help='Keypoint dimension (default: 2)')
    
    # 동적 축 설정
    parser.add_argument('--dynamic', action='store_true',
                       help='Enable dynamic input shapes')
    parser.add_argument('--dynamic-batch', action='store_true', default=True,
                       help='Enable dynamic batch size (default: True)')
    parser.add_argument('--dynamic-frames', action='store_true', default=True,
                       help='Enable dynamic frame count (default: True)')
    parser.add_argument('--dynamic-persons', action='store_true',
                       help='Enable dynamic person count')
    
    # ONNX 설정
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable ONNX optimization')
    
    # 검증 및 벤치마크
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify ONNX model (default: True)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--benchmark-runs', type=int, default=100,
                       help='Number of benchmark runs (default: 100)')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference (default: cuda:0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 파일 존재 확인
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
        # 변환기 초기화
        converter = STGCNONNXConverter(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device
        )
        
        # 모델 로드
        converter.load_model()
        
        # 더미 입력 생성
        dummy_input = converter.create_dummy_input(
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            num_persons=args.num_persons,
            num_keypoints=args.num_keypoints,
            keypoint_dim=args.keypoint_dim
        )
        
        # 동적 축 설정
        dynamic_axes = create_dynamic_axes_config(
            enable_dynamic=args.dynamic,
            dynamic_batch=args.dynamic_batch,
            dynamic_frames=args.dynamic_frames,
            dynamic_persons=args.dynamic_persons
        )
        
        # ONNX 변환
        converter.convert_to_onnx(
            dummy_input=dummy_input,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            optimize=not args.no_optimize,
            verbose=args.verbose
        )
        
        # 검증
        if args.verify:
            verification_passed = converter.verify_onnx_model(dummy_input)
            if not verification_passed:
                logging.warning("ONNX model verification failed!")
        
        # 벤치마크
        if args.benchmark:
            benchmark_results = converter.benchmark_models(
                dummy_input=dummy_input,
                num_runs=args.benchmark_runs
            )
        
        logging.info("✅ Conversion completed successfully!")
        logging.info(f"ONNX model saved to: {args.output}")
        
        # 모델 정보 출력
        onnx_model = onnx.load(args.output)
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logging.info(f"Model size: {file_size_mb:.2f} MB")
        logging.info(f"ONNX opset version: {onnx_model.opset_import[0].version}")
        
    except Exception as e:
        logging.error(f"❌ Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()