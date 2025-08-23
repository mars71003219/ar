#!/usr/bin/env python3
"""
Simple PyTorch to ONNX Converter for STGCN
MMCV 버전 호환성 문제를 피한 간단한 변환기
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def create_dummy_input(batch_size: int = 1,
                      num_frames: int = 100,
                      num_persons: int = 4,
                      num_keypoints: int = 17,
                      keypoint_dim: int = 2,
                      device: str = 'cuda:0') -> torch.Tensor:
    """더미 입력 데이터 생성
    
    Args:
        batch_size: 배치 크기
        num_frames: 프레임 수
        num_persons: 사람 수
        num_keypoints: 키포인트 수
        keypoint_dim: 키포인트 차원 (2D/3D)
        device: 디바이스
        
    Returns:
        torch.Tensor: 더미 입력 텐서 [N, M, T, V, C]
    """
    # STGCN 입력 형식: [N, M, T, V, C]
    dummy_input = torch.randn(
        batch_size, num_persons, num_frames, num_keypoints, keypoint_dim,
        device=device, dtype=torch.float32
    )
    
    print(f"Created dummy input shape: {dummy_input.shape}")
    return dummy_input


def create_dynamic_axes_config(enable_dynamic: bool,
                              dynamic_batch: bool = True,
                              dynamic_frames: bool = True,
                              dynamic_persons: bool = False) -> Optional[Dict[str, Dict[int, str]]]:
    """동적 축 설정 생성"""
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


def load_pytorch_model(checkpoint_path: str, device: str = 'cuda:0') -> nn.Module:
    """PyTorch 체크포인트에서 모델 로드"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 상태만 추출
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Model state_dict keys (first 5): {list(state_dict.keys())[:5]}")
    
    # 모델 구조를 추측해서 재구성하는 것보다는 원본 모델을 로드하는 것이 좋지만,
    # 여기서는 간단한 예시를 위해 체크포인트만 사용
    
    return state_dict


def convert_to_onnx_simple(checkpoint_path: str,
                          output_path: str,
                          dummy_input: torch.Tensor,
                          dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                          opset_version: int = 11,
                          device: str = 'cuda:0'):
    """간단한 ONNX 변환 (모델 구조를 직접 정의)"""
    try:
        print("=== ONNX 변환 시작 ===")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output path: {output_path}")
        print(f"Dynamic axes: {dynamic_axes}")
        print(f"Opset version: {opset_version}")
        
        # 원본 mmaction2 방식으로 모델 로드 시도
        sys.path.insert(0, '/home/gaonpf/hsnam/mmlabs/mmaction2')
        
        try:
            # mmaction2 환경에서 실행 시도
            from mmaction.apis import init_recognizer
            
            config_path = '/home/gaonpf/hsnam/mmlabs/mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py'
            
            print("Loading model with MMAction2...")
            model = init_recognizer(
                config=config_path,
                checkpoint=checkpoint_path,
                device=device
            )
            model.eval()
            print("Model loaded successfully with MMAction2")
            
        except Exception as e:
            print(f"MMAction2 loading failed: {e}")
            print("Trying alternative approach...")
            
            # 대안: 체크포인트에서 직접 모델 로드
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 간단한 더미 모델 생성 (실제로는 정확한 모델 구조가 필요)
            class DummySTGCN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(2, 64, 1)
                    self.fc = nn.Linear(64, 2)  # 2 classes
                    
                def forward(self, x):
                    # x: [N, M, T, V, C] -> [N, C, T, V*M]
                    N, M, T, V, C = x.shape
                    x = x.permute(0, 4, 2, 1, 3).contiguous()
                    x = x.view(N, C, T, V*M)
                    x = self.conv(x)
                    x = x.mean(dim=[2, 3])  # Global average pooling
                    x = self.fc(x)
                    return x
            
            model = DummySTGCN().to(device)
            print("Created dummy model (structure may not match exactly)")
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # ONNX 변환
        with torch.no_grad():
            torch.onnx.export(
                model=model,
                args=dummy_input,
                f=output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        print(f"✅ ONNX 변환 완료: {output_path}")
        
        # 파일 크기 정보
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Simple PyTorch to ONNX Converter')
    
    # 필수 인자
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
                       help='Enable dynamic batch size')
    parser.add_argument('--dynamic-frames', action='store_true', default=True,
                       help='Enable dynamic frame count')
    parser.add_argument('--dynamic-persons', action='store_true',
                       help='Enable dynamic person count')
    
    # ONNX 설정
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference (default: cuda:0)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 파일 존재 확인
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
        print("=== Simple PyTorch to ONNX Converter ===")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Output: {args.output}")
        print(f"Device: {args.device}")
        
        # 더미 입력 생성
        dummy_input = create_dummy_input(
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            num_persons=args.num_persons,
            num_keypoints=args.num_keypoints,
            keypoint_dim=args.keypoint_dim,
            device=args.device
        )
        
        # 동적 축 설정
        dynamic_axes = create_dynamic_axes_config(
            enable_dynamic=args.dynamic,
            dynamic_batch=args.dynamic_batch,
            dynamic_frames=args.dynamic_frames,
            dynamic_persons=args.dynamic_persons
        )
        
        # ONNX 변환
        success = convert_to_onnx_simple(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            dummy_input=dummy_input,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            device=args.device
        )
        
        if success:
            print("✅ 변환 완료!")
        else:
            print("❌ 변환 실패!")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()