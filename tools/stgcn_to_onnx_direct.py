#!/usr/bin/env python3
"""
STGCN 모델을 직접 구성해서 ONNX로 변환하는 스크립트
MMAction2 호환성 문제를 우회
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        
        self.A = nn.Parameter(A.clone())
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * self.num_subsets,
            kernel_size=1
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if residual and in_channels == out_channels:
            self.residual = lambda x: x
        elif residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = lambda x: 0
    
    def forward(self, x):
        # x: [N, C, T, V]
        N, C, T, V = x.size()
        
        # Graph convolution
        x = self.conv(x)  # [N, C*num_subsets, T, V]
        x = x.view(N, self.num_subsets, self.out_channels, T, V)
        
        # Apply adjacency matrix
        x = torch.einsum('nkctv,kvw->nctw', x, self.A)  # [N, C, T, V]
        
        # Residual connection
        res = self.residual(x) if hasattr(self, 'residual') else 0
        x = self.bn(x) + res
        x = self.relu(x)
        
        return x


class TemporalConvolution(nn.Module):
    """Temporal Convolution Layer"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, residual=True):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if residual and in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        elif residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = lambda x: 0
    
    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)
        x = self.bn(x) + res
        x = self.relu(x)
        return x


class STGCNBlock(nn.Module):
    """ST-GCN Block"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, A, residual=False)
        self.tcn = TemporalConvolution(
            out_channels, out_channels, 
            stride=stride, residual=residual
        )
    
    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        return x


class STGCN(nn.Module):
    """ST-GCN Network for Fight Detection"""
    def __init__(self, num_classes=2, in_channels=2, base_channels=48, num_stages=6):
        super().__init__()
        
        # COCO skeleton adjacency matrix (17 keypoints)
        self.A = self.get_coco_adjacency()
        
        # Data normalization
        self.data_bn = nn.BatchNorm1d(in_channels * 17)  # 2 * 17 = 34
        
        # ST-GCN backbone
        self.backbone = nn.ModuleList()
        
        # First layer
        self.backbone.append(STGCNBlock(
            in_channels, base_channels, self.A, residual=False
        ))
        
        # Hidden layers
        channels = base_channels
        for i in range(1, num_stages):
            # Increase channels at certain stages
            if i in [3, 5]:  # inflate_stages
                out_channels = channels * 2
                stride = 2  # down_stages
            else:
                out_channels = channels
                stride = 1
                
            self.backbone.append(STGCNBlock(
                channels, out_channels, self.A, stride=stride
            ))
            channels = out_channels
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc = nn.Linear(channels, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def get_coco_adjacency(self):
        """COCO skeleton adjacency matrix"""
        # COCO 17 keypoints connections
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        num_joints = 17
        A = torch.zeros(3, num_joints, num_joints)
        
        # Self connections
        A[0] = torch.eye(num_joints)
        
        # Inward connections (child -> parent)
        for i, j in edges:
            A[1, j, i] = 1
        
        # Outward connections (parent -> child)  
        for i, j in edges:
            A[2, i, j] = 1
        
        # Normalize
        for i in range(3):
            A[i] = A[i] / (A[i].sum(dim=1, keepdim=True) + 1e-6)
        
        return A
    
    def forward(self, x):
        # Input: [N, M, T, V, C] -> [N, C, T, V*M]
        N, M, T, V, C = x.size()
        
        # Reshape and normalize
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [N, M, V, C, T]
        x = x.view(N, M * V * C, T)  # [N, M*V*C, T]
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [N, M, C, T, V]
        x = x.view(N * M, C, T, V)  # [N*M, C, T, V]
        
        # ST-GCN backbone
        for layer in self.backbone:
            x = layer(x)
        
        # Global pooling
        x = self.pool(x)  # [N*M, C, 1, 1]
        x = x.view(N, M, -1)  # [N, M, C]
        
        # Person-level aggregation (max pooling)
        x = x.max(dim=1)[0]  # [N, C]
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)  # [N, num_classes]
        
        return x


def load_stgcn_weights(model, checkpoint_path):
    """체크포인트에서 가중치 로드"""
    print(f"Loading weights from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # 가중치 매핑
    model_dict = model.state_dict()
    matched_dict = {}
    
    print("Mapping weights...")
    for name, param in model_dict.items():
        if name in state_dict:
            if param.shape == state_dict[name].shape:
                matched_dict[name] = state_dict[name]
                print(f"✓ {name}: {param.shape}")
            else:
                print(f"✗ {name}: shape mismatch {param.shape} vs {state_dict[name].shape}")
        else:
            print(f"? {name}: not found in checkpoint")
    
    # 로드
    model.load_state_dict(matched_dict, strict=False)
    print(f"Loaded {len(matched_dict)}/{len(model_dict)} parameters")
    
    return model


def convert_to_onnx(model, dummy_input, output_path, dynamic_axes=None, opset_version=11):
    """ONNX 변환"""
    print(f"\n=== ONNX 변환 시작 ===")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output path: {output_path}")
    print(f"Dynamic axes: {dynamic_axes}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # 테스트 추론
        output = model(dummy_input)
        print(f"Test output shape: {output.shape}")
        
        # ONNX 변환
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
    
    # 파일 크기
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {file_size_mb:.2f} MB")


def verify_onnx(model, dummy_input, onnx_path):
    """ONNX 모델 검증"""
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"\n=== ONNX 검증 시작 ===")
        
        # ONNX 모델 로드
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 모델 구조 검증 통과")
        
        # ONNX Runtime
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # PyTorch 추론
        model.eval()
        with torch.no_grad():
            pytorch_output = model(dummy_input).cpu().numpy()
        
        # ONNX 추론  
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        
        # 결과 비교
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        if max_diff < 1e-3:
            print(f"✅ 검증 성공! 최대 차이: {max_diff:.6f}")
            return True
        else:
            print(f"❌ 검증 실패! 최대 차이: {max_diff:.6f}")
            return False
            
    except Exception as e:
        print(f"❌ 검증 실패: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='STGCN Direct ONNX Converter')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='PyTorch checkpoint path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output ONNX path')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num-frames', type=int, default=100,
                       help='Number of frames')
    parser.add_argument('--num-persons', type=int, default=4,
                       help='Number of persons')
    parser.add_argument('--dynamic', action='store_true',
                       help='Enable dynamic shapes')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX model')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device')
    
    args = parser.parse_args()
    
    print("=== STGCN Direct ONNX Converter ===")
    
    # 모델 생성
    model = STGCN(num_classes=2, in_channels=2, base_channels=48, num_stages=6)
    
    # 가중치 로드
    model = load_stgcn_weights(model, args.checkpoint)
    
    # 디바이스 이동
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 더미 입력
    dummy_input = torch.randn(
        args.batch_size, args.num_persons, args.num_frames, 17, 2,
        device=device
    )
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # 동적 축 설정
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'num_frames'},
            'output': {0: 'batch_size'}
        }
    
    # ONNX 변환
    convert_to_onnx(model, dummy_input, args.output, dynamic_axes)
    
    # 검증
    if args.verify:
        verify_onnx(model, dummy_input, args.output)
    
    print("✅ 모든 작업 완료!")


if __name__ == '__main__':
    main()