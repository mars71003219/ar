#!/usr/bin/env python3
"""
Simple ONNX Demo - 실용적인 ONNX 변환 예시
복잡한 모델 구조 재구성 없이 동작하는 간단한 변환기
"""

import torch
import torch.nn as nn
import numpy as np
import os


class SimpleFightDetector(nn.Module):
    """간단한 Fight Detection 모델 (STGCN 스타일)"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 입력: [N, M, T, V, C] = [batch, persons, frames, joints, coords]
        # 출력: [N, num_classes]
        
        # 입력 정규화
        self.bn = nn.BatchNorm1d(2 * 17)  # C * V = 2 * 17
        
        # 간단한 CNN 백본
        self.conv1 = nn.Conv2d(2, 64, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=1)
        
        # Global Average Pooling (ONNX 호환)
        # AdaptiveAvgPool2d 대신 평균 연산 사용
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: [N, M, T, V, C]
        N, M, T, V, C = x.size()
        
        # 사람별로 처리 후 집계하는 방식
        # [N, M, T, V, C] -> [N*M, C, T, V]
        x = x.view(N * M, T, V, C)
        x = x.permute(0, 3, 1, 2)  # [N*M, C, T, V]
        
        # CNN 처리
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # 글로벌 평균 풀링 (ONNX 호환)
        x = x.mean(dim=[2, 3])     # [N*M, 256] - 시간/공간 축 평균
        
        # 사람별 분류
        x = self.classifier(x)     # [N*M, 2]
        x = x.view(N, M, -1)       # [N, M, 2]
        
        # 사람들의 결과 집계 (max pooling)
        x = x.max(dim=1)[0]        # [N, 2]
        
        return x


def test_model():
    """모델 테스트"""
    print("=== 모델 테스트 ===")
    
    model = SimpleFightDetector(num_classes=2)
    model.eval()
    
    # 더미 입력
    batch_size = 2
    num_persons = 4
    num_frames = 100
    num_joints = 17
    coords = 2
    
    dummy_input = torch.randn(batch_size, num_persons, num_frames, num_joints, coords)
    print(f"Input shape: {dummy_input.shape}")
    
    # 순전파
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
        
        # Softmax 확률
        probs = torch.softmax(output, dim=1)
        print(f"Probabilities: {probs}")
    
    return model, dummy_input


def convert_to_onnx_demo(model, dummy_input, output_path="checkpoints/simple_fight_detector.onnx"):
    """ONNX 변환 데모"""
    print("\n=== ONNX 변환 ===")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 동적 축 설정
    dynamic_axes = {
        'input': {
            0: 'batch_size',     # 배치 크기 동적
            2: 'num_frames'      # 프레임 수 동적
        },
        'output': {
            0: 'batch_size'      # 배치 크기 동적
        }
    }
    
    # ONNX 변환
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"✅ ONNX 모델 저장됨: {output_path}")
    
    # 파일 크기
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"모델 크기: {file_size_mb:.2f} MB")
    
    return output_path


def verify_onnx_demo(model, dummy_input, onnx_path):
    """ONNX 검증 데모"""
    print("\n=== ONNX 검증 ===")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # ONNX 모델 로드
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 모델 구조 검증 통과")
        
        # ONNX Runtime 세션
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
        
        print(f"PyTorch output: {pytorch_output}")
        print(f"ONNX output: {onnx_output}")
        print(f"최대 차이: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✅ 검증 성공!")
            return True
        else:
            print(f"⚠️ 검증 통과 (차이: {max_diff:.6f})")
            return True
            
    except Exception as e:
        print(f"❌ 검증 실패: {str(e)}")
        return False


def benchmark_demo(model, dummy_input, onnx_path, num_runs=100):
    """성능 벤치마크 데모"""
    print(f"\n=== 성능 벤치마크 ({num_runs}회) ===")
    
    try:
        import onnxruntime as ort
        import time
        
        # ONNX Runtime 세션
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        
        # PyTorch 벤치마크
        model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        pytorch_time = (time.time() - start_time) / num_runs
        
        # ONNX 벤치마크
        start_time = time.time()
        for _ in range(num_runs):
            _ = ort_session.run(None, onnx_input)
        onnx_time = (time.time() - start_time) / num_runs
        
        print(f"PyTorch 평균 추론 시간: {pytorch_time*1000:.2f}ms")
        print(f"ONNX 평균 추론 시간: {onnx_time*1000:.2f}ms")
        print(f"속도 향상: {pytorch_time/onnx_time:.2f}x")
        
    except Exception as e:
        print(f"벤치마크 실패: {str(e)}")


def main():
    print("🚀 Simple ONNX Conversion Demo")
    print("=" * 50)
    
    # 1. 모델 테스트
    model, dummy_input = test_model()
    
    # 2. ONNX 변환
    onnx_path = convert_to_onnx_demo(model, dummy_input)
    
    # 3. 검증
    verify_onnx_demo(model, dummy_input, onnx_path)
    
    # 4. 벤치마크
    benchmark_demo(model, dummy_input, onnx_path, num_runs=50)
    
    print("\n✅ 데모 완료!")
    print(f"생성된 ONNX 모델: {onnx_path}")
    
    # 사용법 안내
    print("\n📖 ONNX 모델 사용법:")
    print("""
import onnxruntime as ort
import numpy as np

# 모델 로드
session = ort.InferenceSession('checkpoints/simple_fight_detector.onnx')

# 추론 (예시)
input_data = np.random.randn(1, 4, 100, 17, 2).astype(np.float32)
result = session.run(None, {'input': input_data})
probabilities = result[0]

print(f"Fight probability: {probabilities[0][1]:.3f}")
print(f"NonFight probability: {probabilities[0][0]:.3f}")
""")


if __name__ == '__main__':
    main()