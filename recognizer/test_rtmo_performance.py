#!/usr/bin/env python3
"""
RTMO 성능 비교 테스트: PyTorch (.pth) vs ONNX GPU

recognizer 프레임워크에서 PyTorch와 ONNX GPU 추론 성능을 비교합니다.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

# recognizer 모듈 경로 추가
sys.path.insert(0, '/workspace/recognizer')

try:
    from utils.factory import ModuleFactory
    from utils.data_structure import PoseEstimationConfig
    from utils.config_loader import load_config
    
    # RTMO 모듈들 등록을 위한 임포트
    import pose_estimation.rtmo.register
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the recognizer directory")
    sys.exit(1)


def create_test_image(width: int = 640, height: int = 640) -> np.ndarray:
    """테스트용 더미 이미지 생성"""
    # 실제 카메라 이미지와 유사한 노이즈가 있는 랜덤 이미지 생성
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # 약간의 구조를 추가 (사람 형태와 유사한 패턴)
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(image, (center_x, center_y), (50, 100), 0, 0, 360, (120, 150, 180), -1)
    cv2.ellipse(image, (center_x, center_y - 80), (20, 30), 0, 0, 360, (200, 180, 160), -1)
    
    return image


def test_pytorch_inference(config_path: str, num_iterations: int = 20) -> Dict[str, Any]:
    """PyTorch (.pth) 추론 성능 테스트"""
    print("=" * 60)
    print("PyTorch (.pth) 추론 성능 테스트")
    print("=" * 60)
    
    # 설정 로드 및 PyTorch 모드로 변경
    config = load_config(config_path)
    config['models']['pose_estimation']['inference_mode'] = 'pth'
    
    # PyTorch 추정기 생성
    pose_config = PoseEstimationConfig(
        model_name=config['models']['pose_estimation']['pth']['model_name'],
        config_file=config['models']['pose_estimation']['pth']['config_file'],
        model_path=config['models']['pose_estimation']['pth']['checkpoint_path'],
        device=config['models']['pose_estimation']['pth']['device'],
        score_threshold=config['models']['pose_estimation']['pth']['score_threshold'],
        nms_threshold=config['models']['pose_estimation']['pth']['nms_threshold'],
        keypoint_threshold=config['models']['pose_estimation']['pth']['keypoint_threshold'],
        input_size=config['models']['pose_estimation']['pth']['input_size']
    )
    
    # 추론 설정 딕셔너리 생성
    inference_config = {
        'inference_mode': 'pth',
        'pth': {
            'model_name': pose_config.model_name,
            'config_file': pose_config.config_file,
            'checkpoint_path': pose_config.model_path,
            'device': pose_config.device,
            'score_threshold': pose_config.score_threshold,
            'nms_threshold': pose_config.nms_threshold,
            'keypoint_threshold': pose_config.keypoint_threshold,
            'input_size': pose_config.input_size
        }
    }
    
    estimator = ModuleFactory.create_pose_estimator_from_inference_config(inference_config)
    
    # 테스트 이미지 생성
    test_image = create_test_image()
    
    # Warmup
    print("Warming up PyTorch model...")
    for _ in range(5):
        estimator.extract_poses(test_image)
    
    # 성능 측정
    print(f"Running {num_iterations} iterations...")
    times = []
    total_persons = 0
    
    for i in range(num_iterations):
        start_time = time.time()
        persons = estimator.extract_poses(test_image, frame_idx=i)
        end_time = time.time()
        
        times.append(end_time - start_time)
        total_persons += len(persons)
        
        if (i + 1) % 5 == 0:
            avg_time = np.mean(times[-5:])
            print(f"Iteration {i + 1:2d}: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
    
    # 통계 계산
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_fps = 1.0 / avg_time
    avg_persons = total_persons / num_iterations
    
    results = {
        'framework': 'PyTorch (.pth)',
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_fps': avg_fps,
        'avg_persons_per_frame': avg_persons,
        'model_info': estimator.get_model_info(),
        'raw_times': times
    }
    
    print(f"\nPyTorch 결과:")
    print(f"평균 처리시간: {avg_time:.3f}±{std_time:.3f}s")
    print(f"평균 FPS: {avg_fps:.2f}")
    print(f"최소/최대 시간: {min_time:.3f}s / {max_time:.3f}s")
    print(f"평균 검출 인원: {avg_persons:.1f}명")
    
    # 정리
    estimator.cleanup()
    
    return results


def test_onnx_inference(config_path: str, num_iterations: int = 20) -> Dict[str, Any]:
    """ONNX GPU 추론 성능 테스트"""
    print("\n" + "=" * 60)
    print("ONNX GPU 추론 성능 테스트")
    print("=" * 60)
    
    # 설정 로드 및 ONNX 모드로 변경
    config = load_config(config_path)
    config['models']['pose_estimation']['inference_mode'] = 'onnx'
    
    # ONNX 추정기 생성
    pose_config = PoseEstimationConfig(
        model_name=config['models']['pose_estimation']['onnx']['model_name'],
        model_path=config['models']['pose_estimation']['onnx']['model_path'],
        device=config['models']['pose_estimation']['onnx']['device'],
        score_threshold=config['models']['pose_estimation']['onnx']['score_threshold'],
        nms_threshold=config['models']['pose_estimation']['onnx']['nms_threshold'],
        keypoint_threshold=config['models']['pose_estimation']['onnx']['keypoint_threshold'],
        max_detections=config['models']['pose_estimation']['onnx']['max_detections'],
        model_input_size=tuple(config['models']['pose_estimation']['onnx']['model_input_size']),
        mean=config['models']['pose_estimation']['onnx']['mean'],
        std=config['models']['pose_estimation']['onnx']['std'],
        backend=config['models']['pose_estimation']['onnx']['backend'],
        to_openpose=config['models']['pose_estimation']['onnx']['to_openpose']
    )
    
    # 추론 설정 딕셔너리 생성
    inference_config = {
        'inference_mode': 'onnx',
        'onnx': {
            'model_name': pose_config.model_name,
            'model_path': pose_config.model_path,
            'device': pose_config.device,
            'score_threshold': pose_config.score_threshold,
            'nms_threshold': pose_config.nms_threshold,
            'keypoint_threshold': pose_config.keypoint_threshold,
            'max_detections': pose_config.max_detections,
            'model_input_size': list(pose_config.model_input_size),
            'mean': pose_config.mean,
            'std': pose_config.std,
            'backend': pose_config.backend,
            'to_openpose': pose_config.to_openpose
        }
    }
    
    estimator = ModuleFactory.create_pose_estimator_from_inference_config(inference_config)
    
    # 테스트 이미지 생성
    test_image = create_test_image()
    
    # Warmup
    print("Warming up ONNX model...")
    for _ in range(5):
        estimator.extract_poses(test_image)
    
    # 성능 측정
    print(f"Running {num_iterations} iterations...")
    times = []
    total_persons = 0
    
    for i in range(num_iterations):
        start_time = time.time()
        persons = estimator.extract_poses(test_image, frame_idx=i)
        end_time = time.time()
        
        times.append(end_time - start_time)
        total_persons += len(persons)
        
        if (i + 1) % 5 == 0:
            avg_time = np.mean(times[-5:])
            print(f"Iteration {i + 1:2d}: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
    
    # 통계 계산
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_fps = 1.0 / avg_time
    avg_persons = total_persons / num_iterations
    
    results = {
        'framework': 'ONNX GPU',
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_fps': avg_fps,
        'avg_persons_per_frame': avg_persons,
        'model_info': estimator.get_model_info(),
        'raw_times': times
    }
    
    print(f"\nONNX 결과:")
    print(f"평균 처리시간: {avg_time:.3f}±{std_time:.3f}s")
    print(f"평균 FPS: {avg_fps:.2f}")
    print(f"최소/최대 시간: {min_time:.3f}s / {max_time:.3f}s")
    print(f"평균 검출 인원: {avg_persons:.1f}명")
    
    # 정리
    estimator.cleanup()
    
    return results


def compare_results(pytorch_results: Dict[str, Any], onnx_results: Dict[str, Any]):
    """결과 비교 및 요약"""
    print("\n" + "=" * 80)
    print("성능 비교 요약")
    print("=" * 80)
    
    pt_fps = pytorch_results['avg_fps']
    onnx_fps = onnx_results['avg_fps']
    
    pt_time = pytorch_results['avg_time']
    onnx_time = onnx_results['avg_time']
    
    speedup = pt_time / onnx_time
    fps_improvement = (onnx_fps / pt_fps - 1) * 100
    
    print(f"{'Framework':<15} {'FPS':<10} {'처리시간(s)':<12} {'표준편차':<10}")
    print("-" * 50)
    print(f"{'PyTorch':<15} {pt_fps:<10.2f} {pt_time:<12.3f} {pytorch_results['std_time']:<10.3f}")
    print(f"{'ONNX GPU':<15} {onnx_fps:<10.2f} {onnx_time:<12.3f} {onnx_results['std_time']:<10.3f}")
    
    print(f"\n성능 개선:")
    print(f"속도 향상: {speedup:.2f}x")
    print(f"FPS 개선: {fps_improvement:+.1f}%")
    
    if speedup > 1.1:
        print("✅ ONNX GPU가 PyTorch보다 빠릅니다!")
    elif speedup > 0.9:
        print("⚖️ 성능이 비슷합니다.")
    else:
        print("⚠️ PyTorch가 더 빠릅니다.")
    
    # 모델 정보 비교
    print(f"\n모델 정보:")
    pt_info = pytorch_results['model_info']
    onnx_info = onnx_results['model_info']
    
    print(f"PyTorch Device: {pt_info.get('device', 'Unknown')}")
    print(f"ONNX Device: {onnx_info.get('device', 'Unknown')}")
    print(f"ONNX Providers: {onnx_info.get('providers', 'Unknown')}")


def main():
    """메인 실행 함수"""
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 설정 파일 경로
    config_path = '/workspace/recognizer/configs/config.yaml'
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print("RTMO 성능 비교 테스트 시작")
    print("PyTorch (.pth) vs ONNX GPU 추론")
    
    # 테스트 반복 횟수
    num_iterations = 30
    
    try:
        # PyTorch 테스트
        pytorch_results = test_pytorch_inference(config_path, num_iterations)
        
        # ONNX 테스트
        onnx_results = test_onnx_inference(config_path, num_iterations)
        
        # 결과 비교
        compare_results(pytorch_results, onnx_results)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()