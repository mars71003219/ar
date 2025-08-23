#!/usr/bin/env python3
"""
STGCN ONNX Classifier 테스트 스크립트
"""

import sys
import os
import numpy as np
import logging

# 패키지 경로 추가
sys.path.insert(0, '/home/gaonpf/hsnam/mmlabs/recognizer')

from action_classification.stgcn import STGCNONNXClassifier
from utils.data_structure import ActionClassificationConfig, WindowAnnotation, PersonPose, FramePoses

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_config():
    """테스트용 설정 생성"""
    config = ActionClassificationConfig(
        model_name='stgcn_onnx',
        model_path='/workspace/mmaction2/checkpoints/stgcn_fight_detection.onnx',
        class_names=['NonFight', 'Fight'],
        device='cuda:0',
        window_size=100,
        max_persons=4,
        coordinate_format='xyz'
    )
    return config

def create_dummy_window_data():
    """더미 윈도우 데이터 생성 (MMAction2 표준 형식)"""
    
    # [M, T, V, C] 형태로 더미 데이터 생성
    M, T, V, C = 4, 100, 17, 3  # Max_persons, Time, Vertices, Coordinates
    
    # 키포인트 데이터: [M, T, V, C] - ONNX는 3D, 전처리는 2D 요구
    C = 2  # 2D 좌표로 변경
    keypoint = np.random.randn(M, T, V, C).astype(np.float32)
    keypoint[:, :, :, :2] *= 100  # x, y 좌표
    keypoint[:, :, :, :2] += 320  # 중앙으로 이동
    
    # 키포인트 신뢰도: [M, T, V]
    keypoint_score = np.random.rand(M, T, V).astype(np.float32) * 0.5 + 0.5  # 0.5-1.0
    
    # WindowAnnotation 생성
    window_data = WindowAnnotation(
        window_idx=0,
        start_frame=0,
        end_frame=99,
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        frame_dir='dummy_frames',
        img_shape=(640, 640),
        original_shape=(640, 640),
        total_frames=100,
        label=1,  # Fight
        video_name='test_video'
    )
    
    return window_data

def test_stgcn_onnx_classifier():
    """STGCN ONNX Classifier 테스트"""
    print("=== STGCN ONNX Classifier 테스트 시작 ===")
    
    try:
        # 1. 설정 생성
        print("1. 설정 생성 중...")
        config = create_test_config()
        
        # 2. 분류기 생성
        print("2. STGCN ONNX 분류기 생성 중...")
        classifier = STGCNONNXClassifier(config)
        
        # 3. 모델 초기화
        print("3. 모델 초기화 중...")
        if not classifier.initialize_model():
            print("❌ 모델 초기화 실패!")
            return False
        
        print("✅ 모델 초기화 성공!")
        
        # 4. 모델 정보 확인
        print("\n4. 모델 정보:")
        model_info = classifier.get_classifier_info()
        print(f"  - 분류기 타입: {model_info.get('classifier_type')}")
        print(f"  - 디바이스: {model_info.get('device')}")
        print(f"  - 클래스 개수: {len(model_info.get('class_names', []))}")
        print(f"  - 클래스 이름: {model_info.get('class_names')}")
        
        onnx_info = model_info.get('onnx_model_info', {})
        if onnx_info:
            print(f"  - ONNX 모델 경로: {onnx_info.get('model_path')}")
            print(f"  - 초기화 상태: {onnx_info.get('is_initialized')}")
            if 'input_info' in onnx_info:
                for i, inp in enumerate(onnx_info['input_info']):
                    print(f"  - 입력 {i}: {inp}")
            if 'output_info' in onnx_info:
                for i, out in enumerate(onnx_info['output_info']):
                    print(f"  - 출력 {i}: {out}")
        
        # 5. 더미 데이터 생성
        print("\n5. 더미 데이터 생성 중...")
        window_data = create_dummy_window_data()
        print(f"  - 윈도우 크기: {window_data.total_frames} 프레임")
        print(f"  - 키포인트 형태: {window_data.keypoint.shape}")
        print(f"  - 사람 수: {window_data.keypoint.shape[0]}")
        
        # 6. 추론 테스트
        print("\n6. 추론 테스트 중...")
        result = classifier.classify_window(window_data)
        
        print(f"✅ 추론 완료!")
        print(f"  - 예측 클래스: {result.prediction}")
        print(f"  - 신뢰도: {result.confidence:.4f}")
        print(f"  - 확률: {result.probabilities}")
        print(f"  - 모델명: {result.model_name}")
        
        if hasattr(result, 'metadata') and result.metadata:
            print(f"  - 메타데이터: {result.metadata}")
        
        # 7. 성능 통계
        print("\n7. 성능 통계:")
        stats = classifier.get_classifier_info().get('onnx_stats', {})
        if stats:
            print(f"  - 총 추론 횟수: {stats.get('total_inferences', 0)}")
            print(f"  - 평균 처리 시간: {stats.get('avg_processing_time', 0)*1000:.2f}ms")
            print(f"  - 예상 FPS: {stats.get('fps_estimate', 0):.1f}")
            print(f"  - 에러 횟수: {stats.get('errors', 0)}")
        
        # 8. 다중 윈도우 테스트
        print("\n8. 다중 윈도우 테스트 중...")
        windows = [create_dummy_window_data() for _ in range(3)]
        results = classifier.classify_multiple_windows(windows)
        
        print(f"✅ 다중 윈도우 추론 완료!")
        for i, result in enumerate(results):
            print(f"  - 윈도우 {i}: 클래스={result.prediction}, 신뢰도={result.confidence:.4f}")
        
        # 9. 리소스 정리
        print("\n9. 리소스 정리 중...")
        classifier.cleanup()
        print("✅ 리소스 정리 완료!")
        
        print("\n🎉 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_stgcn_onnx_classifier()
    sys.exit(0 if success else 1)