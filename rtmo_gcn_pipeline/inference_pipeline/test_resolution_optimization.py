#!/usr/bin/env python3
"""
Resolution Optimization Test
해상도 최적화 테스트 - 즉시 적용 가능한 성능 개선
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.append('/workspace/rtmo_gcn_pipeline/inference_pipeline')

from pose_estimator import RTMOPoseEstimator
from config import POSE_CONFIG, POSE_CHECKPOINT

def test_resolution_impact():
    """해상도별 성능 테스트"""
    print("🎯 해상도별 RTMO 성능 테스트")
    
    # 테스트 비디오
    video_path = "/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime_test/Fight/Fighting033_x264.mp4"
    max_test_frames = 50  # 빠른 테스트를 위해 50프레임만
    
    # RTMO 추정기 초기화
    estimator = RTMOPoseEstimator(POSE_CONFIG, POSE_CHECKPOINT, 'cuda:0')
    
    # 해상도 테스트 설정
    resolution_tests = [
        {'scale': 1.0, 'name': '원본 (100%)', 'target_size': None},
        {'scale': 0.8, 'name': '고품질 (80%)', 'target_size': (512, 384)},
        {'scale': 0.6, 'name': '균형 (60%)', 'target_size': (384, 288)},
        {'scale': 0.4, 'name': '고속 (40%)', 'target_size': (256, 192)},
    ]
    
    results = []
    
    for test_config in resolution_tests:
        print(f"\n📊 테스트: {test_config['name']}")
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 비디오 열기 실패: {video_path}")
            continue
        
        frames = []
        for i in range(max_test_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 해상도 조정
            if test_config['target_size']:
                frame = cv2.resize(frame, test_config['target_size'])
            
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            print("❌ 프레임 읽기 실패")
            continue
        
        print(f"   프레임 수: {len(frames)}")
        print(f"   해상도: {frames[0].shape[1]}x{frames[0].shape[0]}")
        
        # 성능 측정
        start_time = time.time()
        
        pose_results = []
        for frame in frames:
            try:
                # RTMO 추론
                from mmpose.apis import inference_bottomup
                result = inference_bottomup(estimator.model, frame)
                
                if result and len(result.pred_instances) > 0:
                    keypoints = result.pred_instances.keypoints.cpu().numpy()
                    scores = result.pred_instances.keypoint_scores.cpu().numpy()
                    pose_results.append((keypoints, scores))
                else:
                    pose_results.append((np.empty((0, 17, 2)), np.empty((0, 17))))
                    
            except Exception as e:
                print(f"   ⚠️  프레임 처리 실패: {e}")
                pose_results.append((np.empty((0, 17, 2)), np.empty((0, 17))))
        
        processing_time = time.time() - start_time
        fps = len(frames) / processing_time if processing_time > 0 else 0
        
        # 결과 저장
        test_result = {
            'name': test_config['name'],
            'scale': test_config['scale'],
            'frames': len(frames),
            'time': processing_time,
            'fps': fps,
            'valid_poses': sum(1 for kpts, scores in pose_results if len(kpts) > 0)
        }
        results.append(test_result)
        
        print(f"   처리 시간: {processing_time:.2f}초")
        print(f"   FPS: {fps:.1f}")
        print(f"   유효 포즈: {test_result['valid_poses']}/{len(frames)}")
    
    # 결과 비교
    print(f"\n🚀 해상도별 성능 비교 결과")
    print("-" * 60)
    print(f"{'해상도':<12} {'FPS':<8} {'처리시간':<10} {'성능향상':<10} {'정확도':<8}")
    print("-" * 60)
    
    baseline_fps = results[0]['fps'] if results else 1
    
    for result in results:
        speedup = result['fps'] / baseline_fps if baseline_fps > 0 else 0
        accuracy = (result['valid_poses'] / result['frames']) * 100 if result['frames'] > 0 else 0
        
        print(f"{result['name']:<12} {result['fps']:<8.1f} {result['time']:<10.2f} "
              f"{speedup:<10.1f}x {accuracy:<8.1f}%")
    
    # 권장사항
    print(f"\n💡 권장사항:")
    
    if len(results) >= 2:
        best_balance = max(results[1:], key=lambda x: x['fps'] * (x['valid_poses']/x['frames']))
        print(f"   최적 설정: {best_balance['name']}")
        print(f"   예상 성능 향상: {(best_balance['fps']/baseline_fps):.1f}배")
        print(f"   실제 파이프라인 적용 시:")
        print(f"   현재 10.3 FPS → 예상 {10.3 * (best_balance['fps']/baseline_fps):.1f} FPS")
    
    return results

if __name__ == "__main__":
    test_resolution_impact()