#!/usr/bin/env python3
"""
Enhanced ByteTracker with RTMO - Simple Demo
간단한 단일 파일 데모
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path

# 필요한 경로들을 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, '/workspace')

try:
    from mmpose.apis import init_model, inference_bottomup
    import torch
    print("MMPose imported successfully!")
except ImportError as e:
    print(f"MMPose import error: {e}")
    sys.exit(1)

# RTMO 모델과 체크포인트 경로
RTMO_CONFIG = '/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py'
RTMO_CHECKPOINT = '/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'

# 테스트 비디오 경로들
TEST_VIDEOS = [
    '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/cam04_06.mp4',
    '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight/F_4_0_0_0_0.mp4'
]


def init_rtmo_model():
    """RTMO 모델 초기화"""
    print("Initializing RTMO model...")
    model = init_model(RTMO_CONFIG, RTMO_CHECKPOINT, device='cuda:0')
    print("RTMO model initialized successfully!")
    return model


def extract_detections_from_pose_result(pose_result):
    """포즈 결과에서 detection 추출"""
    if not pose_result or not hasattr(pose_result, 'pred_instances'):
        return np.empty((0, 5))
    
    pred_instances = pose_result.pred_instances
    
    if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
        return np.empty((0, 5))
    
    # 바운딩 박스 추출
    if hasattr(pred_instances.bboxes, 'cpu'):
        bboxes = pred_instances.bboxes.cpu().numpy()
    else:
        bboxes = np.array(pred_instances.bboxes)
    
    # 점수 추출
    if hasattr(pred_instances, 'bbox_scores'):
        if hasattr(pred_instances.bbox_scores, 'cpu'):
            scores = pred_instances.bbox_scores.cpu().numpy()
        else:
            scores = np.array(pred_instances.bbox_scores)
    else:
        scores = np.ones(len(bboxes))
    
    # [x1, y1, x2, y2, score] 형태로 결합
    detections = np.column_stack([bboxes, scores])
    return detections


def draw_detections(image, detections):
    """간단한 detection 시각화"""
    vis_image = image.copy()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2, score = det
        
        # 바운딩 박스
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        
        # ID와 점수 표시
        text = f"ID:{i} {score:.2f}"
        cv2.putText(vis_image, text, (int(x1), int(y1-5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image


def process_video(video_path, output_path, max_frames=100):
    """비디오 처리"""
    print(f"\nProcessing: {video_path}")
    
    # RTMO 모델 초기화
    model = init_rtmo_model()
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # 비디오 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while frame_count < min(total_frames, max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # RTMO로 포즈 추정
            pose_results = inference_bottomup(model, frame)
            
            if pose_results:
                # detection 추출
                detections = extract_detections_from_pose_result(pose_results[0])
                
                # 시각화
                vis_frame = draw_detections(frame, detections)
                
                # 정보 패널 추가
                info_text = f"Frame: {frame_count}, Detections: {len(detections)}"
                cv2.putText(vis_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                vis_frame = frame.copy()
                cv2.putText(vis_frame, f"Frame: {frame_count}, No detections", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 비디오에 저장
            out.write(vis_frame)
            
            frame_count += 1
            
            # 진행상황 출력
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                print(f"  Processed {frame_count}/{min(total_frames, max_frames)} frames, avg FPS: {avg_fps:.1f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_count / total_time
        
        print(f"  Completed! Processed {frame_count} frames in {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Output saved to: {output_path}")
        
    except Exception as e:
        print(f"  Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cap.release()
        out.release()


def main():
    """메인 함수"""
    print("Enhanced ByteTracker with RTMO - Simple Demo")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    output_dir = Path('./simple_output')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 각 테스트 비디오 처리
    for video_path in TEST_VIDEOS:
        if Path(video_path).exists():
            video_name = Path(video_path).name
            output_path = str(output_dir / f"simple_{video_name}")
            
            try:
                process_video(video_path, output_path, max_frames=200)
            except Exception as e:
                print(f"Failed to process {video_name}: {e}")
                continue
        else:
            print(f"Warning: Video not found: {video_path}")
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()