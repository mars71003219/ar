#!/usr/bin/env python3
"""
Enhanced ByteTracker with RTMO - Main Demo Script
RTMO 포즈 추정과 향상된 ByteTracker를 활용한 메인 데모
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 현재 tracker 디렉토리를 패스에 추가
tracker_root = Path(__file__).parent
sys.path.insert(0, str(tracker_root))

from demo.rtmo_tracking_pipeline import RTMOTrackingPipeline
from demo.video_processor import VideoProcessor
from demo.visualization import TrackingVisualizer


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Enhanced ByteTracker with RTMO Demo')
    
    # 필수 인자
    parser.add_argument('--rtmo-config', 
                       default='/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py',
                       help='RTMO config file path')
    parser.add_argument('--rtmo-checkpoint',
                       default='/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth', 
                       help='RTMO checkpoint file path')
    
    # 입력/출력
    parser.add_argument('--input', '-i',
                       help='Input video file path')
    parser.add_argument('--output-dir', '-o',
                       default='./output',
                       help='Output directory')
    
    # 디바이스 및 설정
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--show', action='store_true',
                       help='Show video during processing')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save output video')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process')
    
    # 시각화 옵션
    parser.add_argument('--no-pose', action='store_true',
                       help='Disable pose skeleton visualization')
    parser.add_argument('--no-bbox', action='store_true',
                       help='Disable bounding box visualization')
    parser.add_argument('--no-track-id', action='store_true',
                       help='Disable track ID visualization')
    
    # 트래커 설정
    parser.add_argument('--config-mode', 
                       choices=['default', 'fast', 'accurate', 'balanced'],
                       default='balanced',
                       help='Tracker configuration mode')
    
    return parser.parse_args()


def get_tracker_config(mode: str):
    """트래커 설정 모드에 따른 설정 반환"""
    if mode == 'default':
        from configs.default_config import DefaultTrackerConfig
        return DefaultTrackerConfig.get_config_dict()
    elif mode == 'fast':
        from configs.rtmo_tracker_config import RTMO_FAST_CONFIG
        return RTMO_FAST_CONFIG
    elif mode == 'accurate':
        from configs.rtmo_tracker_config import RTMO_ACCURATE_CONFIG
        return RTMO_ACCURATE_CONFIG
    else:  # balanced
        from configs.rtmo_tracker_config import RTMOTrackerConfig
        return RTMOTrackerConfig.get_config_dict()


def test_with_default_videos(args, pipeline, processor):
    """기본 테스트 비디오들로 테스트"""
    test_videos = [
        'cam04_06.mp4',
        'F_4_0_0_0_0.mp4'
    ]
    
    # 비디오 검색 경로들
    search_paths = [
        '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos/Fight',
        '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos',
        '/workspace/rtmo_gcn_pipeline/rtmo_pose_track',
        '/workspace',
        '.'
    ]
    
    results = []
    
    for video_name in test_videos:
        video_path = None
        
        # 비디오 파일 찾기
        for search_path in search_paths:
            potential_path = Path(search_path) / video_name
            if potential_path.exists():
                video_path = potential_path
                break
        
        if video_path is None:
            print(f"Warning: Could not find {video_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"Path: {video_path}")
        print(f"{'='*60}")
        
        # 출력 경로 설정
        output_path = Path(args.output_dir) / f"tracked_{video_name}"
        
        try:
            # 파이프라인 리셋
            pipeline.reset()
            
            # 비디오 처리
            result = processor.process_video(
                input_path=str(video_path),
                output_path=str(output_path),
                show_video=args.show,
                save_video=args.save,
                max_frames=args.max_frames
            )
            
            results.append({
                'video_name': video_name,
                'result': result
            })
            
            print(f"\nCompleted: {video_name}")
            print(f"Output: {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue
    
    return results


def print_summary(results):
    """결과 요약 출력"""
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    total_frames = 0
    total_time = 0
    
    for result_info in results:
        video_name = result_info['video_name']
        result = result_info['result']
        
        print(f"\n📹 {video_name}")
        print(f"  Frames processed: {result['processed_frames']}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Average FPS: {result['average_fps']:.1f}")
        
        if 'frame_times' in result:
            ft = result['frame_times']
            print(f"  Frame time - avg: {ft['mean']:.3f}s, "
                  f"min: {ft['min']:.3f}s, max: {ft['max']:.3f}s")
        
        if 'pipeline_stats' in result:
            stats = result['pipeline_stats']
            print(f"  Pipeline stats:")
            print(f"    Total tracks created: {stats.get('total_tracks', 'N/A')}")
            print(f"    Average tracks per frame: {stats.get('average_tracks_per_frame', 'N/A'):.1f}")
        
        total_frames += result['processed_frames']
        total_time += result['total_time']
    
    if results:
        overall_fps = total_frames / total_time if total_time > 0 else 0
        print(f"\n📊 Overall Performance")
        print(f"  Total frames: {total_frames}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall FPS: {overall_fps:.1f}")


def main():
    """메인 함수"""
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Enhanced ByteTracker with RTMO Demo")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Device: {args.device}")
    print(f"Tracker config mode: {args.config_mode}")
    
    # 모델 파일 확인
    rtmo_config = Path(args.rtmo_config)
    rtmo_checkpoint = Path(args.rtmo_checkpoint)
    
    if not rtmo_config.exists():
        print(f"Error: RTMO config file not found: {rtmo_config}")
        return
    
    if not rtmo_checkpoint.exists():
        print(f"Error: RTMO checkpoint file not found: {rtmo_checkpoint}")
        return
    
    try:
        # 트래커 설정
        tracker_config = get_tracker_config(args.config_mode)
        
        # 파이프라인 초기화
        print("\nInitializing RTMO tracking pipeline...")
        pipeline = RTMOTrackingPipeline(
            rtmo_config=str(rtmo_config),
            rtmo_checkpoint=str(rtmo_checkpoint),
            device=args.device,
            tracker_config=tracker_config
        )
        
        # 시각화기 초기화
        visualizer = TrackingVisualizer(
            show_pose=not args.no_pose,
            show_bbox=not args.no_bbox,
            show_track_id=not args.no_track_id
        )
        
        # 비디오 프로세서 초기화
        processor = VideoProcessor(pipeline, visualizer)
        
        print("Pipeline initialized successfully!")
        
        if args.input:
            # 단일 비디오 처리
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input video not found: {input_path}")
                return
            
            output_path = output_dir / f"tracked_{input_path.name}"
            
            print(f"\nProcessing single video: {input_path}")
            result = processor.process_video(
                input_path=str(input_path),
                output_path=str(output_path),
                show_video=args.show,
                save_video=args.save,
                max_frames=args.max_frames
            )
            
            print_summary([{'video_name': input_path.name, 'result': result}])
        else:
            # 기본 테스트 비디오들로 테스트
            print(f"\nTesting with default videos...")
            results = test_with_default_videos(args, pipeline, processor)
            
            if results:
                print_summary(results)
            else:
                print("No videos were successfully processed.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()