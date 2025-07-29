#!/usr/bin/env python3
"""
폭력 감지 데이터 처리 메인 프로그램 (1B 방식)
- 원본 비디오 → 개별 세그먼트 PKL → 통합 STGCN PKL
- 단일 패스 처리로 중복 제거, 효율적 워크플로우
"""

import os
import argparse
import glob
from unified_pose_processor import UnifiedPoseProcessor

def main():
    parser = argparse.ArgumentParser(description='Violence Detection Unified Processing Pipeline')
    
    # 공통 설정
    parser.add_argument('--input-dir', type=str, 
                    #    default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000',
                       default='/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime_test',
                       help='Input video directory')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/window_sliding_output',
                       help='Output directory')
    
    # 포즈 추출 관련 설정  
    parser.add_argument('--detector-config', type=str,
                       default='/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py',
                       help='RTMO detector config file')
    parser.add_argument('--detector-checkpoint', type=str,
                       default='/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth',
                       help='RTMO detector checkpoint (PTH file)')
    
    # 처리 파라미터
    parser.add_argument('--clip-len', type=int, default=100,
                       help='Segment clip length (frames)')
    parser.add_argument('--num-person', type=int, default=2,
                       help='Maximum persons to display in overlay (모든 인물은 저장됨)')
    parser.add_argument('--training-stride', type=int, default=10,
                       help='Stride for dense training segments')
    parser.add_argument('--inference-stride', type=int, default=50,
                       help='Stride for sparse inference segments')
    parser.add_argument('--max-workers', type=int, default=24,
                       help='Maximum parallel workers')
    
    # 오버레이 설정
    parser.add_argument('--save-overlay', action='store_true', default=True,
                       help='Save pose overlay videos')
    parser.add_argument('--overlay-fps', type=int, default=30,
                       help='Overlay video FPS')
    
    # 데이터 분할
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" Violence Detection Unified Processing Pipeline")
    print("=" * 70)
    print(f"Input videos: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Clip length: {args.clip_len} frames")
    print(f"Training stride: {args.training_stride} (dense)")
    print(f"Inference stride: {args.inference_stride} (sparse)")
    print(f"Overlay persons: {args.num_person} (all persons saved)")
    print(f"Save overlay: {args.save_overlay}")
    print(f"Model: {args.detector_checkpoint}")
    print()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 통합 포즈 처리기 초기화
    processor = UnifiedPoseProcessor(
        detector_config=args.detector_config,
        detector_checkpoint=args.detector_checkpoint,
        clip_len=args.clip_len,
        num_person=args.num_person,
        save_overlay=args.save_overlay,
        overlay_fps=args.overlay_fps
    )
    
    # 비디오 파일 수집
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))
    
    if not video_files:
        print(f" No videos found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} videos")
    print()
    
    # Step 1: 비디오들을 개별 세그먼트 PKL로 변환
    print(" Step 1: Converting videos to segment PKLs")
    print("-" * 50)
    
    segments_data_list = processor.process_batch_videos(
        video_files,
        args.output_dir,
        training_stride=args.training_stride,
        inference_stride=args.inference_stride,
        max_workers=args.max_workers
    )
    
    if not segments_data_list:
        print(" No videos were successfully processed")
        return
    
    print(f" Generated {len(segments_data_list)} segment PKL files")
    print(f"   Individual PKL files: {args.output_dir}/*_segments.pkl")
    print()
    
    # Step 2: 개별 세그먼트 PKL들을 통합 STGCN 데이터로 변환
    print(" Step 2: Creating unified STGCN training data")
    print("-" * 50)
    
    train_segments, inference_segments = processor.create_unified_stgcn_data(
        segments_data_list,
        args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    print(f" Unified STGCN data created:")
    print(f"   Training segments: {train_segments:,} (dense)")
    print(f"   Inference segments: {inference_segments:,} (sparse)")
    print(f"   Training boost: {train_segments/inference_segments:.1f}x more data")
    print()
    
    # 결과 안내
    print(" Generated Files:")
    print("-" * 50)
    print("Individual segment PKLs:")
    print(f"   {args.output_dir}/*_segments.pkl")
    print()
    print("Unified STGCN data:")
    print(f"   {args.output_dir}/dense_training/rwf2000_enhanced_sliding_train.pkl")
    print(f"   {args.output_dir}/dense_training/rwf2000_enhanced_sliding_val.pkl")
    print(f"   {args.output_dir}/sparse_inference/rwf2000_enhanced_sliding_test.pkl")
    print()
    
    print(" Next Steps:")
    print("-" * 50)
    print("1. Update STGCN config file:")
    print(f"   ann_file_train = '{args.output_dir}/dense_training/rwf2000_enhanced_sliding_train.pkl'")
    print(f"   ann_file_val = '{args.output_dir}/dense_training/rwf2000_enhanced_sliding_val.pkl'")
    print(f"   ann_file_test = '{args.output_dir}/sparse_inference/rwf2000_enhanced_sliding_test.pkl'")
    print()
    print("2. Start STGCN training with data augmentation:")
    print("   python3 /workspace/mmaction2/tools/train.py stgcnpp_enhanced_augmented_config.py")
    print()
    print("3. Monitor training progress:")
    print("   tensorboard --logdir work_dirs/stgcnpp_enhanced_augmented_fight_detection")
    
    print("\n Unified pipeline completed successfully!")

if __name__ == "__main__":
    main()