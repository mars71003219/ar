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
                       default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000',
                       help='Input video directory')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Output directory')
    
    # 포즈 추출 관련 설정  
    parser.add_argument('--detector-config', type=str,
                       default='/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py',
                       help='RTMO detector config file')
    parser.add_argument('--detector-checkpoint', type=str,
                       default='/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth',
                       help='RTMO detector checkpoint (PTH file)')
    
    # GPU 설정
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU to use: 0 (cuda:0), 1 (cuda:1), 0,1 (multi-GPU), cpu')  
    
    # 처리 파라미터
    parser.add_argument('--clip-len', type=int, default=100,
                       help='Segment clip length (frames)')
    parser.add_argument('--num-person', type=int, default=4,
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
    
    # GPU 설정 검증 및 조정 (출력 전에 먼저 수행)
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
        print("Warning: PyTorch not available. Using CPU")
    
    # 기본값 초기화
    device = 'cpu'
    gpu_ids = []
    multi_gpu = False
    
    if args.gpu.lower() == 'cpu':
        print("Using CPU")
    elif torch_available and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        
        # 사용 가능한 GPU 목록 표시
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # GPU ID 파싱
        try:
            requested_gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
            valid_gpu_ids = [gpu_id for gpu_id in requested_gpu_ids if gpu_id < gpu_count]
            
            if not valid_gpu_ids:
                print("Warning: No valid GPU IDs specified. Using GPU 0")
                valid_gpu_ids = [0] if gpu_count > 0 else []
            
            # 설정 결정
            if valid_gpu_ids:
                if len(valid_gpu_ids) > 1:
                    device = f'cuda:{valid_gpu_ids[0]}'  # 메인 GPU
                    gpu_ids = valid_gpu_ids
                    multi_gpu = True
                    print(f"Multi-GPU mode: GPUs {valid_gpu_ids}")
                else:
                    device = f'cuda:{valid_gpu_ids[0]}'
                    gpu_ids = valid_gpu_ids
                    multi_gpu = False
                    print(f"Single GPU mode: GPU {valid_gpu_ids[0]}")
        except ValueError:
            print(f"Warning: Invalid GPU specification '{args.gpu}'. Using CPU")
    else:
        if torch_available:
            print("Warning: CUDA not available. Using CPU")
        else:
            print("Using CPU (PyTorch not available)")
    
    # 이제 정보 출력
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
    print(f"Final Device: {device}")
    print(f"Final GPU IDs: {gpu_ids}")
    print(f"Final Multi-GPU: {multi_gpu}")
    print()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 통합 포즈 처리기 초기화
    processor = UnifiedPoseProcessor(
        detector_config=args.detector_config,
        detector_checkpoint=args.detector_checkpoint,
        device=device,
        gpu_ids=gpu_ids,
        multi_gpu=multi_gpu,
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
    
    # Step 1: 비디오들을 윈도우 기반으로 처리
    print(" Step 1: Processing videos with window-based approach")
    print("-" * 50)
    
    video_results_list = processor.process_batch_videos(
        video_files,
        args.output_dir,
        args.input_dir,
        training_stride=args.training_stride,
        inference_stride=args.inference_stride,
        max_workers=args.max_workers
    )
    
    if not video_results_list:
        print(" No videos were successfully processed")
        return
    
    print(f" Generated window data for {len(video_results_list)} videos")
    print()
    
    # Step 2: 윈도우 데이터들을 통합 STGCN 데이터로 변환 및 새로운 구조로 저장
    print(" Step 2: Creating unified STGCN data with new directory structure")
    print("-" * 50)
    
    train_count, val_count, test_count = processor.create_unified_stgcn_data(
        video_results_list,
        args.output_dir,
        args.input_dir,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    print(f" Unified STGCN data created:")
    print(f"   Training samples: {train_count:,}")
    print(f"   Validation samples: {val_count:,}")
    print(f"   Test samples: {test_count:,}")
    print(f"   Total windows: {train_count + val_count + test_count:,}")
    print()
    
    # 결과 안내
    dataset_name = os.path.basename(args.input_dir.rstrip('/\\'))
    final_output_dir = os.path.join(args.output_dir, dataset_name)
    
    print(" Generated Directory Structure:")
    print("-" * 50)
    print(f"{final_output_dir}/")
    print("├── train/")
    print("│   ├── Fight/")
    print("│   │   ├── video1/")
    print("│   │   │   ├── video1_windows.pkl")
    print("│   │   │   ├── video1_0.mp4")
    print("│   │   │   ├── video1_1.mp4")
    print("│   │   │   └── ...")
    print("│   │   └── video2/...")
    print("│   └── Normal/")
    print("│       └── ...")
    print("├── val/")
    print("│   ├── Fight/")
    print("│   └── Normal/")
    print("├── test/")
    print("│   ├── Fight/")
    print("│   └── Normal/")
    print("├── {}_train_windows.pkl".format(dataset_name))
    print("├── {}_val_windows.pkl".format(dataset_name))
    print("└── {}_test_windows.pkl".format(dataset_name))
    print()
    
    print(" File Descriptions:")
    print("-" * 50)
    print("• Individual video folders contain:")
    print("  - *_windows.pkl: Video metadata and all window annotations")
    print("  - *_0.mp4, *_1.mp4, ...: 100-frame segment videos with overlays")
    print("• Unified PKL files contain all STGCN training samples")
    print("• Each window represents a 100-frame segment with pose tracking")
    print()
    
    print(" Next Steps:")
    print("-" * 50)
    print("1. Update STGCN config file:")
    print(f"   ann_file_train = '{final_output_dir}/{dataset_name}_train_windows.pkl'")
    print(f"   ann_file_val = '{final_output_dir}/{dataset_name}_val_windows.pkl'")
    print(f"   ann_file_test = '{final_output_dir}/{dataset_name}_test_windows.pkl'")
    print()
    print("2. Start STGCN training:")
    print("   python3 /workspace/mmaction2/tools/train.py stgcnpp_windows_config.py")
    print()
    print("3. Monitor training progress:")
    print("   tensorboard --logdir work_dirs/stgcnpp_windows_fight_detection")
    
    print("\n Window-based pipeline completed successfully!")

if __name__ == "__main__":
    main()