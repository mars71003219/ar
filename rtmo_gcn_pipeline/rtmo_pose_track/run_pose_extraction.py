#!/usr/bin/env python3
"""
폭력 감지 데이터 처리 메인 프로그램 (1B 방식)
- 원본 비디오 → 개별 세그먼트 PKL → 통합 STGCN PKL
- 단일 패스 처리로 중복 제거, 효율적 워크플로우
"""

import os
import argparse
import glob
import re
from unified_pose_processor import UnifiedPoseProcessor
from error_logger import ProcessingErrorLogger, capture_exception_info

def get_unprocessed_videos(input_dir, output_dir):
    """output 디렉토리를 기준으로 처리되지 않은 비디오 목록을 가져옵니다."""
    all_input_videos = glob.glob(os.path.join(input_dir, '**', '*.*'), recursive=True)
    
    # 부모경로+파일명으로 유니크한 키 생성 (중복 해결)
    input_video_map = {}
    for p in all_input_videos:
        if p.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            basename = os.path.splitext(os.path.basename(p))[0]
            parent_dir = os.path.basename(os.path.dirname(p))
            unique_key = f"{parent_dir}/{basename}"
            input_video_map[unique_key] = p

    processed_video_keys = set()
    dataset_name = os.path.basename(input_dir.rstrip('/\\'))
    temp_path = os.path.join(output_dir, dataset_name, 'temp')

    # temp 폴더 하위에서 비디오 이름으로 만들어진 폴더들을 찾습니다
    if os.path.exists(temp_path):
        for root, dirs, files in os.walk(temp_path):
            for dir_name in dirs:
                # 비디오 이름 폴더인지 확인 (Fight, Normal, train, val 같은 구조 폴더가 아닌)
                if dir_name not in ['Fight', 'Normal', 'train', 'val', 'test']:
                    # root에서 카테고리 추출 (Fight 또는 Normal)
                    root_parts = root.split(os.sep)
                    if 'Fight' in root_parts:
                        temp_category = 'Fight'
                    elif 'Normal' in root_parts or 'NonFight' in root_parts:
                        temp_category = 'Normal'
                    else:
                        # 카테고리를 찾을 수 없는 경우 건너뛰기
                        continue
                    
                    # Input에서 사용하는 카테고리명으로 매핑
                    input_categories = ['Fight', 'NonFight']  # Input에서 사용하는 실제 카테고리명
                    
                    for input_category in input_categories:
                        unique_key = f"{input_category}/{dir_name}"
                        if unique_key in input_video_map:
                            processed_video_keys.add(unique_key)
                            break

    unprocessed_video_keys = set(input_video_map.keys()) - processed_video_keys
    unprocessed_video_paths = [input_video_map[key] for key in unprocessed_video_keys if key in input_video_map]
    
    return sorted(unprocessed_video_paths), len(processed_video_keys)




def process_videos_with_logging(processor, video_files, args, error_logger):
    """에러 로깅을 포함한 비디오 처리"""
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # 비디오 처리 시작
    
    successful_videos_data = []
    failed_count = 0
    
    # 각 비디오 처리 시작 로그
    for video_path in video_files:
        error_logger.log_video_start(video_path)
    
    try:
        # max_workers가 1이면 순차 처리 (모델 재로딩 방지)
        if args.max_workers == 1:
            print("Using single-process mode for optimal model efficiency")
            for video_path in tqdm(video_files, desc="Processing videos"):
                try:
                    result = processor.process_single_video_to_segments(
                        video_path, args.output_dir, args.input_dir, args.training_stride
                    )
                    
                    if isinstance(result, dict) and 'failure_stage' in result:
                        failed_count += 1
                        error_logger.log_video_failure(video_path, result)
                        print(f"Failed: {os.path.basename(video_path)} - {result.get('root_cause', 'Unknown error')}")
                    else:
                        successful_videos_data.append(result)
                        windows_count = len(result.get('windows', [])) if result else 0
                        error_logger.log_video_success(video_path, windows_count)
                        print(f"Success: {os.path.basename(video_path)}")
                        
                except Exception as e:
                    failed_count += 1
                    error_type, error_message, full_traceback = capture_exception_info()
                    error_logger.log_video_failure(video_path, None, error_type, error_message, full_traceback)
                    print(f"Exception: {os.path.basename(video_path)} - {str(e)}")
        
        else:
            # 멀티프로세싱 사용 (각 프로세스마다 모델 로딩됨)
            print(f"Using multi-process mode with {args.max_workers} workers (model will be loaded per process)")
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                # 작업 제출
                future_to_video = {
                    executor.submit(
                        processor.process_single_video_to_segments, 
                        video, args.output_dir, args.input_dir, 
                        args.training_stride
                    ): video 
                    for video in video_files
                }
                
                # 결과 수집
                for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing videos"):
                    video_path = future_to_video[future]
                    
                    try:
                        result = future.result()
                        if result and result.get('windows'):
                            # 성공 로그
                            windows_count = len(result['windows'])
                            error_logger.log_video_success(video_path, windows_count)
                            successful_videos_data.append(result)
                        elif result and isinstance(result, dict) and 'failure_stage' in result:
                            # 상세 실패 분석 결과가 포함된 경우 (새로운 분석 방식)
                            error_logger.log_video_failure(video_path, failure_analysis=result)
                            failed_count += 1
                        elif result and isinstance(result, dict) and 'error_type' in result:
                            # 에러 상세 정보가 포함된 결과 (기존 방식)
                            specific_cause = result.get('specific_cause', 'UNKNOWN_ERROR')
                            diagnosis = result.get('diagnosis', 'No diagnosis available')
                            error_message = f"{diagnosis} ({result.get('error_message', 'No details')})"
                            full_traceback = result.get('traceback', '')
                            
                            error_logger.log_video_failure(video_path, None, specific_cause, error_message, full_traceback)
                            failed_count += 1
                        else:
                            # None이나 빈 결과의 경우 - 구체적인 실패 원인을 찾을 수 없는 상황
                            error_msg = "Video processing returned None or empty result - check video file integrity"
                            error_logger.log_video_failure(video_path, None, "PROCESSING_RETURNED_NONE", error_msg)
                            failed_count += 1
                            
                    except Exception as exc:
                        # 처리 중 예외 발생
                        error_type, error_message, full_traceback = capture_exception_info()
                        error_logger.log_video_failure(video_path, None, error_type, error_message, full_traceback)
                        # 비디오 처리 예외 발생
                        failed_count += 1
    
    except Exception as e:
        error_type, error_message, full_traceback = capture_exception_info()
        error_logger.log_general_error("BATCH_PROCESSING", f"{error_type}: {error_message}", full_traceback)
        raise
    
    # 배치 처리 완료
    
    return successful_videos_data

def main():
    parser = argparse.ArgumentParser(description='Violence Detection Unified Processing Pipeline')
    
    # 모드 설정
    parser.add_argument('--mode', type=str, choices=['full', 'merge'], default='full',
                       help='Processing mode: full (complete pipeline) or merge (PKL merge only)')
    
    # 공통 설정
    parser.add_argument('--input-dir', type=str, 
                       default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2002',
                       help='Input video directory (for full mode) or processed data directory (for merge mode)')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Output directory')
    
    
    
    # Resume 설정
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume processing by comparing input and output directories')

    # 포즈 추출 관련 설정  
    parser.add_argument('--detector-config', type=str,
                       default='/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py',
                       help='RTMO detector config file')
    parser.add_argument('--detector-checkpoint', type=str,
                       default='/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth',
                       help='RTMO detector checkpoint (PTH file)')
    
    # GPU 설정
    parser.add_argument('--gpu', type=str, default='0,1',
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
    parser.add_argument('--max-workers', type=int, default=32,
                       help='Maximum parallel workers')
    
    # 오버레이 설정
    parser.add_argument('--save-overlay', action='store_true', default=True,
                       help='Save pose overlay videos')
    parser.add_argument('--overlay-fps', type=int, default=30,
                       help='Overlay video FPS')
    
    # 포즈 추출 임계값 설정
    parser.add_argument('--score-thr', type=float, default=0.3,
                       help='Pose detection score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.35,
                       help='NMS threshold for pose detection')
    parser.add_argument('--quality-threshold', type=float, default=0.3,
                       help='Minimum quality threshold for tracks')
    parser.add_argument('--min-track-length', type=int, default=10,
                       help='Minimum track length for valid tracks')
    
    # ByteTracker 설정
    parser.add_argument('--track-high-thresh', type=float, default=0.6,
                       help='High threshold for ByteTracker')
    parser.add_argument('--track-low-thresh', type=float, default=0.1,
                       help='Low threshold for ByteTracker')
    parser.add_argument('--track-max-disappeared', type=int, default=30,
                       help='Maximum frames a track can be lost before deletion')
    parser.add_argument('--track-min-hits', type=int, default=3,
                       help='Minimum hits required to consider a track valid')
    
    # 윈도우 처리 설정
    parser.add_argument('--min-success-rate', type=float, default=0.1,
                       help='Minimum success rate required for window processing (0.1 = 10%)')
    
    # 복합 점수 가중치 설정
    parser.add_argument('--movement-weight', type=float, default=0.45,
                       help='Weight for movement score in composite scoring')
    parser.add_argument('--position-weight', type=float, default=0.10,
                       help='Weight for position score in composite scoring')
    parser.add_argument('--interaction-weight', type=float, default=0.30,
                       help='Weight for interaction score in composite scoring')
    parser.add_argument('--temporal-weight', type=float, default=0.10,
                       help='Weight for temporal consistency in composite scoring')
    parser.add_argument('--persistence-weight', type=float, default=0.05,
                       help='Weight for persistence score in composite scoring')
    
    # 데이터 분할
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터셋명 추출 (에러 로깅용)
    dataset_name = os.path.basename(args.input_dir.rstrip('/\\'))
    
    # 에러 로거 초기화 (resume 시 최신 로그에 이어쓰기)
    error_logger = ProcessingErrorLogger(args.output_dir, dataset_name, resume=args.resume)
    
    try:
        # 모드에 따른 분기 처리
        if args.mode == 'merge':
            print("=" * 70)
            print(" PKL Merge Mode - Processing existing data")
            print("=" * 70)
            print(f"Mode: {args.mode}")
            print(f"Processed data directory: {args.input_dir}")
            print(f"Output directory: {args.output_dir}")
            print(f"Train split: {args.train_split}")
            print(f"Val split: {args.val_split}")
            print()
            
            # 간단한 프로세서 초기화 (merge 모드에서는 GPU 설정 불필요)
            processor = UnifiedPoseProcessor(
                detector_config="",  # merge 모드에서는 불필요
                detector_checkpoint="",  # merge 모드에서는 불필요
                device='cpu',
                gpu_ids=[],
                multi_gpu=False,
                clip_len=args.clip_len,
                num_person=args.num_person,
                save_overlay=False,  # merge 모드에서는 오버레이 생성 안함
                overlay_fps=args.overlay_fps,
                # 포즈 추출 파라미터
                score_thr=args.score_thr,
                nms_thr=args.nms_thr,
                quality_threshold=args.quality_threshold,
                min_track_length=args.min_track_length,
                # ByteTracker 파라미터
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                track_max_disappeared=args.track_max_disappeared,
                track_min_hits=args.track_min_hits,
                # 복합 점수 가중치  
                weights=[
                    args.movement_weight,
                    args.position_weight,
                    args.interaction_weight,
                    args.temporal_weight,
                    args.persistence_weight
                ],
                # 윈도우 처리 파라미터
                min_success_rate=args.min_success_rate
            )
            
            try:
                # merge 모드 실행
                train_count, val_count, test_count = processor.merge_existing_pkl_files(
                    args.input_dir,
                    args.output_dir,
                    train_split=args.train_split,
                    val_split=args.val_split
                )
                
                if train_count + val_count + test_count > 0:
                    # 결과 안내
                    final_output_dir = os.path.join(args.output_dir, dataset_name)
                    
                    print(" Final Results:")
                    print("-" * 50)
                    print(f"  Training samples: {train_count:,}")
                    print(f"  Validation samples: {val_count:,}")
                    print(f"  Test samples: {test_count:,}")
                    print(f"  Total windows: {train_count + val_count + test_count:,}")
                    print()
                    print(f" Unified PKL files created:")
                    print(f"   {final_output_dir}/{dataset_name}_train_windows.pkl")
                    print(f"   {final_output_dir}/{dataset_name}_val_windows.pkl")
                    print(f"   {final_output_dir}/{dataset_name}_test_windows.pkl")
                    print("\n PKL merge completed successfully!")
                else:
                    error_msg = "No data was processed in merge mode"
                    print(f"Error: {error_msg}")
                    error_logger.log_general_error("MERGE_MODE", error_msg)
                
            except Exception as e:
                error_type, error_message, full_traceback = capture_exception_info()
                error_logger.log_general_error("MERGE_MODE", f"{error_type}: {error_message}", full_traceback)
                print(f"Error in merge mode: {error_message}")
                raise
            
            finally:
                error_logger.finalize_logging()
            
            return
        
        # Full 모드 처리 (기존 로직)
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
        print(f"Mode: {args.mode}")
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
        
        # 모델을 한 번만 로드하기 위해 멀티프로세싱 비활성화 옵션 추가
        if args.max_workers > 1:
            print(f"Warning: Using max_workers > 1 ({args.max_workers}) will cause model reloading for each process")
            print("Consider using --max-workers 1 for optimal memory efficiency")
        
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
            overlay_fps=args.overlay_fps,
            # 포즈 추출 파라미터
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            quality_threshold=args.quality_threshold,
            min_track_length=args.min_track_length,
            # ByteTracker 파라미터
            track_high_thresh=args.track_high_thresh,
            track_low_thresh=args.track_low_thresh,
            track_max_disappeared=args.track_max_disappeared,
            track_min_hits=args.track_min_hits,
            # 복합 점수 가중치
            weights=[
                args.movement_weight, 
                args.position_weight, 
                args.interaction_weight, 
                args.temporal_weight, 
                args.persistence_weight
            ],
            # 윈도우 처리 파라미터
            min_success_rate=args.min_success_rate
        )
        
        # 비디오 파일 수집
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        all_video_files = []
        
        for ext in video_extensions:
            all_video_files.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))
        
        if not all_video_files:
            print(f" No videos found in {args.input_dir}")
            return
        
        print(f"Found {len(all_video_files)} total videos in dataset")
        
        # Resume 기능 처리
        if args.resume:
            print("\nResume mode enabled - comparing input and output directories...")
            all_video_files_in_input = glob.glob(os.path.join(args.input_dir, '**', '*.mp4'), recursive=True) + \
                                     glob.glob(os.path.join(args.input_dir, '**', '*.avi'), recursive=True) + \
                                     glob.glob(os.path.join(args.input_dir, '**', '*.mov'), recursive=True) + \
                                     glob.glob(os.path.join(args.input_dir, '**', '*.mkv'), recursive=True)
            total_input_videos = len(all_video_files_in_input)

            video_files, processed_count = get_unprocessed_videos(args.input_dir, args.output_dir)
            unprocessed_count = len(video_files)

            # 상태 요약 출력 및 로깅
            summary = (
                f"\n{'='*60}\n"
                f" Resume Status Summary\n"
                f"{ '='*60}\n"
                f"Total videos in input directory: {total_input_videos:,}\n"
                f"Processed videos (found in output dir): {processed_count:,}\n"
                f"Remaining to process: {unprocessed_count:,}\n"
                f"{ '='*60}"
            )
            print(summary)
            error_logger.log_info(summary)

            if unprocessed_count == 0:
                print("\nAll videos appear to be processed. Nothing to do.")
                error_logger.finalize_logging()
                return
            else:
                print(f"\nResuming processing for {unprocessed_count:,} remaining videos...")
                print()
        else:
            video_files = all_video_files
            print(f"Processing all {len(video_files)} videos (resume disabled)")
            print()
        
        # 모델과 추출기 사전 로드 (배치 처리 시작 전)
        print(" Pre-loading pose model and extractor for batch processing...")
        print("-" * 50)
        processor._initialize_pose_model()
        processor._initialize_pose_extractor()
        print("Model and extractor pre-loaded successfully")
        print()
        
        # Step 1: 비디오들을 윈도우 기반으로 처리
        print(" Step 1: Processing videos with window-based approach")
        print("-" * 50)
        
        # 에러 로깅을 포함한 비디오 처리
        video_results_list = process_videos_with_logging(
            processor, video_files, args, error_logger
        )
        
        if not video_results_list:
            error_msg = "No videos were successfully processed"
            print(f" {error_msg}")
            error_logger.log_general_error("VIDEO_PROCESSING", error_msg)
            return
        
        print(f" Generated window data for {len(video_results_list)} videos")
        print()

        # Step 2: 윈도우 데이터들을 통합 STGCN 데이터로 변환 및 temp에서 train/val/test 구조로 재배치
        print(" Step 2: Creating unified STGCN data and organizing final directory structure")
        print("-" * 50)
        
        try:
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
            
        except Exception as e:
            error_type, error_message, full_traceback = capture_exception_info()
            error_logger.log_general_error("STGCN_DATA_CREATION", f"{error_type}: {error_message}", full_traceback)
            print(f"Error creating unified STGCN data: {error_message}")
            raise
        
        # 결과 안내
        final_output_dir = os.path.join(args.output_dir, dataset_name)
        
        print(" Generated Directory Structure:")
        print("-" * 50)
        print(f"{final_output_dir}/")
        print("├── train/                              # 최종 분배된 학습 데이터")
        print("│   ├── Fight/")
        print("│   │   ├── video1/")
        print("│   │   │   ├── video1_windows.pkl")
        print("│   │   │   ├── video1_0.mp4")
        print("│   │   │   ├── video1_1.mp4")
        print("│   │   │   └── ...")
        print("│   │   └── video2/...")
        print("│   └── Normal/")
        print("│       └── ...")
        print("├── val/                                # 최종 분배된 검증 데이터")
        print("│   ├── Fight/")
        print("│   └── Normal/")
        print("├── test/                               # 최종 분배된 테스트 데이터")
        print("│   ├── Fight/")
        print("│   └── Normal/")
        print("├── {}_train_windows.pkl                # 통합 학습 PKL".format(dataset_name))
        print("├── {}_val_windows.pkl                  # 통합 검증 PKL".format(dataset_name))
        print("└── {}_test_windows.pkl                 # 통합 테스트 PKL".format(dataset_name))
        print()
        print(" Processing Workflow:")
        print("-" * 50)
        print("1. Input videos processed → temp/train|val/Fight|Normal/video_name/")
        print("2. All inference completed → split into train/val/test by ratio")
        print("3. Files moved from temp → final train/val/test structure") 
        print("4. Unified PKL files created → temp folder cleaned up")
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
    
    except Exception as e:
        error_type, error_message, full_traceback = capture_exception_info()
        error_logger.log_general_error("MAIN_PIPELINE", f"{error_type}: {error_message}", full_traceback)
        print(f"\nPipeline failed with error: {error_message}")
        raise

    finally:
        # 최종 에러 로깅 정리
        error_logger.finalize_logging()

if __name__ == "__main__":
    main()