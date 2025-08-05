#!/usr/bin/env python3
"""
폭력 감지 데이터 처리 메인 프로그램 (1B 방식)
- 원본 비디오 → 개별 세그먼트 PKL → 통합 STGCN PKL
- 단일 패스 처리로 중복 제거, 효율적 워크플로우
- Config 파일 기반 설정 관리
"""

import os
import sys
import glob
from configs import load_config
from unified_pose_processor import UnifiedPoseProcessor
from error_logger import ProcessingErrorLogger, capture_exception_info


def _is_successful_result(result):
    """처리 결과가 성공인지 확인다."""
    return result and result.get('windows') and len(result['windows']) > 0


def _handle_failed_result(result, video_path, error_logger):
    """실패한 결과를 처리한다."""
    if isinstance(result, dict) and 'failure_stage' in result:
        error_logger.log_video_failure(video_path, failure_analysis=result)
    elif isinstance(result, dict) and 'error_type' in result:
        specific_cause = result.get('specific_cause', 'UNKNOWN_ERROR')
        diagnosis = result.get('diagnosis', 'No diagnosis available')
        error_message = f"{diagnosis} ({result.get('error_message', 'No details')})"
        full_traceback = result.get('traceback', '')
        error_logger.log_video_failure(video_path, None, specific_cause, error_message, full_traceback)
    else:
        error_msg = "Video processing returned None or empty result"
        error_logger.log_video_failure(video_path, None, "PROCESSING_RETURNED_NONE", error_msg)


def _setup_gpu_environment(config):
    """초기 GPU 환경 설정을 처리한다."""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
        print("Warning: PyTorch not available. Using CPU")
    
    device = 'cpu'
    gpu_ids = []
    multi_gpu = False
    
    if config.gpu.lower() == 'cpu':
        print("Using CPU")
    elif torch_available and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        try:
            requested_gpu_ids = [int(x.strip()) for x in config.gpu.split(',')]
            valid_gpu_ids = [gpu_id for gpu_id in requested_gpu_ids if gpu_id < gpu_count]
            
            if not valid_gpu_ids:
                print("Warning: No valid GPU IDs specified. Using GPU 0")
                valid_gpu_ids = [0] if gpu_count > 0 else []
            
            if valid_gpu_ids:
                if len(valid_gpu_ids) > 1:
                    device = f'cuda:{valid_gpu_ids[0]}'
                    gpu_ids = valid_gpu_ids
                    multi_gpu = True
                    print(f"Multi-GPU mode: GPUs {valid_gpu_ids}")
                else:
                    device = f'cuda:{valid_gpu_ids[0]}'
                    gpu_ids = valid_gpu_ids
                    multi_gpu = False
                    print(f"Single GPU mode: GPU {valid_gpu_ids[0]}")
        except ValueError:
            print(f"Warning: Invalid GPU specification '{config.gpu}'. Using CPU")
    else:
        if torch_available:
            print("Warning: CUDA not available. Using CPU")
        else:
            print("Using CPU (PyTorch not available)")
    
    return device, gpu_ids, multi_gpu

def get_unprocessed_videos(input_dir, output_dir):
    """output 디렉토리를 기준으로 처리되지 않은 비디오 목록을 가져옵니다."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    all_input_videos = glob.glob(os.path.join(input_dir, '**', '*.*'), recursive=True)
    
    input_video_map = {}
    for video_path in all_input_videos:
        if video_path.endswith(video_extensions):
            basename = os.path.splitext(os.path.basename(video_path))[0]
            parent_dir = os.path.basename(os.path.dirname(video_path))
            unique_key = f"{parent_dir}/{basename}"
            input_video_map[unique_key] = video_path

    processed_video_keys = _get_processed_videos(output_dir, os.path.basename(input_dir.rstrip('/\\')))
    
    unprocessed_video_keys = set(input_video_map.keys()) - processed_video_keys
    unprocessed_video_paths = [input_video_map[key] for key in unprocessed_video_keys]
    
    return sorted(unprocessed_video_paths), len(processed_video_keys)


def _get_processed_videos(output_dir, dataset_name):
    """처리된 비디오 키 목록을 반환합니다. PKL 파일이 존재하는지 확인."""
    processed_video_keys = set()
    
    # 1. temp 폴더에서 완료된 비디오들 확인 (PKL 파일 존재 여부)
    temp_path = os.path.join(output_dir, dataset_name, 'temp')
    if os.path.exists(temp_path):
        for root, dirs, files in os.walk(temp_path):
            for dir_name in dirs:
                if dir_name not in ['Fight', 'NonFight', 'train', 'val', 'test']:
                    # 비디오 폴더에 PKL 파일이 있는지 확인
                    video_dir_path = os.path.join(root, dir_name)
                    pkl_file = os.path.join(video_dir_path, f"{dir_name}_windows.pkl")
                    
                    if os.path.exists(pkl_file):
                        root_parts = root.split(os.sep)
                        
                        # temp 폴더에서 실제 카테고리 확인하여 입력 키와 매칭
                        if 'Fight' in root_parts:
                            unique_key = f"Fight/{dir_name}"
                            processed_video_keys.add(unique_key)
                        elif 'NonFight' in root_parts:
                            unique_key = f"NonFight/{dir_name}"
                            processed_video_keys.add(unique_key)
    
    # 2. 최종 폴더(train/val/test)에서 완료된 비디오들 확인
    final_dirs = ['train', 'val', 'test']
    for split_dir in final_dirs:
        split_path = os.path.join(output_dir, dataset_name, split_dir)
        if os.path.exists(split_path):
            for category in ['Fight', 'NonFight']:
                category_path = os.path.join(split_path, category)
                if os.path.exists(category_path):
                    for video_dir in os.listdir(category_path):
                        pkl_file = os.path.join(category_path, video_dir, f"{video_dir}_windows.pkl")
                        if os.path.exists(pkl_file):
                            if category == 'Fight':
                                processed_video_keys.add(f"Fight/{video_dir}")
                            else:  # NonFight
                                processed_video_keys.add(f"NonFight/{video_dir}")
    
    return processed_video_keys

def process_videos_with_logging(processor, video_files, config, error_logger):
    """에러 로깅을 포함한 비디오 처리"""
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    successful_videos_data = []
    failed_count = 0
    
    # 각 비디오 처리 시작 로그
    for video_path in video_files:
        error_logger.log_video_start(video_path)
    
    try:
        # max_workers가 1이면 순차 처리 (모델 재로딩 방지)
        if config.max_workers == 1:
            print("Using single-process mode for optimal model efficiency")
            for video_path in tqdm(video_files, desc="Processing videos"):
                try:
                    result = processor.process_single_video_to_segments(
                        video_path, config.output_dir, config.input_dir, config.training_stride
                    )
                    
                    if isinstance(result, dict) and 'failure_stage' in result:
                        failed_count += 1
                        error_logger.log_video_failure(video_path, result)
                        print(f"Failed: {os.path.basename(video_path)} - {result.get('root_cause', 'Unknown error')}")
                    elif isinstance(result, dict) and 'windows' in result:
                        successful_videos_data.append(result)
                        windows_count = len(result.get('windows', []))
                        error_logger.log_video_success(video_path, windows_count)
                        print(f"Success: {os.path.basename(video_path)}")
                    else:
                        failed_count += 1
                        error_logger.log_video_failure(video_path, None, "UNKNOWN_FAILURE", "Processing returned an unexpected result type.")
                        print(f"Failed: {os.path.basename(video_path)} - Unknown error")
                        
                except Exception as e:
                    failed_count += 1
                    error_type, error_message, full_traceback = capture_exception_info()
                    error_logger.log_video_failure(video_path, None, error_type, error_message, full_traceback)
                    print(f"Exception: {os.path.basename(video_path)} - {str(e)}")
        
        else:
            # 멀티프로세싱 사용 (각 프로세스마다 모델 로딩됨)
            print(f"Using multi-process mode with {config.max_workers} workers")
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                # 작업 제출
                executor = ProcessPoolExecutor(max_workers=config.max_workers)
    try:
        # 작업 제출
        future_to_video = {
            executor.submit(
                processor.process_single_video_to_segments, 
                video, config.output_dir, config.input_dir, 
                config.training_stride
            ): video 
            for video in video_files
        }
        
        # 결과 수집
        for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing videos"):
            video_path = future_to_video[future]
            
            try:
                result = future.result()
                
                if _is_successful_result(result):
                    windows_count = len(result['windows'])
                    error_logger.log_video_success(video_path, windows_count)
                    successful_videos_data.append(result)
                else:
                    _handle_failed_result(result, video_path, error_logger)
                    failed_count += 1
                    
            except Exception as exc:
                error_type, error_message, full_traceback = capture_exception_info()
                error_logger.log_video_failure(video_path, None, error_type, error_message, full_traceback)
                failed_count += 1

    finally:
        executor.shutdown(wait=True)  # 모든 작업이 완료될 때까지 기다림
    
    except Exception as e:
        error_type, error_message, full_traceback = capture_exception_info()
        error_logger.log_general_error("BATCH_PROCESSING", f"{error_type}: {error_message}", full_traceback)
        raise
    
    return successful_videos_data

def main():
    # 명령행 인수 처리 (설정 파일 지정용)
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"Loading config from: {config_file}")
    else:
        # 기본 설정 사용
        config_file = None
        print("Using default configuration")
    
    # 추가 덮어쓰기 설정이 있는 경우 처리
    overrides = {}
    for i, arg in enumerate(sys.argv[2:], 2):
        if '=' in arg:
            key, value = arg.split('=', 1)
            # 타입 추론해서 변환
            try:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif '.' in value:
                    value = float(value)
                elif value.isdigit():
                    value = int(value)
            except:
                pass  # 문자열로 유지
            overrides[key] = value
    
    try:
        # 설정 로드
        config = load_config(config_file=config_file, overrides=overrides)
        
        # 설정 출력
        if hasattr(config, 'print_config'):
            config.print_config()
        else:
            print("Configuration loaded successfully")
        
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        print("\nUsage:")
        print("  python run_pose_extraction.py [config_file] [key=value ...]")
        print("\nExamples:")
        print("  python run_pose_extraction.py")
        print("  python run_pose_extraction.py configs/rwf2000_config.py")
        print("  python run_pose_extraction.py configs/rwf2000_config.py mode=merge resume=true")
        print("  python run_pose_extraction.py mode=full gpu=0 max_workers=1")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 데이터셋명 추출 (에러 로깅용)
    dataset_name = os.path.basename(config.input_dir.rstrip('/\\'))
    
    # 에러 로거 초기화 (resume 시 최신 로그에 이어쓰기)
    error_logger = ProcessingErrorLogger(config.output_dir, dataset_name, resume=config.resume)
    
    try:
        # 모드에 따른 분기 처리
        if config.mode == 'merge':
            print("=" * 70)
            print(" PKL Merge Mode - Processing existing data")
            print("=" * 70)
            print(f"Mode: {config.mode}")
            print(f"Processed data directory: {config.input_dir}")
            print(f"Output directory: {config.output_dir}")
            print(f"Train split: {config.train_split}")
            print(f"Val split: {config.val_split}")
            print()
            
            # 간단한 프로세서 초기화 (merge 모드에서는 GPU 설정 불필요)
            processor = UnifiedPoseProcessor(
                detector_config="",  # merge 모드에서는 불필요
                detector_checkpoint="",  # merge 모드에서는 불필요
                device='cpu',
                gpu_ids=[],
                multi_gpu=False,
                clip_len=config.clip_len,
                num_person=config.num_person,
                save_overlay=False,  # merge 모드에서는 오버레이 생성 안함
                overlay_fps=config.overlay_fps,
                # 포즈 추출 파라미터
                score_thr=config.score_thr,
                nms_thr=config.nms_thr,
                quality_threshold=config.quality_threshold,
                min_track_length=config.min_track_length,
                # ByteTracker 파라미터
                track_high_thresh=config.track_high_thresh,
                track_low_thresh=config.track_low_thresh,
                track_max_disappeared=config.track_max_disappeared,
                track_min_hits=config.track_min_hits,
                # 복합 점수 가중치  
                weights=config.get_weights(),
                # 윈도우 처리 파라미터
                min_success_rate=config.min_success_rate
            )
            
            try:
                # merge 모드 실행
                train_count, val_count, test_count = processor.merge_existing_pkl_files(
                    config.input_dir,
                    config.output_dir,
                    train_split=config.train_split,
                    val_split=config.val_split
                )
                
                if train_count + val_count + test_count > 0:
                    # 결과 안내
                    final_output_dir = os.path.join(config.output_dir, dataset_name)
                    
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
        device, gpu_ids, multi_gpu = _setup_gpu_environment(config)
        
        # 처리 정보 출력 (설정에서 이미 출력했으므로 간단히)
        print("\n" + "=" * 70)
        print(" Starting Full Pipeline Processing")
        print("=" * 70)
        print(f"Final Device: {device}")
        print(f"Final GPU IDs: {gpu_ids}")
        print(f"Final Multi-GPU: {multi_gpu}")
        print()
        
        # 모델을 한 번만 로드하기 위해 멀티프로세싱 비활성화 옵션 추가
        if config.max_workers > 1:
            print(f"Warning: Using max_workers > 1 ({config.max_workers}) will cause model reloading for each process")
            print("Consider using max_workers=1 for optimal memory efficiency")
        
        # 통합 포즈 처리기 초기화
        processor = UnifiedPoseProcessor(
            detector_config=config.detector_config,
            detector_checkpoint=config.detector_checkpoint,
            device=device,
            gpu_ids=gpu_ids,
            multi_gpu=multi_gpu,
            clip_len=config.clip_len,
            num_person=config.num_person,
            save_overlay=config.save_overlay,
            overlay_fps=config.overlay_fps,
            # 포즈 추출 파라미터
            score_thr=config.score_thr,
            nms_thr=config.nms_thr,
            quality_threshold=config.quality_threshold,
            min_track_length=config.min_track_length,
            # ByteTracker 파라미터
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            track_max_disappeared=config.track_max_disappeared,
            track_min_hits=config.track_min_hits,
            # 복합 점수 가중치
            weights=config.get_weights(),
            # 윈도우 처리 파라미터
            min_success_rate=config.min_success_rate
        )
        
        # 비디오 파일 수집
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        all_video_files = []
        
        for ext in video_extensions:
            all_video_files.extend(glob.glob(os.path.join(config.input_dir, '**', ext), recursive=True))
        
        if not all_video_files:
            print(f" No videos found in {config.input_dir}")
            return
        
        print(f"Found {len(all_video_files)} total videos in dataset")
        
        # Resume 기능 처리
        if config.resume:
            print("\nResume mode enabled - comparing input and output directories...")
            all_video_files_in_input = glob.glob(os.path.join(config.input_dir, '**', '*.mp4'), recursive=True) + \
                                     glob.glob(os.path.join(config.input_dir, '**', '*.avi'), recursive=True) + \
                                     glob.glob(os.path.join(config.input_dir, '**', '*.mov'), recursive=True) + \
                                     glob.glob(os.path.join(config.input_dir, '**', '*.mkv'), recursive=True)
            total_input_videos = len(all_video_files_in_input)

            video_files, processed_count = get_unprocessed_videos(config.input_dir, config.output_dir)
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
            processor, video_files, config, error_logger
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
                config.output_dir,
                config.input_dir,
                train_split=config.train_split,
                val_split=config.val_split
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
        final_output_dir = os.path.join(config.output_dir, dataset_name)
        
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
        print("│   └── NonFight/")
        print("│       └── ...")
        print("├── val/                                # 최종 분배된 검증 데이터")
        print("│   ├── Fight/")
        print("│   └── NonFight/")
        print("├── test/                               # 최종 분배된 테스트 데이터")
        print("│   ├── Fight/")
        print("│   └── NonFight/")
        print("├── {}_train_windows.pkl                # 통합 학습 PKL".format(dataset_name))
        print("├── {}_val_windows.pkl                  # 통합 검증 PKL".format(dataset_name))
        print("└── {}_test_windows.pkl                 # 통합 테스트 PKL".format(dataset_name))
        print()
        print(" Processing Workflow:")
        print("-" * 50)
        print("1. Input videos processed → temp/train|val/Fight|NonFight/video_name/")
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