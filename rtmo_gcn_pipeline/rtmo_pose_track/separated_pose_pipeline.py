#!/usr/bin/env python3
"""
분리된 포즈 추정 및 트래킹 파이프라인
1. 포즈 추정: 설정별 원본 포즈 데이터 저장
2. 트래킹: 포즈 데이터 기반 트래킹 및 복합점수 계산
3. 통합: train/val/test 분할 및 통합 PKL 생성
"""

import os
import argparse
import json
import pickle
import glob
import hashlib
from pathlib import Path
from collections import defaultdict
from enhanced_rtmo_bytetrack_pose_extraction import EnhancedRTMOPoseExtractor
from error_logger import ProcessingErrorLogger, capture_exception_info


class SeparatedPoseProcessor:
    """포즈 추정과 트래킹을 분리한 처리기"""
    
    def __init__(self, 
                 detector_config: str,
                 detector_checkpoint: str,
                 device: str = 'cuda:0',
                 gpu_ids: list = None,
                 multi_gpu: bool = False,
                 # 포즈 추정 파라미터
                 score_thr: float = 0.3,
                 nms_thr: float = 0.35,
                 # 트래킹 파라미터
                 track_high_thresh: float = 0.6,
                 track_low_thresh: float = 0.1,
                 track_max_disappeared: int = 30,
                 track_min_hits: int = 3,
                 quality_threshold: float = 0.3,
                 min_track_length: int = 10,
                 # 복합 점수 가중치
                 weights: list = None,
                 # 윈도우 설정
                 clip_len: int = 100,
                 training_stride: int = 10):
        
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.device = device
        self.gpu_ids = gpu_ids or []
        self.multi_gpu = multi_gpu
        
        # 포즈 추정 설정
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        
        # 트래킹 설정
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_max_disappeared = track_max_disappeared
        self.track_min_hits = track_min_hits
        self.quality_threshold = quality_threshold
        self.min_track_length = min_track_length
        
        # 복합 점수 가중치
        self.weights = weights or [0.45, 0.10, 0.30, 0.10, 0.05]
        
        # 윈도우 설정
        self.clip_len = clip_len
        self.training_stride = training_stride
        
        # 포즈 추출기
        self.pose_extractor = None
    
    def _initialize_pose_extractor(self):
        """포즈 추출기 초기화"""
        if self.pose_extractor is None:
            print("Initializing pose extractor...")
            self.pose_extractor = EnhancedRTMOPoseExtractor(
                config_file=self.detector_config,
                checkpoint_file=self.detector_checkpoint,
                device=self.device,
                gpu_ids=self.gpu_ids,
                multi_gpu=self.multi_gpu,
                score_thr=self.score_thr,
                nms_thr=self.nms_thr,
                track_high_thresh=self.track_high_thresh,
                track_low_thresh=self.track_low_thresh,
                track_max_disappeared=self.track_max_disappeared,
                track_min_hits=self.track_min_hits,
                quality_threshold=self.quality_threshold,
                min_track_length=self.min_track_length,
                weights=self.weights
            )
            print("Pose extractor initialized successfully")
    
    def _get_pose_estimation_hash(self):
        """포즈 추정 설정의 해시값 생성"""
        pose_config = {
            'model': os.path.basename(self.detector_checkpoint),
            'score_thr': self.score_thr,
            'nms_thr': self.nms_thr
        }
        config_str = json.dumps(pose_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_tracking_hash(self):
        """트래킹 설정의 해시값 생성"""
        tracking_config = {
            'track_high_thresh': self.track_high_thresh,
            'track_low_thresh': self.track_low_thresh,
            'track_max_disappeared': self.track_max_disappeared,
            'track_min_hits': self.track_min_hits,
            'quality_threshold': self.quality_threshold,
            'min_track_length': self.min_track_length,
            'weights': self.weights
        }
        config_str = json.dumps(tracking_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_pose_estimation_folder_name(self):
        """포즈 추정 설정 폴더명 생성"""
        model_name = Path(self.detector_checkpoint).stem.replace('rtmo-', '').replace('_16xb16-600e_body7-640x640-39e78cc4_20231211', '')
        return f"{model_name}_score{self.score_thr}_nms{self.nms_thr}"
    
    def _get_tracking_folder_name(self):
        """트래킹 설정 폴더명 생성"""
        weights_str = "_".join([f"{w:.2f}" for w in self.weights])
        return f"track{self.track_high_thresh}_{self.track_low_thresh}_weights{weights_str}"
    
    def extract_poses_only(self, input_dir: str, output_dir: str):
        """1단계: 포즈 추정만 수행하여 원본 포즈 데이터 저장"""
        dataset_name = os.path.basename(input_dir.rstrip('/\\\\'))
        pose_folder = self._get_pose_estimation_folder_name()
        
        # 포즈 추정 결과 저장 경로
        pose_output_dir = os.path.join(output_dir, dataset_name, 'pose_estimation', pose_folder)
        
        print(f"\\n=== Step 1: Pose Estimation ===")
        print(f"Input: {input_dir}")
        print(f"Output: {pose_output_dir}")
        print(f"Settings: score_thr={self.score_thr}, nms_thr={self.nms_thr}")
        
        # 비디오 파일 수집
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        all_video_files = []
        for ext in video_extensions:
            all_video_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        
        if not all_video_files:
            print(f"No videos found in {input_dir}")
            return False
        
        self._initialize_pose_extractor()
        
        successful_count = 0
        
        for video_path in all_video_files:
            try:
                # 비디오 정보 추출
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                
                # 카테고리 폴더 찾기 (Fight/NonFight)
                path_parts = video_path.replace('\\\\', '/').split('/')
                label_folder = None
                for part in path_parts:
                    if part in ['Fight', 'NonFight']:
                        label_folder = part
                        break
                
                if label_folder is None:
                    print(f"Warning: Could not determine category for {video_path}")
                    continue
                
                # 출력 디렉토리 생성
                video_pose_dir = os.path.join(pose_output_dir, label_folder, video_name)
                os.makedirs(video_pose_dir, exist_ok=True)
                
                # 이미 처리된 경우 스킵
                if self._check_pose_extraction_complete(video_pose_dir):
                    print(f"Skipping {video_name} (already processed)")
                    successful_count += 1
                    continue
                
                print(f"Extracting poses from: {video_name}")
                
                # 포즈 추정 수행
                poses_data = self.pose_extractor.extract_poses_only(video_path)
                
                if poses_data and len(poses_data) > 0:
                    # 윈도우별로 포즈 데이터 저장
                    windows_generated = self._save_pose_windows(poses_data, video_pose_dir, video_name)
                    
                    if windows_generated > 0:
                        print(f"Success: {video_name} - {windows_generated} windows saved")
                        successful_count += 1
                    else:
                        print(f"Failed: {video_name} - No valid windows generated")
                else:
                    print(f"Failed: {video_name} - No poses extracted")
                    
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                continue
        
        print(f"\\nPose extraction completed: {successful_count}/{len(all_video_files)} videos processed")
        return successful_count > 0
    
    def _check_pose_extraction_complete(self, video_pose_dir: str) -> bool:
        """포즈 추출이 완료되었는지 확인"""
        if not os.path.exists(video_pose_dir):
            return False
        
        # window_*.json 파일이 있는지 확인
        window_files = glob.glob(os.path.join(video_pose_dir, 'window_*.json'))
        return len(window_files) > 0
    
    def _save_pose_windows(self, poses_data: list, video_pose_dir: str, video_name: str) -> int:
        """포즈 데이터를 윈도우별로 저장"""
        if not poses_data:
            return 0
        
        total_frames = len(poses_data)
        windows_generated = 0
        
        # 윈도우 생성
        for start_frame in range(0, total_frames - self.clip_len + 1, self.training_stride):
            end_frame = start_frame + self.clip_len
            
            if end_frame > total_frames:
                break
            
            window_idx = start_frame // self.training_stride
            window_poses = poses_data[start_frame:end_frame]
            
            # 윈도우 데이터 구성
            window_data = {
                'video_name': video_name,
                'window_idx': window_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'clip_len': self.clip_len,
                'poses': window_poses,
                'pose_settings': {
                    'score_thr': self.score_thr,
                    'nms_thr': self.nms_thr,
                    'model': os.path.basename(self.detector_checkpoint)
                }
            }
            
            # JSON 파일로 저장
            window_file = os.path.join(video_pose_dir, f'window_{window_idx:03d}_poses.json')
            with open(window_file, 'w', encoding='utf-8') as f:
                json.dump(window_data, f, indent=2, ensure_ascii=False)
            
            windows_generated += 1
        
        return windows_generated
    
    def apply_tracking_to_poses(self, input_dir: str, output_dir: str):
        """2단계: 저장된 포즈 데이터에 트래킹 및 복합점수 적용"""
        dataset_name = os.path.basename(input_dir.rstrip('/\\\\'))
        pose_folder = self._get_pose_estimation_folder_name()
        tracking_folder = self._get_tracking_folder_name()
        
        # 포즈 데이터 경로
        pose_input_dir = os.path.join(output_dir, dataset_name, 'pose_estimation', pose_folder)
        
        # 트래킹 결과 저장 경로
        tracking_output_dir = os.path.join(output_dir, dataset_name, 'tracking', tracking_folder)
        
        print(f"\\n=== Step 2: Tracking & Composite Scoring ===")
        print(f"Pose data: {pose_input_dir}")
        print(f"Output: {tracking_output_dir}")
        print(f"Tracking settings: high={self.track_high_thresh}, low={self.track_low_thresh}")
        print(f"Weights: {self.weights}")
        
        if not os.path.exists(pose_input_dir):
            print(f"Error: Pose data not found at {pose_input_dir}")
            return False
        
        self._initialize_pose_extractor()
        
        successful_count = 0
        
        # 카테고리별로 처리 (Fight, NonFight)
        for category in ['Fight', 'NonFight']:
            category_pose_dir = os.path.join(pose_input_dir, category)
            category_tracking_dir = os.path.join(tracking_output_dir, category)
            
            if not os.path.exists(category_pose_dir):
                continue
            
            os.makedirs(category_tracking_dir, exist_ok=True)
            
            # 각 비디오 처리
            for video_name in os.listdir(category_pose_dir):
                video_pose_dir = os.path.join(category_pose_dir, video_name)
                
                if not os.path.isdir(video_pose_dir):
                    continue
                
                try:
                    # 트래킹 결과 파일 경로
                    video_tracking_dir = os.path.join(category_tracking_dir, video_name)
                    os.makedirs(video_tracking_dir, exist_ok=True)
                    
                    tracking_pkl_file = os.path.join(video_tracking_dir, f"{video_name}_windows.pkl")
                    
                    # 이미 처리된 경우 스킵
                    if os.path.exists(tracking_pkl_file):
                        print(f"Skipping {video_name} (already tracked)")
                        successful_count += 1
                        continue
                    
                    print(f"Processing tracking for: {video_name}")
                    
                    # 윈도우별 포즈 데이터 로드
                    window_files = sorted(glob.glob(os.path.join(video_pose_dir, 'window_*_poses.json')))
                    
                    if not window_files:
                        print(f"No pose windows found for {video_name}")
                        continue
                    
                    # 트래킹 적용
                    video_result = self._apply_tracking_to_windows(window_files, video_name, category)
                    
                    if video_result and video_result.get('windows'):
                        # PKL 파일로 저장
                        with open(tracking_pkl_file, 'wb') as f:
                            pickle.dump(video_result, f)
                        
                        windows_count = len(video_result['windows'])
                        print(f"Success: {video_name} - {windows_count} windows tracked")
                        successful_count += 1
                    else:
                        print(f"Failed: {video_name} - No valid tracking results")
                        
                except Exception as e:
                    print(f"Error processing tracking for {video_name}: {str(e)}")
                    continue
        
        print(f"\\nTracking completed: {successful_count} videos processed")
        return successful_count > 0
    
    def _apply_tracking_to_windows(self, window_files: list, video_name: str, category: str) -> dict:
        """윈도우별 포즈 데이터에 트래킹 적용"""
        try:
            processed_windows = []
            
            for window_file in window_files:
                with open(window_file, 'r', encoding='utf-8') as f:
                    window_data = json.load(f)
                
                # 포즈 데이터에 트래킹 적용
                tracked_result = self.pose_extractor.apply_tracking_to_poses(
                    window_data['poses'], 
                    window_data['start_frame'],
                    window_data['end_frame'],
                    window_data['window_idx']
                )
                
                if tracked_result:
                    processed_windows.append(tracked_result)
            
            if not processed_windows:
                return None
            
            # 비디오 결과 구성
            video_result = {
                'video_name': video_name,
                'label_folder': category,
                'label': 1 if category == 'Fight' else 0,
                'dataset_name': 'retracked',
                'windows': processed_windows,
                'tracking_settings': {
                    'track_high_thresh': self.track_high_thresh,
                    'track_low_thresh': self.track_low_thresh,
                    'track_max_disappeared': self.track_max_disappeared,
                    'track_min_hits': self.track_min_hits,
                    'quality_threshold': self.quality_threshold,
                    'min_track_length': self.min_track_length,
                    'weights': self.weights
                }
            }
            
            return video_result
            
        except Exception as e:
            print(f"Error applying tracking to windows: {str(e)}")
            return None
    
    def create_unified_pkl(self, output_dir: str, dataset_name: str, train_split: float = 0.7, val_split: float = 0.2):
        """3단계: 통합 PKL 파일 생성"""
        tracking_folder = self._get_tracking_folder_name()
        tracking_dir = os.path.join(output_dir, dataset_name, 'tracking', tracking_folder)
        
        print(f"\\n=== Step 3: Creating Unified PKL ===")
        print(f"Tracking data: {tracking_dir}")
        print(f"Split ratios: train={train_split}, val={val_split}, test={1-train_split-val_split}")
        
        if not os.path.exists(tracking_dir):
            print(f"Error: Tracking data not found at {tracking_dir}")
            return False
        
        # PKL 파일들 수집
        video_results = []
        
        for category in ['Fight', 'NonFight']:
            category_dir = os.path.join(tracking_dir, category)
            if not os.path.exists(category_dir):
                continue
            
            for video_name in os.listdir(category_dir):
                video_dir = os.path.join(category_dir, video_name)
                pkl_file = os.path.join(video_dir, f"{video_name}_windows.pkl")
                
                if os.path.exists(pkl_file):
                    try:
                        with open(pkl_file, 'rb') as f:
                            video_result = pickle.load(f)
                        video_results.append(video_result)
                    except Exception as e:
                        print(f"Error loading {pkl_file}: {str(e)}")
                        continue
        
        if not video_results:
            print("No PKL files found for unified processing")
            return False
        
        print(f"Loaded {len(video_results)} video PKL files")
        
        # STGCN 샘플 생성 및 분할 로직은 기존 unified_pose_processor.py의 로직 재사용
        # 여기서는 간단히 통합 PKL만 생성
        all_samples = []
        for video_result in video_results:
            for window_data in video_result['windows']:
                # STGCN 형식으로 변환 (기존 로직 재사용)
                sample = self._convert_to_stgcn_sample(window_data, video_result)
                if sample:
                    all_samples.append(sample)
        
        # 간단한 분할 (실제로는 더 정교한 로직 필요)
        import random
        random.seed(42)
        random.shuffle(all_samples)
        
        total = len(all_samples)
        train_end = int(total * train_split)
        val_end = int(total * (train_split + val_split))
        
        train_samples = all_samples[:train_end]
        val_samples = all_samples[train_end:val_end]
        test_samples = all_samples[val_end:]
        
        # 통합 PKL 저장
        unified_files = {
            'train': f"{dataset_name}_train_windows.pkl",
            'val': f"{dataset_name}_val_windows.pkl", 
            'test': f"{dataset_name}_test_windows.pkl"
        }
        
        for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            if samples:
                unified_file = os.path.join(tracking_dir, unified_files[split_name])
                with open(unified_file, 'wb') as f:
                    pickle.dump(samples, f)
                print(f"Saved {len(samples)} samples to {unified_files[split_name]}")
        
        print(f"\\nUnified PKL creation completed")
        print(f"Total samples: {total} (train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)})")
        
        return True
    
    def _convert_to_stgcn_sample(self, window_data: dict, video_result: dict) -> dict:
        """윈도우 데이터를 STGCN 샘플로 변환 (간단 버전)"""
        try:
            # 실제 구현에서는 unified_pose_processor.py의 _convert_window_to_stgcn_format 로직 사용
            annotation = window_data.get('annotation', {})
            if 'persons' not in annotation or not annotation['persons']:
                return None
            
            # 간단한 STGCN 샘플 구성
            sample = {
                'keypoint': [],  # 실제 키포인트 데이터 처리 필요
                'label': video_result['label'],
                'window_info': {
                    'video_name': video_result['video_name'],
                    'window_idx': window_data.get('window_idx', 0),
                    'start_frame': window_data.get('start_frame', 0),
                    'end_frame': window_data.get('end_frame', 100)
                }
            }
            
            return sample
            
        except Exception as e:
            print(f"Error converting window to STGCN sample: {str(e)}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Separated Pose Estimation and Tracking Pipeline')
    
    # 모드 설정
    parser.add_argument('--mode', type=str, 
                       choices=['pose_only', 'tracking_only', 'unified_only', 'full'], 
                       default='full',
                       help='Processing mode: pose_only, tracking_only, unified_only, or full')
    
    # 기본 설정
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input video directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    
    # 포즈 추정 설정
    parser.add_argument('--detector-config', type=str,
                       default='/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py',
                       help='RTMO detector config file')
    parser.add_argument('--detector-checkpoint', type=str,
                       default='/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth',
                       help='RTMO detector checkpoint')
    parser.add_argument('--score-thr', type=float, default=0.3,
                       help='Pose detection score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.35,
                       help='NMS threshold for pose detection')
    
    # 트래킹 설정
    parser.add_argument('--track-high-thresh', type=float, default=0.6,
                       help='High threshold for ByteTracker')
    parser.add_argument('--track-low-thresh', type=float, default=0.1,
                       help='Low threshold for ByteTracker')
    parser.add_argument('--track-max-disappeared', type=int, default=30,
                       help='Maximum frames a track can be lost')
    parser.add_argument('--track-min-hits', type=int, default=3,
                       help='Minimum hits required for valid track')
    parser.add_argument('--quality-threshold', type=float, default=0.3,
                       help='Minimum quality threshold for tracks')
    parser.add_argument('--min-track-length', type=int, default=10,
                       help='Minimum track length for valid tracks')
    
    # 복합 점수 가중치
    parser.add_argument('--movement-weight', type=float, default=0.45,
                       help='Weight for movement score')
    parser.add_argument('--position-weight', type=float, default=0.10,
                       help='Weight for position score')
    parser.add_argument('--interaction-weight', type=float, default=0.30,
                       help='Weight for interaction score')
    parser.add_argument('--temporal-weight', type=float, default=0.10,
                       help='Weight for temporal consistency')
    parser.add_argument('--persistence-weight', type=float, default=0.05,
                       help='Weight for persistence score')
    
    # 윈도우 설정
    parser.add_argument('--clip-len', type=int, default=100,
                       help='Segment clip length (frames)')
    parser.add_argument('--training-stride', type=int, default=10,
                       help='Stride for dense training segments')
    
    # 데이터 분할
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # GPU 설정
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU to use')
    
    args = parser.parse_args()
    
    # GPU 설정
    device = f'cuda:{args.gpu}' if args.gpu != 'cpu' else 'cpu'
    
    # 처리기 초기화
    processor = SeparatedPoseProcessor(
        detector_config=args.detector_config,
        detector_checkpoint=args.detector_checkpoint,
        device=device,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        track_max_disappeared=args.track_max_disappeared,
        track_min_hits=args.track_min_hits,
        quality_threshold=args.quality_threshold,
        min_track_length=args.min_track_length,
        weights=[
            args.movement_weight,
            args.position_weight,
            args.interaction_weight,
            args.temporal_weight,
            args.persistence_weight
        ],
        clip_len=args.clip_len,
        training_stride=args.training_stride
    )
    
    dataset_name = os.path.basename(args.input_dir.rstrip('/\\\\'))
    
    try:
        if args.mode in ['pose_only', 'full']:
            print("Starting pose extraction...")
            success = processor.extract_poses_only(args.input_dir, args.output_dir)
            if not success:
                print("Pose extraction failed")
                return
        
        if args.mode in ['tracking_only', 'full']:
            print("Starting tracking application...")
            success = processor.apply_tracking_to_poses(args.input_dir, args.output_dir)
            if not success:
                print("Tracking application failed")
                return
        
        if args.mode in ['unified_only', 'full']:
            print("Creating unified PKL files...")
            success = processor.create_unified_pkl(
                args.output_dir, 
                dataset_name,
                args.train_split, 
                args.val_split
            )
            if not success:
                print("Unified PKL creation failed")
                return
        
        print("\\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\\nPipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()