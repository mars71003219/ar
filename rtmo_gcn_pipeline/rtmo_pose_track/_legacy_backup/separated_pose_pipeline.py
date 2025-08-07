#!/usr/bin/env python3
"""
분리된 포즈 추정 및 트래킹 파이프라인

3단계 분리 처리:
1. 포즈 추정: 설정별 원본 포즈 데이터 저장 (PKL)
2. 트래킹: 포즈 데이터 기반 트래킹 및 복합점수 계산  
3. 통합: train/val/test 분할 및 통합 PKL 생성

사용법:
  # 전체 파이프라인 실행
  python separated_pose_pipeline.py
  
  # 특정 단계만 실행
  python separated_pose_pipeline.py --stage 1
  python separated_pose_pipeline.py --stage 2
  python separated_pose_pipeline.py --stage 3
  
  # 다른 설정 파일 사용
  python separated_pose_pipeline.py --config configs/custom_config.py
  
  # Resume 기능 (이미 처리된 파일은 건너뜀)
  python separated_pose_pipeline.py --resume
  
  # Force 기능 (모든 파일 재처리)
  python separated_pose_pipeline.py --force
"""

import os
import sys
import pickle
import glob
import importlib.util
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from enhanced_rtmo_bytetrack_pose_extraction import EnhancedRTMOPoseExtractor
from error_logger import ProcessingErrorLogger


class SeparatedPosePipeline:
    """분리된 포즈 추정 파이프라인"""
    
    def __init__(self, config_module):
        self.config = config_module
        self.pose_extractor = None
        self.error_logger = ProcessingErrorLogger(
            os.path.join(self.config.output_dir, 'processing_errors.log')
        )
        
        # 출력 디렉토리 생성
        self._create_output_directories()
    
    def _get_weights_as_list(self):
        """가중치를 리스트 형태로 반환 (기존 코드 호환성)"""
        return [
            self.config.movement_weight,
            self.config.position_weight,
            self.config.interaction_weight,
            self.config.temporal_consistency_weight,
            self.config.persistence_weight
        ]
    
    def _create_output_directories(self):
        """출력 디렉토리 구조 생성"""
        # input_dir의 마지막 폴더명을 dataset_name으로 사용
        dataset_name = os.path.basename(self.config.input_dir.rstrip('/\\'))
        
        # 1단계: 포즈 추정 결과 저장 경로
        pose_settings = f"score{self.config.score_thr}_nms{self.config.nms_thr}"
        self.poses_output_dir = os.path.join(
            self.config.output_dir,
            dataset_name,
            "step1_poses", 
            pose_settings
        )
        
        # 2단계: 트래킹 결과 저장 경로  
        tracking_settings = (f"clip{self.config.clip_len}_stride{self.config.training_stride}_"
                           f"thresh{self.config.track_high_thresh}_{self.config.track_low_thresh}_"
                           f"quality{self.config.quality_threshold}")
        self.tracking_output_dir = os.path.join(
            self.config.output_dir,
            dataset_name,
            "step2_tracking", 
            tracking_settings
        )
        
        # 3단계: 통합 PKL 저장 경로
        self.unified_output_dir = os.path.join(
            self.config.output_dir,
            dataset_name,
            "step3_unified"
        )
        
        # 디렉토리 생성
        # step1과 step2는 Fight/NonFight 폴더 생성
        for output_dir in [self.poses_output_dir, self.tracking_output_dir]:
            os.makedirs(output_dir, exist_ok=True)
            for category in ['Fight', 'NonFight']:
                os.makedirs(os.path.join(output_dir, category), exist_ok=True)
        
        # step3는 통합 폴더만 생성 (Fight/NonFight 폴더 없음)
        os.makedirs(self.unified_output_dir, exist_ok=True)
    
    def _initialize_pose_extractor(self):
        """포즈 추출기 초기화"""
        if self.pose_extractor is None:
            print(f"Initializing pose extractor...")
            self.pose_extractor = EnhancedRTMOPoseExtractor(
                config_file=self.config.detector_config,
                checkpoint_file=self.config.detector_checkpoint,
                device=self.config.device,
                score_thr=self.config.score_thr,
                nms_thr=self.config.nms_thr,
                track_high_thresh=self.config.track_high_thresh,
                track_low_thresh=self.config.track_low_thresh,
                track_max_disappeared=self.config.track_max_disappeared,
                track_min_hits=self.config.track_min_hits,
                quality_threshold=self.config.quality_threshold,
                min_track_length=self.config.min_track_length,
                weights=self._get_weights_as_list()
            )
            print("Pose extractor initialized successfully")
    
    def run_stage1_pose_extraction(self, force_reprocess=False):
        """1단계: 포즈 추정만 수행"""
        print("=" * 60)
        print("STAGE 1: Pose Extraction Only")
        print("=" * 60)
        
        self._initialize_pose_extractor()
        
        # 비디오 파일 수집
        video_files = self._collect_video_files()
        
        if force_reprocess:
            remaining_videos = video_files
            print("Force mode: Reprocessing all videos")
        else:
            processed_videos = self._get_processed_videos_stage1()
            remaining_videos = [v for v in video_files if self._get_video_key(v) not in processed_videos]
            print(f"Total videos: {len(video_files)}")
            print(f"Already processed: {len(processed_videos)}")
        
        print(f"Remaining videos: {len(remaining_videos)}")
        
        if not remaining_videos:
            print("All videos already processed for stage 1")
            return
        
        # 포즈 추정 수행
        for video_path in tqdm(remaining_videos, desc="Extracting poses"):
            try:
                self._extract_poses_from_video(video_path)
            except Exception as e:
                self.error_logger.log_video_failure(video_path, None, "POSE_EXTRACTION_ERROR", str(e))
                print(f"Error processing {video_path}: {str(e)}")
        
        print(f"Stage 1 completed. Results saved to: {self.poses_output_dir}")
    
    def run_stage2_tracking(self, force_reprocess=False):
        """2단계: 트래킹 및 복합점수 계산"""
        print("=" * 60)
        print("STAGE 2: Tracking and Composite Scoring")
        print("=" * 60)
        
        self._initialize_pose_extractor()
        
        # 1단계 결과 수집
        pose_files = self._collect_pose_results()
        
        if not pose_files:
            print("No pose files found from stage 1. Please run stage 1 first.")
            return
        
        if force_reprocess:
            remaining_files = pose_files
            print("Force mode: Reprocessing all pose files")
        else:
            processed_videos = self._get_processed_videos_stage2()
            remaining_files = [f for f in pose_files if self._get_video_key_from_pose_path(f) not in processed_videos]
            print(f"Total pose files: {len(pose_files)}")
            print(f"Already processed: {len(processed_videos)}")
        
        print(f"Remaining files: {len(remaining_files)}")
        
        if not remaining_files:
            print("All videos already processed for stage 2")
            return
        
        # 트래킹 수행
        for poses_file in tqdm(remaining_files, desc="Applying tracking"):
            try:
                self._apply_tracking_to_video(poses_file)
            except Exception as e:
                self.error_logger.log_video_failure(poses_file, None, "TRACKING_ERROR", str(e))
                print(f"Error processing {poses_file}: {str(e)}")
        
        print(f"Stage 2 completed. Results saved to: {self.tracking_output_dir}")
    
    def run_stage3_unification(self, force_reprocess=False):
        """3단계: 통합 PKL 생성"""
        print("=" * 60)
        print("STAGE 3: Dataset Unification")
        print("=" * 60)
        
        # 2단계 결과 수집
        tracking_results = self._collect_tracking_results()
        
        if not tracking_results:
            print("No tracking results found from stage 2. Please run stage 2 first.")
            return
        
        # 기존 통합 결과 확인
        existing_unified_files = self._get_existing_unified_files()
        
        if not force_reprocess and existing_unified_files:
            print(f"Unified files already exist: {existing_unified_files}")
            print("Use --force to reprocess unified dataset")
            return
        
        if force_reprocess and existing_unified_files:
            print("Force mode: Reprocessing unified dataset")
        
        # 데이터셋 분할 및 통합 PKL 생성
        self._create_unified_dataset(tracking_results)
        
        print(f"Stage 3 completed. Results saved to: {self.unified_output_dir}")
    
    def _collect_video_files(self) -> List[str]:
        """비디오 파일 수집"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = []
        
        for root, dirs, files in os.walk(self.config.input_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return sorted(video_files)
    
    def _get_video_key(self, video_path: str) -> str:
        """비디오 경로에서 키 생성"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        category = 'Fight' if '/Fight/' in video_path else 'NonFight'
        return f"{category}/{video_name}"
    
    def _get_processed_videos_stage1(self) -> set:
        """1단계에서 이미 처리된 비디오 확인"""
        processed = set()
        
        for category in ['Fight', 'NonFight']:
            category_dir = os.path.join(self.poses_output_dir, category)
            if os.path.exists(category_dir):
                for file_name in os.listdir(category_dir):
                    if file_name.endswith('_poses.pkl'):
                        # 파일명에서 비디오 이름 추출
                        video_name = file_name.replace('_poses.pkl', '')
                        processed.add(f"{category}/{video_name}")
        
        return processed
    
    def _get_existing_unified_files(self) -> List[str]:
        """3단계에서 이미 생성된 통합 파일 확인"""
        dataset_name = os.path.basename(self.config.input_dir.rstrip('/\\\\'))
        unified_files = []
        
        for split in ['train', 'val', 'test']:
            pkl_filename = f"{dataset_name}_{split}_windows.pkl"
            pkl_path = os.path.join(self.unified_output_dir, pkl_filename)
            if os.path.exists(pkl_path):
                unified_files.append(pkl_filename)
        
        return unified_files
    
    def _extract_poses_from_video(self, video_path: str):
        """비디오에서 포즈 추정 수행"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        category = 'Fight' if '/Fight/' in video_path else 'NonFight'
        
        print(f"  Processing: {video_name}")
        
        # 포즈 추정 수행
        pose_results = self.pose_extractor.extract_poses_only(video_path, self.error_logger)
        
        if pose_results is None or len(pose_results) == 0:
            raise ValueError(f"No poses extracted from {video_name}")
        
        # PKL 파일로 저장 (바로 category 폴더에 저장)
        poses_data = {
            'video_name': video_name,
            'category': category,
            'total_frames': len(pose_results),
            'poses': pose_results,
            'settings': {
                'score_thr': self.config.score_thr,
                'nms_thr': self.config.nms_thr,
                'device': self.config.device
            }
        }
        
        poses_file = os.path.join(self.poses_output_dir, category, f"{video_name}_poses.pkl")
        with open(poses_file, 'wb') as f:
            pickle.dump(poses_data, f)
        
        print(f"  Saved {len(pose_results)} pose frames to: {poses_file}")
    
    def _collect_pose_results(self) -> List[str]:
        """1단계 포즈 결과 수집"""
        pose_files = []
        
        for category in ['Fight', 'NonFight']:
            category_dir = os.path.join(self.poses_output_dir, category)
            if os.path.exists(category_dir):
                for file_name in os.listdir(category_dir):
                    if file_name.endswith('_poses.pkl'):
                        poses_file = os.path.join(category_dir, file_name)
                        if os.path.exists(poses_file):
                            pose_files.append(poses_file)
        
        return sorted(pose_files)
    
    def _get_video_key_from_pose_path(self, pose_file: str) -> str:
        """포즈 파일 경로에서 비디오 키 생성"""
        file_name = os.path.basename(pose_file)
        video_name = file_name.replace('_poses.pkl', '')
        category = 'Fight' if '/Fight/' in pose_file else 'NonFight'
        return f"{category}/{video_name}"
    
    def _get_processed_videos_stage2(self) -> set:
        """2단계에서 이미 처리된 비디오 확인"""
        processed = set()
        
        for category in ['Fight', 'NonFight']:
            category_dir = os.path.join(self.tracking_output_dir, category)
            if os.path.exists(category_dir):
                for video_file in os.listdir(category_dir):
                    if video_file.endswith('_windows.pkl'):
                        video_name = video_file.replace('_windows.pkl', '')
                        processed.add(f"{category}/{video_name}")
        
        return processed
    
    def _apply_tracking_to_video(self, poses_file: str):
        """포즈 데이터에 트래킹 적용 (기존 로직 사용)"""
        from enhanced_rtmo_bytetrack_pose_extraction import (
            ByteTracker, create_detection_results, assign_track_ids_from_bytetrack,
            create_enhanced_annotation
        )
        
        # 포즈 데이터 로드
        with open(poses_file, 'rb') as f:
            pose_data = pickle.load(f)
        
        video_name = pose_data['video_name']
        category = pose_data['category']
        poses = pose_data['poses']
        
        print(f"  Processing: {video_name} ({len(poses)} frames)")
        
        # 최소 프레임 패딩 적용 (윈도우 생성 전에 수행)
        if len(poses) < self.config.clip_len:
            print(f"    Applying padding: {len(poses)} -> {self.config.clip_len} frames")
            poses = self._apply_temporal_padding(poses, self.config.clip_len)
        
        # 윈도우 생성
        windows = self._create_windows_from_poses(poses)
        processed_windows = []
        
        # 포즈 모델 (어노테이션 생성에 필요)
        pose_model = self.pose_extractor.pose_model
        
        print(f"    Processing {len(windows)} windows...")
        
        for window_idx, window_poses in enumerate(windows):
            try:
                # ByteTracker 초기화 (각 윈도우마다 새로 초기화)
                tracker = ByteTracker(
                    high_thresh=self.config.track_high_thresh,
                    low_thresh=self.config.track_low_thresh,
                    max_disappeared=self.config.track_max_disappeared,
                    min_hits=self.config.track_min_hits
                )
                
                # 포즈 데이터에 트래킹 적용
                tracked_pose_results = []
                for pose_result in window_poses:
                    detections = create_detection_results(pose_result)
                    active_tracks = tracker.update(detections)
                    tracked_result = assign_track_ids_from_bytetrack(pose_result, active_tracks)
                    tracked_pose_results.append(tracked_result)
                
                # 기존 로직으로 복합점수 계산 및 어노테이션 생성
                annotation, status_message = create_enhanced_annotation(
                    tracked_pose_results, 
                    video_name,  # video_path 대신 video_name 사용
                    pose_model,
                    min_track_length=self.config.min_track_length,
                    quality_threshold=self.config.quality_threshold,
                    weights=self._get_weights_as_list()
                )
                
                if annotation is None:
                    print(f"      Warning: Window {window_idx} annotation failed: {status_message}")
                    continue
                
                # 윈도우 데이터 구성 (unified_pose_processor 형식과 동일)
                start_frame = window_idx * self.config.training_stride
                end_frame = min(start_frame + self.config.clip_len, len(poses))
                
                window_result = {
                    'window_idx': window_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'num_frames': end_frame - start_frame,
                    'annotation': annotation,
                    'segment_video_path': None,  # 분리된 파이프라인에서는 비디오 생성 안함
                    'persons_ranking': self._extract_persons_ranking(annotation)
                }
                
                processed_windows.append(window_result)
                
            except Exception as e:
                print(f"      Error processing window {window_idx}: {str(e)}")
                continue
        
        if not processed_windows:
            raise ValueError(f"No valid windows processed for {video_name}")
        
        # 윈도우들을 복합점수로 정렬 (내림차순)
        processed_windows = self._sort_windows_by_composite_score(processed_windows)
        
        # 비디오 결과 구성
        video_result = {
            'video_name': video_name,
            'label_folder': category,
            'label': 1 if category == 'Fight' else 0,
            'dataset_name': 'separated_pipeline',
            'total_frames': len(poses),
            'num_windows': len(processed_windows),
            'windows': processed_windows,
            'tracking_settings': {
                'track_high_thresh': self.config.track_high_thresh,
                'track_low_thresh': self.config.track_low_thresh,
                'track_max_disappeared': self.config.track_max_disappeared,
                'track_min_hits': self.config.track_min_hits,
                'quality_threshold': self.config.quality_threshold,
                'min_track_length': self.config.min_track_length,
                'weights': self._get_weights_as_list(),
                'clip_len': self.config.clip_len,
                'training_stride': self.config.training_stride
            }
        }
        
        # PKL 파일로 저장
        output_pkl = os.path.join(self.tracking_output_dir, category, f"{video_name}_windows.pkl")
        with open(output_pkl, 'wb') as f:
            pickle.dump(video_result, f)
        
        print(f"  Successfully processed {len(processed_windows)} windows for {video_name}")
        print(f"  Saved to: {output_pkl}")
    
    def _create_windows_from_poses(self, poses: List) -> List[List]:
        """포즈 데이터에서 윈도우 생성 (패딩은 이미 적용된 상태)"""
        windows = []
        total_frames = len(poses)
        
        # 이미 패딩이 적용되어 total_frames >= clip_len 보장됨
        for start_frame in range(0, total_frames - self.config.clip_len + 1, self.config.training_stride):
            end_frame = start_frame + self.config.clip_len
            window_poses = poses[start_frame:end_frame]
            windows.append(window_poses)
        
        return windows
    
    def _apply_temporal_padding(self, poses: List, target_length: int) -> List:
        """시간적 패딩 적용 - modulo 루핑 방식"""
        if len(poses) >= target_length:
            return poses[:target_length]
        
        if not poses:
            # 빈 포즈 리스트인 경우 빈 프레임으로 채움
            return [[]] * target_length
        
        # modulo 루핑 방식으로 패딩
        padded_poses = []
        
        for i in range(target_length):
            # 순환 인덱스 사용 (비디오 전체를 반복)
            pose_idx = i % len(poses)
            padded_poses.append(poses[pose_idx])
        
        return padded_poses
    
    def _extract_persons_ranking(self, annotation: Dict) -> List[Dict]:
        """어노테이션에서 person 랭킹 추출"""
        if 'annotation' not in annotation:
            return []
        
        persons = []
        for person_key, person_data in annotation['annotation'].items():
            if person_key.startswith('person_'):
                persons.append({
                    'person_id': person_key,
                    'rank': person_data.get('rank', 999),
                    'composite_score': person_data.get('composite_score', 0.0),
                    'track_id': person_data.get('track_id', -1)
                })
        
        # 랭킹 순서로 정렬
        persons.sort(key=lambda x: x['rank'])
        return persons
    
    def _sort_windows_by_composite_score(self, windows: List[Dict]) -> List[Dict]:
        """윈도우들을 복합점수로 정렬"""
        def get_window_composite_score(window: Dict) -> float:
            """윈도우의 복합점수 계산 (모든 person의 평균)"""
            annotation = window.get('annotation', {})
            if 'annotation' not in annotation:
                return 0.0
            
            scores = []
            for person_key, person_data in annotation['annotation'].items():
                if person_key.startswith('person_'):
                    score = person_data.get('composite_score', 0.0)
                    scores.append(score)
            
            return sum(scores) / len(scores) if scores else 0.0
        
        # 각 윈도우에 복합점수 추가
        for window in windows:
            window['composite_score'] = get_window_composite_score(window)
        
        # 복합점수로 내림차순 정렬
        windows.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        
        return windows
    
    def _collect_tracking_results(self) -> List[str]:
        """2단계 트래킹 결과 수집"""
        tracking_files = []
        
        for category in ['Fight', 'NonFight']:
            category_dir = os.path.join(self.tracking_output_dir, category)
            if os.path.exists(category_dir):
                for pkl_file in glob.glob(os.path.join(category_dir, '*_windows.pkl')):
                    tracking_files.append(pkl_file)
        
        return sorted(tracking_files)
    
    def _create_unified_dataset(self, tracking_results: List[str]):
        """통합 데이터셋 생성"""
        print(f"Creating unified dataset from {len(tracking_results)} tracking results...")
        
        # 모든 트래킹 결과 로드
        all_video_data = []
        for pkl_file in tracking_results:
            try:
                with open(pkl_file, 'rb') as f:
                    video_data = pickle.load(f)
                    all_video_data.append(video_data)
            except Exception as e:
                print(f"Error loading {pkl_file}: {str(e)}")
                continue
        
        if not all_video_data:
            print("No valid tracking results found")
            return
        
        print(f"Loaded {len(all_video_data)} video results")
        
        # 카테고리별로 분리
        fight_videos = [v for v in all_video_data if v['label'] == 1]
        nonfight_videos = [v for v in all_video_data if v['label'] == 0]
        
        print(f"Fight videos: {len(fight_videos)}, NonFight videos: {len(nonfight_videos)}")
        
        # 데이터셋 분할
        train_data, val_data, test_data = self._split_dataset(fight_videos, nonfight_videos)
        
        # 통합 PKL 파일 생성
        dataset_name = os.path.basename(self.config.input_dir.rstrip('/\\'))
        
        splits = {
            'train': train_data,
            'val': val_data, 
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            if split_data:
                pkl_filename = f"{dataset_name}_{split_name}_windows.pkl"
                pkl_path = os.path.join(self.unified_output_dir, pkl_filename)
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(split_data, f)
                
                print(f"  {split_name}: {len(split_data)} videos -> {pkl_path}")
            else:
                print(f"  {split_name}: No data")
        
        # 통계 정보 저장
        self._save_dataset_statistics(all_video_data, dataset_name)
    
    def _split_dataset(self, fight_videos: List[Dict], nonfight_videos: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """데이터셋을 train/val/test로 분할"""
        import random
        
        # 재현 가능한 분할을 위한 시드 설정
        random.seed(42)
        
        def split_category(videos: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
            """카테고리별 분할"""
            random.shuffle(videos)
            total = len(videos)
            
            train_end = int(total * self.config.train_ratio)
            val_end = train_end + int(total * self.config.val_ratio)
            
            train = videos[:train_end]
            val = videos[train_end:val_end]
            test = videos[val_end:]
            
            return train, val, test
        
        # 각 카테고리별로 분할
        fight_train, fight_val, fight_test = split_category(fight_videos)
        nonfight_train, nonfight_val, nonfight_test = split_category(nonfight_videos)
        
        # 통합
        train_data = fight_train + nonfight_train
        val_data = fight_val + nonfight_val
        test_data = fight_test + nonfight_test
        
        # 셔플
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data
    
    def _save_dataset_statistics(self, all_video_data: List[Dict], dataset_name: str):
        """데이터셋 통계 정보 저장"""
        stats = {
            'dataset_name': dataset_name,
            'total_videos': len(all_video_data),
            'fight_videos': len([v for v in all_video_data if v['label'] == 1]),
            'nonfight_videos': len([v for v in all_video_data if v['label'] == 0]),
            'total_windows': sum(v['num_windows'] for v in all_video_data),
            'avg_windows_per_video': sum(v['num_windows'] for v in all_video_data) / len(all_video_data),
            'avg_frames_per_video': sum(v['total_frames'] for v in all_video_data) / len(all_video_data),
            'config': {
                'clip_len': self.config.clip_len,
                'training_stride': self.config.training_stride,
                'score_thr': self.config.score_thr,
                'nms_thr': self.config.nms_thr,
                'track_high_thresh': self.config.track_high_thresh,
                'track_low_thresh': self.config.track_low_thresh,
                'quality_threshold': self.config.quality_threshold,
                'min_track_length': self.config.min_track_length,
                'weights': self._get_weights_as_list()
            },
            'split_ratios': {
                'train': self.config.train_ratio,
                'val': self.config.val_ratio,
                'test': self.config.test_ratio
            }
        }
        
        # 윈도우별 통계
        all_windows = []
        for video in all_video_data:
            all_windows.extend(video['windows'])
        
        if all_windows:
            window_scores = [w.get('composite_score', 0.0) for w in all_windows]
            stats['window_statistics'] = {
                'total_windows': len(all_windows),
                'avg_composite_score': sum(window_scores) / len(window_scores),
                'max_composite_score': max(window_scores),
                'min_composite_score': min(window_scores)
            }
        
        # 통계 파일 저장
        import json
        stats_file = os.path.join(self.unified_output_dir, f"{dataset_name}_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics saved to: {stats_file}")
        
        # 간단한 요약 출력
        print(f"\nDataset Summary:")
        print(f"  Total Videos: {stats['total_videos']} (Fight: {stats['fight_videos']}, NonFight: {stats['nonfight_videos']})")
        print(f"  Total Windows: {stats['total_windows']}")
        print(f"  Avg Windows/Video: {stats['avg_windows_per_video']:.1f}")
        print(f"  Avg Frames/Video: {stats['avg_frames_per_video']:.1f}")
        if 'window_statistics' in stats:
            print(f"  Avg Composite Score: {stats['window_statistics']['avg_composite_score']:.3f}")
            print(f"  Score Range: {stats['window_statistics']['min_composite_score']:.3f} - {stats['window_statistics']['max_composite_score']:.3f}")


def load_config(config_path: str):
    """Python 설정 파일 로드"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def main():
    parser = argparse.ArgumentParser(description="Separated Pose Pipeline")
    parser.add_argument('--config', type=str, 
                       default='configs/separated_pipeline_config.py',
                       help='Configuration file path')
    parser.add_argument('--stage', type=str, choices=['1', '2', '3', 'all'], default='all',
                       help='Pipeline stage to run (1: pose extraction, 2: tracking, 3: unification)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint (smart resume)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of all files (ignore existing results)')
    
    args = parser.parse_args()
    
    # 인자 검증
    if args.resume and args.force:
        print("Error: --resume and --force options are mutually exclusive")
        print("--resume: Skip already processed files (default behavior)")
        print("--force: Reprocess all files")
        sys.exit(1)
    
    # Config 파일 확인
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Config 로드
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        sys.exit(1)
    
    # 입력 디렉토리 확인
    if not os.path.exists(config.input_dir):
        print(f"Error: Input directory not found: {config.input_dir}")
        sys.exit(1)
    
    # 파이프라인 초기화
    pipeline = SeparatedPosePipeline(config)
    
    print("=" * 80)
    print("SEPARATED POSE PIPELINE")
    print("=" * 80)
    print(f"Input Directory: {config.input_dir}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Detector: {config.detector_config}")
    print(f"Device: {config.device}")
    print(f"Clip Length: {config.clip_len}, Stride: {config.training_stride}")
    print(f"Stage to run: {args.stage}")
    
    # Resume/Force 모드 정보 출력
    if args.resume:
        print("Resume Mode: Skipping already processed files")
    if args.force:
        print("Force Mode: Reprocessing all files")
    print()
    
    try:
        # 단계별 실행
        if args.stage in ['1', 'all']:
            print("Starting Stage 1: Pose Extraction...")
            pipeline.run_stage1_pose_extraction(force_reprocess=args.force)
            print("Stage 1 completed!\n")
        
        if args.stage in ['2', 'all']:
            print("Starting Stage 2: Tracking and Scoring...")
            pipeline.run_stage2_tracking(force_reprocess=args.force)
            print("Stage 2 completed!\n")
        
        if args.stage in ['3', 'all']:
            print("Starting Stage 3: Dataset Unification...")
            pipeline.run_stage3_unification(force_reprocess=args.force)
            print("Stage 3 completed!\n")
        
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # 결과 요약
        dataset_name = os.path.basename(config.input_dir.rstrip('/\\'))
        print("\nOutput Structure:")
        print(f"├── {config.output_dir}")
        print(f"│   └── {dataset_name}/")
        print("│       ├── step1_poses/")
        print("│       │   └── score{}_nms{}/".format(config.score_thr, config.nms_thr))
        print("│       │       ├── Fight/")
        print("│       │       │   └── [video_name]_poses.pkl")
        print("│       │       └── NonFight/")
        print("│       │           └── [video_name]_poses.pkl")
        print("│       ├── step2_tracking/")
        print("│       │   └── clip{}_stride{}_thresh{}_{}_quality{}/".format(
            config.clip_len, config.training_stride, 
            config.track_high_thresh, config.track_low_thresh, 
            config.quality_threshold))
        print("│       │       ├── Fight/")
        print("│       │       │   └── [video_name]_windows.pkl")
        print("│       │       └── NonFight/")
        print("│       │           └── [video_name]_windows.pkl")
        print("│       └── step3_unified/")
        print("│           ├── {}_train_windows.pkl".format(dataset_name))
        print("│           ├── {}_val_windows.pkl".format(dataset_name))
        print("│           ├── {}_test_windows.pkl".format(dataset_name))
        print("│           └── {}_statistics.json".format(dataset_name))
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()