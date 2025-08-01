#!/usr/bin/env python3
"""
통합 포즈 처리 모듈 (1B 방식)
- 원본 비디오 → 세그먼트별 개별 PKL → 통합 STGCN PKL
- 중복 처리 제거, 효율적인 단일 패스 처리
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2
import psutil
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from enhanced_rtmo_bytetrack_pose_extraction import EnhancedRTMOPoseExtractor

# CUDA multiprocessing 설정 - spawn 방식 사용
mp.set_start_method('spawn', force=True)

class UnifiedPoseProcessor:
    """통합 포즈 처리기 - 비디오에서 최종 STGCN 데이터까지 원스톱"""
    
    def __init__(self, detector_config, detector_checkpoint, device='cuda:0', gpu_ids=[0], multi_gpu=False,
                 clip_len=100, num_person=5, save_overlay=True, overlay_fps=30,
                 # 포즈 추출 파라미터
                 score_thr=0.3, nms_thr=0.35, quality_threshold=0.3, min_track_length=10,
                 # ByteTracker 파라미터
                 track_high_thresh=0.6, track_low_thresh=0.1, track_max_disappeared=30, track_min_hits=3,
                 # 복합 점수 가중치
                 weights=None,
                 # 윈도우 처리 파라미터
                 min_success_rate=0.1):
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.device = device
        self.gpu_ids = gpu_ids if isinstance(gpu_ids, list) else [gpu_ids]
        self.multi_gpu = multi_gpu
        self.clip_len = clip_len
        self.num_person = num_person
        self.save_overlay = save_overlay
        self.overlay_fps = overlay_fps
        
        # 포즈 추출 파라미터
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.quality_threshold = quality_threshold
        self.min_track_length = min_track_length
        
        # ByteTracker 파라미터
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_max_disappeared = track_max_disappeared
        self.track_min_hits = track_min_hits
        
        # 복합 점수 가중치
        self.weights = weights if weights is not None else [0.30, 0.35, 0.20, 0.10, 0.05]
        
        # 윈도우 처리 파라미터
        self.min_success_rate = min_success_rate
        
        # 모델 초기화 (한 번만 로드)
        self.pose_model = None
        self.pose_extractor = None  # EnhancedRTMOPoseExtractor 인스턴스
        
        self._setup_gpu_environment()
        
        atexit.register(self._cleanup_multiprocessing_resources)
    
    def _initialize_pose_model(self):
        """포즈 모델을 한 번만 초기화"""
        if self.pose_model is None:
            try:
                from mmpose.apis import init_model
                print(f"Initializing pose model: {self.detector_config}")
                self.pose_model = init_model(self.detector_config, self.detector_checkpoint, device=self.device)
                print("Pose model initialized successfully")
            except Exception as e:
                print(f"Failed to initialize pose model: {str(e)}")
                raise e
        return self.pose_model
    
    def _initialize_pose_extractor(self):
        """포즈 추출기를 한 번만 초기화"""
        if self.pose_extractor is None:
            try:
                from enhanced_rtmo_bytetrack_pose_extraction import EnhancedRTMOPoseExtractor
                print(f"Initializing pose extractor: {self.detector_config}")
                self.pose_extractor = EnhancedRTMOPoseExtractor(
                    config_path=self.detector_config,
                    checkpoint_path=self.detector_checkpoint,
                    device=self.device,
                    score_thr=self.score_thr,
                    nms_thr=self.nms_thr
                )
                print("Pose extractor initialized successfully")
            except Exception as e:
                print(f"Failed to initialize pose extractor: {str(e)}")
                raise e
        return self.pose_extractor
    
    def _setup_gpu_environment(self):
        """초기 GPU 환경 설정"""
        try:
            import torch
            if torch.cuda.is_available() and len(self.gpu_ids) > 0:
                main_gpu = self.gpu_ids[0]
                if main_gpu < torch.cuda.device_count():
                    torch.cuda.set_device(main_gpu)
                else:
                    self.device = 'cuda:0'
                    self.gpu_ids = [0]
            else:
                self.device = 'cpu'
                self.gpu_ids = []
        except Exception as e:
            self.device = 'cpu'
            self.gpu_ids = []
    
    def _clear_gpu_cache(self):
        """GPU 메모리 캐시 정리"""
        try:
            import torch
            if torch.cuda.is_available() and 'cuda' in self.device:
                torch.cuda.empty_cache()
                # 필요시 synchronize 추가
                torch.cuda.synchronize()
        except Exception as e:
            # GPU 캐시 정리 실패는 치명적이지 않으므로 계속 진행
            pass
    
    def process_single_video_to_segments(self, video_path, output_dir, input_dir, training_stride=10):
        try:
            if not os.path.exists(video_path):
                return self._create_failure_analysis(
                    video_path,
                    "FILE_NOT_FOUND",
                    "Video file does not exist at specified path",
                    {"checked_path": video_path, "directory_exists": os.path.exists(os.path.dirname(video_path))}
                )
            
            # 모델과 추출기를 미리 초기화 (이후 모든 처리에서 재사용)
            if self.pose_model is None:
                print("Pre-initializing pose model for efficient processing...")
                self._initialize_pose_model()
            else:
                # GPU 메모리 정리 및 모델 상태 리셋
                self._clear_gpu_cache()
                print("Pose model already initialized, cleared GPU cache for next video")
                
            if self.pose_extractor is None:
                print("Pre-initializing pose extractor for efficient processing...")
                self._initialize_pose_extractor()
            else:
                print("Pose extractor already initialized, reusing existing extractor")
            
            all_pose_results = self._extract_full_video_poses(video_path, output_dir)
            
            if isinstance(all_pose_results, dict) and 'failure_stage' in all_pose_results:
                return all_pose_results
            elif not all_pose_results:
                return self._create_failure_analysis(
                    video_path, 
                    "POSE_EXTRACTION_FAILED", 
                    "No poses detected in video - video may not contain people or RTMO model failed",
                    {"stage": "pose_extraction", "poses_detected": 0}
                )
            
            if len(all_pose_results) < self.clip_len:
                all_pose_results = self._apply_temporal_padding(all_pose_results, self.clip_len)
            
            stride = training_stride
            windows_data = self._process_windows_with_tracking(all_pose_results, video_path, output_dir, stride)
            
            if isinstance(windows_data, dict) and 'failure_stage' in windows_data:
                return windows_data
            elif not windows_data:
                return self._create_failure_analysis(
                    video_path,
                    "WINDOW_PROCESSING_FAILED", 
                    "No valid windows generated - insufficient tracking data or processing error",
                    {"stage": "window_processing", "total_frames": len(all_pose_results), "windows_generated": 0}
                )
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label = 1 if '/Fight/' in video_path else 0
            label_folder = 'Fight' if label == 1 else 'Normal'
            dataset_name = os.path.basename(input_dir.rstrip('/\\'))
            
            video_result = {
                'video_name': video_name,
                'video_path': video_path,
                'label': label,
                'label_folder': label_folder,
                'total_frames': len(all_pose_results),
                'num_windows': len(windows_data),
                'windows': windows_data,
                'dataset_name': dataset_name
            }
            
            return video_result
            
        except Exception as e:
            import traceback
            video_name = os.path.basename(video_path)
            failure_analysis = self._analyze_video_failure(video_path, e, traceback.format_exc())
            print(f"Video processing failed: {video_name}")
            print(f"  Failure Stage: {failure_analysis.get('failure_stage', 'UNKNOWN')}")
            print(f"  Root Cause: {failure_analysis.get('root_cause', 'Unknown error')}")
            return failure_analysis
    
    def _extract_full_video_poses(self, video_path, output_dir):
        try:
            # 초기화된 추출기 재사용 (매번 새로 생성하지 않음)
            extractor = self._initialize_pose_extractor()
            
            pose_results = extractor.extract_poses_only(video_path)
            
            if not pose_results:
                return self._create_failure_analysis(
                    video_path,
                    "POSE_EXTRACTION_EMPTY",
                    "RTMO pose extraction returned empty results - no persons detected in video",
                    {
                        "stage": "pose_extraction",
                        "extractor_device": self.device,
                        "poses_detected": 0,
                        "video_accessible": True
                    }
                )
                
            return pose_results
            
        except Exception as e:
            import traceback
            return self._create_failure_analysis(
                video_path,
                "POSE_EXTRACTION_EXCEPTION",
                f"Exception during pose extraction: {str(e)}",
                {
                    "stage": "pose_extraction",
                    "extractor_device": self.device,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _process_windows_with_tracking(self, all_pose_results, video_path, output_dir, stride=10):
        try:
            total_frames = len(all_pose_results)
            windows_data = []
            failed_windows = []
            total_windows = len(range(0, total_frames - self.clip_len + 1, stride))
            
            print(f"Processing {total_windows} windows for {total_frames} frames (stride={stride})")
            
            for window_idx, start_frame in enumerate(range(0, total_frames - self.clip_len + 1, stride)):
                end_frame = min(start_frame + self.clip_len, total_frames)
                window_pose_results = all_pose_results[start_frame:end_frame]
                
                window_data = self._process_single_window(
                    window_pose_results, 
                    window_idx, 
                    start_frame, 
                    end_frame,
                    video_path,
                    output_dir
                )
                
                if window_data:
                    if isinstance(window_data, dict) and 'failure_stage' in window_data:
                        # 개별 윈도우 실패를 기록하되 전체 처리는 계속
                        failed_windows.append({
                            'window_idx': window_idx,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'failure_reason': window_data.get('failure_reason', 'Unknown'),
                            'failure_stage': window_data.get('failure_stage', 'Unknown')
                        })
                        print(f"Window {window_idx} ({start_frame}-{end_frame}) failed: {window_data.get('failure_reason', 'Unknown')}")
                    else:
                        windows_data.append(window_data)
                        print(f"Window {window_idx} ({start_frame}-{end_frame}) processed successfully")
            
            # 결과 요약 및 상태 결정
            success_rate = len(windows_data) / total_windows if total_windows > 0 else 0
            print(f"Window processing completed: {len(windows_data)}/{total_windows} successful ({success_rate:.1%})")
            
            if len(failed_windows) > 0:
                print(f"Failed windows summary:")
                for failed in failed_windows:
                    print(f"  - Window {failed['window_idx']}: {failed['failure_reason']}")
            
            # 성공률이 최소 기준을 만족하는지 확인
            if success_rate < self.min_success_rate:
                return self._create_failure_analysis(
                    video_path,
                    "INSUFFICIENT_SUCCESS_RATE",
                    f"Success rate {success_rate:.1%} is below minimum threshold {self.min_success_rate:.1%}",
                    {
                        "stage": "window_processing",
                        "total_windows_attempted": total_windows,
                        "valid_windows_generated": len(windows_data),
                        "failed_windows_count": len(failed_windows),
                        "success_rate": success_rate,
                        "min_success_rate": self.min_success_rate,
                        "failed_windows_details": failed_windows
                    }
                )
            
            # 성공한 윈도우가 하나도 없는 경우
            if len(windows_data) == 0:
                return self._create_failure_analysis(
                    video_path,
                    "NO_VALID_WINDOWS",
                    f"No valid windows could be processed - all {total_windows} windows failed",
                    {
                        "stage": "window_processing",
                        "total_windows_attempted": total_windows,
                        "valid_windows_generated": 0,
                        "failed_windows_count": len(failed_windows),
                        "failed_windows_details": failed_windows
                    }
                )
            
            # 성공률은 만족하지만 일부 윈도우가 실패한 경우
            if len(failed_windows) > 0:
                print(f"Warning: {len(failed_windows)} windows failed but success rate {success_rate:.1%} meets minimum threshold {self.min_success_rate:.1%}")
            
            return windows_data
            
        except Exception as e:
            import traceback
            return self._create_failure_analysis(
                video_path,
                "WINDOW_PROCESSING_EXCEPTION",
                f"Exception during window processing: {str(e)}",
                {
                    "stage": "window_processing",
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _process_single_window(self, window_pose_results, window_idx, start_frame, end_frame, video_path, output_dir):
        try:
            from enhanced_rtmo_bytetrack_pose_extraction import (
                ByteTracker, create_detection_results, assign_track_ids_from_bytetrack,
                create_enhanced_annotation
            )
            
            tracker = ByteTracker(
                high_thresh=self.track_high_thresh,
                low_thresh=self.track_low_thresh,
                max_disappeared=self.track_max_disappeared,
                min_hits=self.track_min_hits
            )
            
            tracked_pose_results = []
            for pose_result in window_pose_results:
                detections = create_detection_results(pose_result)
                active_tracks = tracker.update(detections)
                tracked_result = assign_track_ids_from_bytetrack(pose_result, active_tracks)
                tracked_pose_results.append(tracked_result)
            
            # 초기화된 모델 사용 (매번 로드하지 않음)
            pose_model = self._initialize_pose_model()
            
            annotation, status_message = create_enhanced_annotation(
                tracked_pose_results, video_path, pose_model,
                min_track_length=self.min_track_length,
                quality_threshold=self.quality_threshold,
                weights=self.weights
            )
            
            if annotation is None:
                return self._create_failure_analysis(
                    video_path,
                    "ANNOTATION_FAILED",
                    f"Annotation creation failed for window {window_idx}: {status_message}",
                    {
                        "stage": "annotation_creation",
                        "window_index": window_idx,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "annotation_error": status_message,
                        "tracked_results_count": len(tracked_pose_results) if tracked_pose_results else 0
                    }
                )
            
            segment_video_path = None
            if self.save_overlay:
                segment_video_path = self._create_window_segment_video(
                    video_path, output_dir, window_idx, start_frame, end_frame, 
                    tracked_pose_results, pose_model, annotation
                )
            
            window_data = {
                'window_idx': window_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'num_frames': end_frame - start_frame,
                'annotation': annotation,
                'segment_video_path': segment_video_path,
                'persons_ranking': self._extract_persons_ranking(annotation)
            }
            
            return window_data
            
        except Exception as e:
            import traceback
            return self._create_failure_analysis(
                video_path,
                "WINDOW_PROCESSING_EXCEPTION",
                f"Exception in window {window_idx} processing: {str(e)}",
                {
                    "stage": "single_window_processing",
                    "window_index": window_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _create_window_segment_video(self, video_path, output_dir, window_idx, start_frame, end_frame, 
                                   tracked_pose_results, pose_model, annotation):
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label = 1 if '/Fight/' in video_path else 0
            label_folder = 'Fight' if label == 1 else 'Normal'
            
            # input_dir 구조에서 dataset_name과 split_folder 추출
            path_parts = video_path.replace('\\', '/').split('/')
            dataset_name = None
            split_folder = None  # train, val, test 등
            
            # Fight, Normal 폴더를 찾고, 그 상위와 상위의 상위 폴더 확인
            for i, part in enumerate(path_parts):
                if part in ['Fight', 'Normal'] and i >= 2:
                    split_folder = path_parts[i-1]  # train, val 등
                    dataset_name = path_parts[i-2]  # RWF-2001 등
                    break
            
            if dataset_name is None or split_folder is None:
                # fallback: 기본값 사용
                dataset_name = 'RWF-2001'
                split_folder = 'train'  # 기본값
            
            # input-dir 구조를 따른 temp 폴더: output_dir/dataset_name/temp/split_folder/label_folder/video_name/
            temp_output_dir = os.path.join(output_dir, dataset_name, 'temp', split_folder, label_folder, video_name)
            os.makedirs(temp_output_dir, exist_ok=True)
            segment_video_path = os.path.join(temp_output_dir, f"{video_name}_{window_idx}.mp4")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(segment_video_path, fourcc, fps, (width, height), True)
            
            track_id_to_rank = {}
            if 'persons' in annotation and annotation['persons']:
                for person_data in annotation['persons'].values():
                    track_id = person_data.get('track_id')
                    rank = person_data.get('rank')
                    if track_id is not None and rank is not None:
                        track_id_to_rank[track_id] = rank
            
            from mmpose.registry import VISUALIZERS
            visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
            visualizer.set_dataset_meta(pose_model.dataset_meta)
            
            for frame_idx, pose_result in enumerate(tracked_pose_results):
                success, frame = cap.read()
                if not success: break
                
                try:
                    visualizer.add_datasample('result', frame, data_sample=pose_result, draw_gt=False, draw_heatmap=False, draw_bbox=False, show_kpt_idx=False, skeleton_style='mmpose')
                    vis_frame = visualizer.get_image()
                    if vis_frame.shape[:2] != (height, width):
                        vis_frame = cv2.resize(vis_frame, (width, height))
                    
                    from enhanced_rtmo_bytetrack_pose_extraction import draw_track_ids
                    vis_frame = draw_track_ids(vis_frame, pose_result, track_id_to_rank, None)
                    out_writer.write(vis_frame)
                except Exception as e:
                    out_writer.write(frame)
            
            cap.release()
            out_writer.release()
            return segment_video_path
        except Exception as e:
            return None
    
    def _extract_persons_ranking(self, annotation):
        try:
            if 'persons' not in annotation or not annotation['persons']: return []
            rankings = []
            for person_key, person_data in annotation['persons'].items():
                rankings.append({
                    'person_key': person_key,
                    'track_id': person_data.get('track_id'),
                    'rank': person_data.get('rank'),
                    'composite_score': person_data.get('composite_score', 0.0),
                    'score_breakdown': person_data.get('score_breakdown', {})
                })
            rankings.sort(key=lambda x: x['rank'])
            return rankings
        except Exception as e:
            return []
    
    def _apply_temporal_padding(self, pose_results, target_length):
        try:
            if len(pose_results) >= target_length: return pose_results
            current_length = len(pose_results)
            needed_frames = target_length - current_length
            if current_length > 0:
                last_frame = pose_results[-1]
                padded_results = pose_results.copy()
                for _ in range(needed_frames):
                    padded_results.append(last_frame)
                return padded_results
            else:
                return pose_results
        except Exception as e:
            return pose_results

    def process_batch_videos(self, video_list, output_dir, input_dir, training_stride=10, max_workers=2):
        successful_videos_data = []
        failed_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {executor.submit(self.process_single_video_to_segments, video, output_dir, input_dir, training_stride): video for video in video_list}
            for future in tqdm(as_completed(future_to_video), total=len(video_list), desc="Processing videos"):
                video = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        successful_videos_data.append(result)
                    else:
                        failed_count += 1
                except Exception as exc:
                    failed_count += 1
        return successful_videos_data
    
    def create_unified_stgcn_data(self, video_results_list, output_dir, input_dir, train_split=0.7, val_split=0.2):
        dataset_name = os.path.basename(input_dir.rstrip('/\\'))
        
        print(f"Creating unified STGCN data for dataset: {dataset_name}")
        print(f"Step 1: Saving individual video results to temp folder...")
        
        # Step 1: 모든 개별 비디오 결과를 temp 폴더에 저장
        all_stgcn_samples = []
        for video_result in video_results_list:
            video_name = video_result['video_name']
            label = video_result['label']
            label_folder = video_result['label_folder']
            
            # temp 폴더에 개별 비디오 PKL 저장
            self._save_video_pkl(video_result, output_dir, dataset_name)
            
            # STGCN 샘플 생성 (메모리에 유지)
            for window_data in video_result['windows']:
                stgcn_sample = self._convert_window_to_stgcn_format(window_data, video_name, label, label_folder)
                if stgcn_sample:
                    all_stgcn_samples.append(stgcn_sample)
        
        print(f"Step 2: Splitting data into train/val/test sets...")
        # Step 2: 모든 추론 완료 후 train/val/test 분할
        train_segments, val_segments, test_segments = self._split_samples_by_video(
            all_stgcn_samples, video_results_list, train_split, val_split
        )
        
        print(f"Step 3: Moving temp files to final train/val/test structure...")
        # Step 3: temp에서 최종 train/val/test 구조로 파일 이동
        self._save_split_data_new_structure(
            train_segments, val_segments, test_segments, output_dir, dataset_name, video_results_list
        )
        
        print(f"Step 4: Creating unified PKL files...")
        # Step 4: 통합 PKL 파일 생성
        self._save_unified_pkl_files(train_segments, val_segments, test_segments, output_dir, dataset_name)
        
        print(f"Step 5: Cleaning up temp folders...")
        # Step 5: temp 폴더 정리
        self._cleanup_temp_folder(output_dir, dataset_name)
        
        return len(train_segments), len(val_segments), len(test_segments)
    
    def _save_video_pkl(self, video_result, output_dir, dataset_name):
        try:
            video_name = video_result['video_name'] 
            label_folder = video_result['label_folder']
            video_path = video_result['video_path']
            
            # input_dir 구조에서 split_folder 추출 (train, val 등)
            path_parts = video_path.replace('\\', '/').split('/')
            split_folder = None
            
            for i, part in enumerate(path_parts):
                if part in ['Fight', 'Normal'] and i >= 1:
                    split_folder = path_parts[i-1]  # train, val 등
                    break
            
            if split_folder is None:
                split_folder = 'train'  # 기본값
            
            # input-dir 구조를 따른 temp 폴더: output_dir/dataset_name/temp/split_folder/label_folder/video_name/
            video_pkl_dir = os.path.join(output_dir, dataset_name, 'temp', split_folder, label_folder, video_name)
            os.makedirs(video_pkl_dir, exist_ok=True)
            video_pkl_path = os.path.join(video_pkl_dir, f"{video_name}_windows.pkl")
            
            with open(video_pkl_path, 'wb') as f:
                pickle.dump(video_result, f)
            print(f"Saved video PKL to temp: {video_pkl_path}")
        except Exception as e:
            print(f"Error saving video PKL to temp: {str(e)}")
    
    def _convert_window_to_stgcn_format(self, window_data, video_name, label, label_folder):
        try:
            annotation = window_data['annotation']
            if 'persons' not in annotation or not annotation['persons']: return None
            
            sorted_persons = sorted(annotation['persons'].items(), key=lambda x: x[1]['rank'])
            all_keypoints = []
            all_scores = []
            for person_key, person_data in sorted_persons:
                person_annotation = person_data['annotation']
                keypoints = person_annotation['keypoint']
                scores = person_annotation['keypoint_score']
                if keypoints.ndim == 4: keypoints = keypoints.squeeze(0)
                if scores.ndim == 3: scores = scores.squeeze(0)
                all_keypoints.append(keypoints)
                all_scores.append(scores)
            
            final_keypoints = np.array(all_keypoints)
            final_scores = np.array(all_scores)
            
            stgcn_sample = {
                'frame_dir': f"{video_name}_window_{window_data['window_idx']:03d}",
                'total_frames': window_data['num_frames'],
                'img_shape': annotation['video_info']['img_shape'],
                'original_shape': annotation['video_info']['original_shape'],
                'label': label,
                'label_folder': label_folder,
                'keypoint': final_keypoints,
                'keypoint_score': final_scores,
                'window_info': {
                    'video_name': video_name,
                    'window_idx': window_data['window_idx'],
                    'start_frame': window_data['start_frame'],
                    'end_frame': window_data['end_frame'],
                    'persons_ranking': window_data['persons_ranking'],
                    'segment_video_path': window_data.get('segment_video_path')
                }
            }
            return stgcn_sample
        except Exception as e:
            return None
    
    def _split_samples_by_video(self, all_samples, video_results_list, train_split, val_split):
        try:
            fight_video_groups = defaultdict(list)
            normal_video_groups = defaultdict(list)
            for video_result in video_results_list:
                video_name = video_result['video_name']
                label = video_result['label']
                group_key = video_name.split('_')[0] if '_' in video_name else video_name
                if label == 1: fight_video_groups[group_key].append(video_name)
                else: normal_video_groups[group_key].append(video_name)

            def split_video_groups(video_groups, train_ratio, val_ratio):
                group_keys = list(video_groups.keys())
                np.random.seed(42)
                np.random.shuffle(group_keys)
                total_groups = len(group_keys)
                if total_groups < 3:
                    train_videos = []
                    for group_key in group_keys:
                        train_videos.extend(video_groups[group_key])
                    return train_videos, [], []
                
                train_size = int(total_groups * train_ratio)
                val_size = int(total_groups * val_ratio)
                train_group_keys = group_keys[:train_size]
                val_group_keys = group_keys[train_size:train_size + val_size]
                test_group_keys = group_keys[train_size + val_size:]
                
                train_videos = [v for gk in train_group_keys for v in video_groups[gk]]
                val_videos = [v for gk in val_group_keys for v in video_groups[gk]]
                test_videos = [v for gk in test_group_keys for v in video_groups[gk]]
                return train_videos, val_videos, test_videos

            fight_train_videos, fight_val_videos, fight_test_videos = split_video_groups(fight_video_groups, train_split, val_split)
            normal_train_videos, normal_val_videos, normal_test_videos = split_video_groups(normal_video_groups, train_split, val_split)

            all_train_videos = fight_train_videos + normal_train_videos
            all_val_videos = fight_val_videos + normal_val_videos
            all_test_videos = fight_test_videos + normal_test_videos

            train_segments = [s for s in all_samples if s['window_info']['video_name'] in all_train_videos]
            val_segments = [s for s in all_samples if s['window_info']['video_name'] in all_val_videos]
            test_segments = [s for s in all_samples if s['window_info']['video_name'] in all_test_videos]

            np.random.shuffle(train_segments)
            np.random.shuffle(val_segments)
            np.random.shuffle(test_segments)
            return train_segments, val_segments, test_segments
        except Exception as e:
            return [], [], []
    
    def _save_split_data_new_structure(self, train_segments, val_segments, test_segments, output_dir, dataset_name, video_results_list):
        try:
            base_output_dir = os.path.join(output_dir, dataset_name)
            splits_data = {'train': train_segments, 'val': val_segments, 'test': test_segments}
            for split_name, segments in splits_data.items():
                if not segments: continue
                fight_segments = [s for s in segments if s['label'] == 1]
                normal_segments = [s for s in segments if s['label'] == 0]
                if fight_segments:
                    fight_dir = os.path.join(output_dir, dataset_name, split_name, 'Fight')
                    os.makedirs(fight_dir, exist_ok=True)
                    self._move_segment_files_to_split(fight_segments, fight_dir, 'Fight', video_results_list)
                if normal_segments:
                    normal_dir = os.path.join(output_dir, dataset_name, split_name, 'Normal')
                    os.makedirs(normal_dir, exist_ok=True)
                    self._move_segment_files_to_split(normal_segments, normal_dir, 'Normal', video_results_list)
        except Exception as e:
            print(f"Error saving split data: {str(e)}")
    
    def _move_segment_files_to_split(self, segments, target_dir, label_folder, video_results_list):
        try:
            import shutil
            video_groups = defaultdict(list)
            for segment in segments:
                video_groups[segment['window_info']['video_name']].append(segment)
            
            print(f"Moving {len(video_groups)} videos from temp to {os.path.basename(target_dir)}/{label_folder}/")
            
            for video_name in video_groups.keys():
                try:
                    # 최종 비디오 폴더 생성
                    video_target_dir = os.path.join(target_dir, video_name)
                    os.makedirs(video_target_dir, exist_ok=True)
                    
                    # 해당 비디오 결과 찾기
                    video_result = next((vr for vr in video_results_list if vr['video_name'] == video_name and vr['label_folder'] == label_folder), None)
                    if not video_result: 
                        print(f"Warning: Video result not found for {video_name}")
                        continue
                    
                    # video_path에서 원래 split_folder 추출 (train, val 등)
                    video_path = video_result['video_path']
                    path_parts = video_path.replace('\\', '/').split('/')
                    original_split_folder = None
                    
                    for i, part in enumerate(path_parts):
                        if part in ['Fight', 'Normal'] and i >= 1:
                            original_split_folder = path_parts[i-1]  # train, val 등
                            break
                    
                    if original_split_folder is None:
                        original_split_folder = 'train'  # 기본값
                    
                    # temp 폴더 경로: dataset_name/temp/original_split_folder/label_folder/video_name/
                    dataset_output_dir = os.path.dirname(os.path.dirname(target_dir))  # train/val/test의 상위 폴더
                    temp_video_dir = os.path.join(dataset_output_dir, 'temp', original_split_folder, label_folder, video_name)
                    
                    if not os.path.exists(temp_video_dir): 
                        print(f"Warning: Temp directory not found: {temp_video_dir}")
                        continue
                    
                    # PKL 파일 이동
                    temp_pkl_path = os.path.join(temp_video_dir, f"{video_name}_windows.pkl")
                    final_pkl_path = os.path.join(video_target_dir, f"{video_name}_windows.pkl")
                    if os.path.exists(temp_pkl_path):
                        shutil.move(temp_pkl_path, final_pkl_path)
                        print(f"Moved PKL: {temp_pkl_path} -> {final_pkl_path}")
                    
                    # 세그먼트 비디오 파일들 이동
                    moved_count = 0
                    for window_data in video_result['windows']:
                        segment_video_path = window_data.get('segment_video_path')
                        if segment_video_path and os.path.exists(segment_video_path):
                            target_segment_path = os.path.join(video_target_dir, os.path.basename(segment_video_path))
                            shutil.move(segment_video_path, target_segment_path)
                            moved_count += 1
                    
                    print(f"Moved {moved_count} segment videos for {video_name}")
                    
                    # 빈 temp 폴더 제거
                    if os.path.exists(temp_video_dir) and not os.listdir(temp_video_dir):
                        os.rmdir(temp_video_dir)
                        print(f"Removed empty temp directory: {temp_video_dir}")
                        
                except Exception as video_error:
                    print(f"Error moving files for video {video_name}: {str(video_error)}")
                    continue
                    
        except Exception as e:
            print(f"Error in _move_segment_files_to_split: {str(e)}")
    
    def _save_unified_pkl_files(self, train_segments, val_segments, test_segments, output_dir, dataset_name):
        try:
            base_output_dir = os.path.join(output_dir, dataset_name)
            splits_data = {'train': train_segments, 'val': val_segments, 'test': test_segments}
            for split_name, segments in splits_data.items():
                if not segments: continue
                pkl_path = os.path.join(base_output_dir, f"{dataset_name}_{split_name}_windows.pkl")
                with open(pkl_path, 'wb') as f:
                    pickle.dump(segments, f)
        except Exception as e:
            print(f"Error saving unified PKL files: {str(e)}")
    
    def _cleanup_temp_folder(self, output_dir, dataset_name):
        try:
            import shutil
            
            # temp 폴더 경로: output_dir/dataset_name/temp/
            temp_folder_path = os.path.join(output_dir, dataset_name, 'temp')
            
            if os.path.exists(temp_folder_path):
                print(f"Cleaning up temp folder: {temp_folder_path}")
                
                # temp 폴더 내용 확인
                remaining_files = []
                try:
                    for root, dirs, files in os.walk(temp_folder_path):
                        for file in files:
                            remaining_files.append(os.path.join(root, file))
                except Exception:
                    pass
                
                if remaining_files:
                    print(f"Warning: {len(remaining_files)} files still remain in temp folder")
                    for file in remaining_files[:5]:  # 처음 5개만 출력
                        print(f"  - {file}")
                    if len(remaining_files) > 5:
                        print(f"  ... and {len(remaining_files) - 5} more files")
                
                # temp 폴더 제거
                shutil.rmtree(temp_folder_path, ignore_errors=True)
                print(f"Successfully cleaned up temp folder")
            else:
                print(f"Temp folder not found: {temp_folder_path}")
                
        except Exception as e:
            print(f"ERROR: Failed to cleanup temp folders: {e}")
    
    def _force_remove_directory(self, dir_path):
        try:
            import stat, shutil
            def handle_remove_readonly(func, path, exc):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception: pass
            shutil.rmtree(dir_path, onerror=handle_remove_readonly)
        except Exception: pass
    
    def _analyze_video_failure(self, video_path, exception, full_traceback):
        video_name = os.path.basename(video_path)
        failure_stage, root_cause, detailed_info = "UNKNOWN", "Unknown error", {}
        try:
            if not os.path.exists(video_path):
                failure_stage, root_cause = "FILE_NOT_FOUND", "Video file does not exist"
            else:
                file_size = os.path.getsize(video_path)
                if file_size == 0:
                    failure_stage, root_cause = "EMPTY_FILE", "Video file is empty"
                else:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        failure_stage, root_cause = "CODEC_ERROR", "Cannot decode video"
                    else:
                        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
                            failure_stage, root_cause = "NO_FRAMES", "Video has no frames"
                        else:
                            failure_stage, root_cause = self._analyze_processing_failure(exception, full_traceback)
                        cap.release()
        except Exception as analysis_error:
            detailed_info['analysis_error'] = str(analysis_error)
        
        solution_map = {
            'FILE_NOT_FOUND': 'Check file path',
            'EMPTY_FILE': 'Re-download or restore file',
            'CODEC_ERROR': 'Convert video to MP4/H264',
            'NO_FRAMES': 'Check video integrity',
            'GPU_ERROR': 'Check GPU/CUDA setup',
            'MEMORY_ERROR': 'Free up memory',
            'POSE_EXTRACTION_FAILED': 'Check model and video content',
            'TRACKING_FAILED': 'Adjust tracking parameters',
            'WINDOW_PROCESSING_FAILED': 'Check track length and annotation requirements'
        }
        return {
            'video_path': video_path, 'video_name': video_name, 'failure_stage': failure_stage,
            'root_cause': root_cause, 'suggested_solution': solution_map.get(failure_stage, 'Review logs'),
            'detailed_info': detailed_info, 'error_type': type(exception).__name__,
            'error_message': str(exception), 'full_traceback': full_traceback,
            'timestamp': self._get_current_timestamp()
        }
    
    def _analyze_processing_failure(self, exception, full_traceback):
        traceback_lower = full_traceback.lower()
        if '_extract_full_video_poses' in traceback_lower: return "POSE_EXTRACTION_FAILED", "RTMO pose detection failed"
        if '_process_windows_with_tracking' in traceback_lower: return "WINDOW_PROCESSING_FAILED", "Window tracking failed"
        if 'bytetrack' in traceback_lower: return "TRACKING_FAILED", "ByteTracker algorithm failed"
        if 'annotation' in traceback_lower: return "ANNOTATION_FAILED", "Annotation creation failed"
        if 'segment_video' in traceback_lower: return "SEGMENT_VIDEO_FAILED", "Segment video creation failed"
        return "PROCESSING_ERROR", f"Processing failed with {type(exception).__name__}"
    
    def _get_current_timestamp(self):
        try: return datetime.now().isoformat()
        except: return None
    
    def _cleanup_multiprocessing_resources(self):
        try:
            # 모델 정리
            if hasattr(self, 'pose_model') and self.pose_model is not None:
                try:
                    import torch
                    if hasattr(self.pose_model, 'cpu'):
                        self.pose_model.cpu()
                    del self.pose_model
                    self.pose_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Pose model resources cleaned up")
                except Exception as e:
                    print(f"Warning: Failed to cleanup pose model: {str(e)}")
            
            # 추출기 정리
            if hasattr(self, 'pose_extractor') and self.pose_extractor is not None:
                try:
                    del self.pose_extractor
                    self.pose_extractor = None
                    print("Pose extractor resources cleaned up")
                except Exception as e:
                    print(f"Warning: Failed to cleanup pose extractor: {str(e)}")
            
            # 멀티프로세싱 정리
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            for obj in mp.active_children():
                obj.terminate()
                obj.join(timeout=1)
        except Exception: pass
    
    def merge_existing_pkl_files(self, processed_data_dir, output_dir, train_split=0.7, val_split=0.2):
        try:
            dataset_name = os.path.basename(processed_data_dir.rstrip('/\\'))
            temp_dir = os.path.join(processed_data_dir, 'temp')
            if not os.path.exists(temp_dir): return 0, 0, 0
            
            video_results_list, all_stgcn_samples = [], []
            
            # input-dir 구조를 따른 temp 폴더 스캔: temp/split_folder/label_folder/video_name/
            for split_folder in os.listdir(temp_dir):
                split_dir = os.path.join(temp_dir, split_folder)
                if not os.path.isdir(split_dir): continue
                
                for label_folder in ['Fight', 'Normal']:
                    label_dir = os.path.join(split_dir, label_folder)
                    if not os.path.exists(label_dir): continue
                    label = 1 if label_folder == 'Fight' else 0
                    
                    for video_name in os.listdir(label_dir):
                        video_dir = os.path.join(label_dir, video_name)
                        if not os.path.isdir(video_dir): continue
                        pkl_file = os.path.join(video_dir, f"{video_name}_windows.pkl")
                        if not os.path.exists(pkl_file): continue
                        try:
                            with open(pkl_file, 'rb') as f:
                                video_result = pickle.load(f)
                            video_result.setdefault('video_name', video_name)
                            video_result.setdefault('label', label)
                            video_result.setdefault('label_folder', label_folder)
                            video_result.setdefault('dataset_name', dataset_name)
                            video_results_list.append(video_result)
                            if 'windows' in video_result:
                                for window_data in video_result['windows']:
                                    stgcn_sample = self._convert_window_to_stgcn_format(window_data, video_name, label, label_folder)
                                    if stgcn_sample: all_stgcn_samples.append(stgcn_sample)
                        except Exception as e: continue
            
            if not video_results_list: return 0, 0, 0
            
            train_segments, val_segments, test_segments = self._split_samples_by_video(all_stgcn_samples, video_results_list, train_split, val_split)
            self._save_split_data_new_structure(train_segments, val_segments, test_segments, output_dir, dataset_name, video_results_list)
            self._save_unified_pkl_files(train_segments, val_segments, test_segments, output_dir, dataset_name)
            self._cleanup_temp_folder(output_dir, dataset_name)
            return len(train_segments), len(val_segments), len(test_segments)
        except Exception as e:
            return 0, 0, 0
    
    def _create_failure_analysis(self, video_path, failure_stage, root_cause, detailed_info=None):
        video_name = os.path.basename(video_path)
        solution_map = {
            'POSE_EXTRACTION_FAILED': 'Check if video contains people, verify RTMO model and GPU availability',
            'POSE_EXTRACTION_EMPTY': 'Video may not contain people or may be too dark/blurry for detection',
            'POSE_EXTRACTION_EXCEPTION': 'Check GPU memory, RTMO model files, and video codec compatibility',
            'WINDOW_PROCESSING_FAILED': 'Verify tracking parameters and minimum track length requirements',
            'WINDOW_PROCESSING_EXCEPTION': 'Check memory usage and ByteTracker parameters',
            'NO_VALID_WINDOWS': 'All windows failed annotation - reduce quality threshold or min_track_length', 
            'ANNOTATION_FAILED': 'Insufficient tracked poses - adjust tracking parameters or quality threshold',
            'TRACKING_FAILED': 'Adjust ByteTracker parameters or improve pose detection quality',
            'FILE_NOT_FOUND': 'Check file path and ensure video file exists',
            'CODEC_ERROR': 'Convert video to MP4/H264 format'
        }
        return {
            'video_path': video_path, 'video_name': video_name, 'failure_stage': failure_stage,
            'root_cause': root_cause, 'suggested_solution': solution_map.get(failure_stage, 'Review error logs for specific details'),
            'detailed_info': detailed_info or {},
            'error_type': failure_stage, 'error_message': root_cause,
            'timestamp': self._get_current_timestamp(), 'full_traceback': None
        }