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
from enhanced_rtmo_bytetrack_pose_extraction import EnhancedRTMOPoseExtractor, FailureLogger

# CUDA multiprocessing 설정 - spawn 방식 사용
mp.set_start_method('spawn', force=True)

class UnifiedPoseProcessor:
    """통합 포즈 처리기 - 비디오에서 최종 STGCN 데이터까지 원스톱"""
    
    def __init__(self, detector_config, detector_checkpoint, device='cuda:0', gpu_ids=[0], multi_gpu=False,
                 clip_len=100, num_person=5, save_overlay=True, overlay_fps=30):
        """
        Args:
            detector_config: RTMO 검출기 설정 파일
            detector_checkpoint: RTMO 검출기 체크포인트 (PTH 파일)
            device: 메인 GPU 디바이스 (cuda:0, cuda:1, cpu)
            gpu_ids: 사용할 GPU ID 목록 [0, 1]
            multi_gpu: 멀티 GPU 사용 여부
            clip_len: 세그먼트 길이 (프레임)
            num_person: 오버레이 표시할 최대 인물 수 (모든 인물은 저장됨)
            save_overlay: 오버레이 비디오 저장 여부
            overlay_fps: 오버레이 비디오 FPS
        """
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.device = device
        self.gpu_ids = gpu_ids if isinstance(gpu_ids, list) else [gpu_ids]
        self.multi_gpu = multi_gpu
        self.clip_len = clip_len
        self.num_person = num_person
        self.save_overlay = save_overlay
        self.overlay_fps = overlay_fps
        
        # GPU 설정 초기화
        self._setup_gpu_environment()
        
        print(f"Initialized UnifiedPoseProcessor:")
        print(f"  Device: {self.device}")
        print(f"  GPU IDs: {self.gpu_ids}")
        print(f"  Multi-GPU: {self.multi_gpu}")
        
        # 멀티프로세싱 리소스 정리
        atexit.register(self._cleanup_multiprocessing_resources)
    
    def _setup_gpu_environment(self):
        """초기 GPU 환경 설정"""
        try:
            import torch
            if torch.cuda.is_available() and len(self.gpu_ids) > 0:
                # 메인 GPU 설정
                main_gpu = self.gpu_ids[0]
                if main_gpu < torch.cuda.device_count():
                    torch.cuda.set_device(main_gpu)
                    print(f"Set main GPU to {main_gpu}: {torch.cuda.get_device_name(main_gpu)}")
                    
                    # 멀티 GPU 환경인 경우 사용 가능한 GPU 목록 표시
                    if self.multi_gpu and len(self.gpu_ids) > 1:
                        print(f"Available GPUs for processing:")
                        for gpu_id in self.gpu_ids:
                            if gpu_id < torch.cuda.device_count():
                                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                else:
                    print(f"Warning: Main GPU {main_gpu} not available, using GPU 0")
                    self.device = 'cuda:0'
                    self.gpu_ids = [0]
            else:
                print("Warning: CUDA not available or no GPUs specified, using CPU")
                self.device = 'cpu'
                self.gpu_ids = []
        except Exception as e:
            print(f"GPU setup error: {e}, using CPU")
            self.device = 'cpu'
            self.gpu_ids = []
    
    def process_single_video_to_segments(self, video_path, output_dir, input_dir, training_stride=10, inference_stride=50):
        """
        단일 비디오를 윈도우 기반으로 처리하여 세그먼트별 PKL 생성
        
        Returns:
            dict: {
                'windows': [...],  # 윈도우별 처리 결과
                'video_info': {...}
            }
        """
        try:
            print(f"Processing video: {os.path.basename(video_path)}")
            
            # 1단계: 전체 비디오에 대한 포즈 추정 (윈도우별 트래킹은 별도)
            all_pose_results = self._extract_full_video_poses(video_path, output_dir)
            if not all_pose_results:
                print(f"Failed to extract poses from {video_path}")
                return None
            
            # 2단계: 윈도우별 ByteTrack + 점수 계산
            windows_data = self._process_windows_with_tracking(all_pose_results, video_path, output_dir)
            if not windows_data:
                print(f"Failed to process windows for {video_path}")
                return None
            
            # 3단계: 비디오별 PKL 파일 생성 및 저장
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 라벨 결정 (Fight/Normal)
            label = 1 if '/Fight/' in video_path else 0
            label_folder = 'Fight' if label == 1 else 'Normal'
            
            # 데이터셋명
            dataset_name = os.path.basename(input_dir.rstrip('/\\'))
            
            # 출력 구조: output/{dataset}/train|val|test/{Fight|Normal}/{video_name}/
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
            
            print(f"Generated {len(windows_data)} windows for {video_name}")
            
            return video_result
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
    
    def _extract_full_video_poses(self, video_path, output_dir):
        """전체 비디오에 대한 포즈 추정 수행 (트래킹 제외)"""
        try:
            # 실패 로거 초기화
            failure_log_path = os.path.join(output_dir, 'enhanced_failed_videos.txt')
            failure_logger = FailureLogger(failure_log_path)
            
            # Enhanced RTMO 포즈 추출기 초기화
            extractor = EnhancedRTMOPoseExtractor(
                config_path=self.detector_config,
                checkpoint_path=self.detector_checkpoint,
                device='cuda:0',
                save_overlay=False,  # 전체 비디오 처리시에는 오버레이 생성 안함
                num_person=self.num_person,
                overlay_fps=self.overlay_fps
            )
            
            # 전체 비디오에 대한 포즈 추정만 수행
            pose_results = extractor.extract_poses_only(video_path, failure_logger)
            
            return pose_results
            
        except Exception as e:
            print(f"Exception in pose extraction: {str(e)}")
            return None
    
    def _process_windows_with_tracking(self, all_pose_results, video_path, output_dir):
        """윈도우별로 ByteTrack + 점수 계산 수행"""
        try:
            total_frames = len(all_pose_results)
            windows_data = []
            
            # 윈도우 생성
            for window_idx, start_frame in enumerate(range(0, total_frames - self.clip_len + 1, self.clip_len)):
                end_frame = min(start_frame + self.clip_len, total_frames)
                
                # 윈도우 범위의 포즈 결과 추출
                window_pose_results = all_pose_results[start_frame:end_frame]
                
                # 윈도우별 트래킹 + 점수 계산
                window_data = self._process_single_window(
                    window_pose_results, 
                    window_idx, 
                    start_frame, 
                    end_frame,
                    video_path,
                    output_dir
                )
                
                if window_data:
                    windows_data.append(window_data)
            
            return windows_data
            
        except Exception as e:
            print(f"Error processing windows: {str(e)}")
            return None
    
    def _process_single_window(self, window_pose_results, window_idx, start_frame, end_frame, video_path, output_dir):
        """단일 윈도우 처리: ByteTrack + 점수 계산 + 조각 비디오 생성"""
        try:
            from enhanced_rtmo_bytetrack_pose_extraction import (
                ByteTracker, create_detection_results, assign_track_ids_from_bytetrack,
                create_enhanced_annotation, EnhancedRTMOPoseExtractor
            )
            
            # ByteTracker 초기화 (윈도우별로 독립)
            tracker = ByteTracker(
                high_thresh=0.6,
                low_thresh=0.1,
                max_disappeared=30,
                min_hits=3
            )
            
            # 윈도우 내에서 트래킹 수행
            tracked_pose_results = []
            for pose_result in window_pose_results:
                # Detection 결과 생성
                detections = create_detection_results(pose_result)
                
                # ByteTrack으로 트래킹 수행
                active_tracks = tracker.update(detections)
                
                # 포즈 결과에 트래킹 ID 할당
                tracked_result = assign_track_ids_from_bytetrack(pose_result, active_tracks)
                tracked_pose_results.append(tracked_result)
            
            # 포즈 모델 초기화 (점수 계산용)
            from mmpose.apis import init_model
            pose_model = init_model(self.detector_config, self.detector_checkpoint, device='cuda:0')
            
            # 윈도우별 점수 계산 및 어노테이션 생성
            annotation, status_message = create_enhanced_annotation(
                tracked_pose_results, video_path, pose_model,
                min_track_length=5,  # 윈도우가 짧으므로 최소 길이 단축
                quality_threshold=0.3,
                weights=[0.30, 0.35, 0.20, 0.10, 0.05]
            )
            
            if annotation is None:
                print(f"Window {window_idx} annotation failed: {status_message}")
                return None
            
            # 조각 비디오 생성
            segment_video_path = None
            if self.save_overlay:
                segment_video_path = self._create_window_segment_video(
                    video_path, output_dir, window_idx, start_frame, end_frame, 
                    tracked_pose_results, pose_model, annotation
                )
            
            # 윈도우 데이터 구성
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
            print(f"Error processing window {window_idx}: {str(e)}")
            return None
    
    def _create_window_segment_video(self, video_path, output_dir, window_idx, start_frame, end_frame, 
                                   tracked_pose_results, pose_model, annotation):
        """윈도우별 조각 비디오 생성"""
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 라벨 폴더 결정
            label = 1 if '/Fight/' in video_path else 0
            label_folder = 'Fight' if label == 1 else 'Normal'
            
            # 데이터셋명
            dataset_name = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
            
            # 출력 경로: output/{dataset}/temp/{Fight|Normal}/{video_name}/
            temp_output_dir = os.path.join(output_dir, dataset_name, 'temp', label_folder, video_name)
            os.makedirs(temp_output_dir, exist_ok=True)
            
            segment_video_path = os.path.join(temp_output_dir, f"{video_name}_{window_idx}.mp4")
            
            # 원본 비디오에서 해당 구간 추출 및 오버레이 적용
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # 시작 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 비디오 속성
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 비디오 라이터 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(segment_video_path, fourcc, fps, (width, height), True)
            
            # 순위 정보 생성
            track_id_to_rank = {}
            if 'persons' in annotation and annotation['persons']:
                for person_data in annotation['persons'].values():
                    track_id = person_data.get('track_id')
                    rank = person_data.get('rank')
                    if track_id is not None and rank is not None:
                        track_id_to_rank[track_id] = rank
            
            # Visualizer 초기화
            from mmpose.registry import VISUALIZERS
            visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
            visualizer.set_dataset_meta(pose_model.dataset_meta)
            
            # 프레임별 처리
            for frame_idx, pose_result in enumerate(tracked_pose_results):
                success, frame = cap.read()
                if not success:
                    break
                
                try:
                    # 포즈 오버레이
                    visualizer.add_datasample(
                        'result',
                        frame,
                        data_sample=pose_result,
                        draw_gt=False,
                        draw_heatmap=False,
                        draw_bbox=False,
                        show_kpt_idx=False,
                        skeleton_style='mmpose'
                    )
                    
                    vis_frame = visualizer.get_image()
                    
                    # 크기 조정
                    if vis_frame.shape[:2] != (height, width):
                        vis_frame = cv2.resize(vis_frame, (width, height))
                    
                    # 트랙 ID 및 순위 표시
                    from enhanced_rtmo_bytetrack_pose_extraction import draw_track_ids
                    vis_frame = draw_track_ids(vis_frame, pose_result, track_id_to_rank, None)
                    
                    out_writer.write(vis_frame)
                    
                except Exception as e:
                    print(f"Warning: Frame {frame_idx} processing failed: {e}")
                    out_writer.write(frame)
            
            # 리소스 정리
            cap.release()
            out_writer.release()
            
            print(f"Created segment video: {segment_video_path}")
            return segment_video_path
            
        except Exception as e:
            print(f"Error creating segment video: {str(e)}")
            return None
    
    def _extract_persons_ranking(self, annotation):
        """어노테이션에서 인물 순위 정보 추출"""
        try:
            if 'persons' not in annotation or not annotation['persons']:
                return []
            
            rankings = []
            for person_key, person_data in annotation['persons'].items():
                rankings.append({
                    'person_key': person_key,
                    'track_id': person_data.get('track_id'),
                    'rank': person_data.get('rank'),
                    'composite_score': person_data.get('composite_score', 0.0),
                    'score_breakdown': person_data.get('score_breakdown', {})
                })
            
            # 순위별 정렬
            rankings.sort(key=lambda x: x['rank'])
            return rankings
            
        except Exception as e:
            print(f"Error extracting rankings: {str(e)}")
            return []
    
    def _create_segments(self, basic_data, stride):
        """레거시 메서드 - 호환성을 위해 유지"""
        try:
            video_info = basic_data['video_info']
            persons_dict = basic_data['persons']
            
            if not persons_dict:
                return []
            
            # 모든 person 데이터 수집
            all_persons = []
            max_frames = 0
            
            for person_key, person_data in persons_dict.items():
                if person_key.startswith('person_'):
                    annotation = person_data['annotation']
                    keypoints = annotation['keypoint']
                    scores = annotation['keypoint_score']
                    
                    # Enhanced annotation은 (1, T, V, C) 형태이므로 squeeze
                    if keypoints.ndim == 4 and keypoints.shape[0] == 1:
                        keypoints = keypoints.squeeze(0)  # (T, V, C)
                    if scores.ndim == 3 and scores.shape[0] == 1:
                        scores = scores.squeeze(0)  # (T, V)
                    
                    enhanced_info = person_data.get('enhanced_info', {})
                    
                    all_persons.append({
                        'person_key': person_key,
                        'keypoints': keypoints,
                        'scores': scores,
                        'composite_score': enhanced_info.get('composite_score', 0.0),
                        'enhanced_info': enhanced_info
                    })
                    
                    max_frames = max(max_frames, keypoints.shape[0])
            
            if not all_persons:
                return []
            
            # 짧은 비디오 처리
            if max_frames < self.clip_len:
                return self._handle_short_video_segments(all_persons, video_info, max_frames)
            
            # 일반 슬라이딩 윈도우 처리
            return self._handle_normal_video_segments(all_persons, video_info, max_frames, stride)
            
        except Exception as e:
            print(f"Error creating segments: {str(e)}")
            return []
    
    def _handle_short_video_segments(self, all_persons, video_info, max_frames):
        """짧은 비디오 세그먼트 처리 (패딩 적용)"""
        segments = []
        
        # 전체 비디오를 하나의 세그먼트로 처리
        segment_persons = []
        for person in all_persons:
            person_keypoints = person['keypoints'][:max_frames]
            person_scores = person['scores'][:max_frames]
            
            # 움직임 강도 계산
            segment_movement = self._calculate_movement_intensity(person_keypoints)
            segment_score = person['composite_score'] * 0.3 + segment_movement * 0.7
            
            segment_persons.append({
                'person_key': person['person_key'],
                'keypoints': person_keypoints,
                'scores': person_scores,
                'segment_score': segment_score,
                'original_composite': person['composite_score'],
                'segment_movement': segment_movement
            })
        
        # 점수 기준 정렬 (전체 인물 유지)
        segment_persons.sort(key=lambda x: x['segment_score'], reverse=True)
        selected_persons = segment_persons  # 모든 인물 유지
        
        # 패딩 처리
        padded_keypoints = []
        padded_scores = []
        
        for person_info in selected_persons:
            actual_frames = person_info['keypoints'].shape[0]
            
            padded_kp = np.zeros((self.clip_len, 17, 2))
            padded_sc = np.zeros((self.clip_len, 17))
            
            padded_kp[:actual_frames] = person_info['keypoints']
            padded_sc[:actual_frames] = person_info['scores']
            
            padded_keypoints.append(padded_kp)
            padded_scores.append(padded_sc)
        
        # 패딩은 필요 없음 (전체 인물 유지)
        # 실제 감지된 인물 수만큼 keypoints 배열 생성
        
        # 세그먼트 생성
        segment = {
            'start_frame': 0,
            'end_frame': max_frames,
            'actual_frames': max_frames,
            'padded_frames': self.clip_len - max_frames,
            'keypoint': np.array(padded_keypoints),  # (num_person, clip_len, V, C)
            'keypoint_score': np.array(padded_scores),  # (num_person, clip_len, V)
            'label': video_info['label'],
            'selected_persons': [
                {
                    'person_key': p['person_key'],
                    'segment_score': p['segment_score'],
                    'original_composite': p['original_composite'],
                    'segment_movement': p['segment_movement']
                } for p in selected_persons
            ]
        }
        
        segments.append(segment)
        return segments
    
    def _handle_normal_video_segments(self, all_persons, video_info, max_frames, stride):
        """일반 비디오 세그먼트 처리 (슬라이딩 윈도우)"""
        segments = []
        
        for start_frame in range(0, max_frames - self.clip_len + 1, stride):
            end_frame = start_frame + self.clip_len
            
            # 현재 세그먼트에서 각 person의 점수 계산
            segment_persons = []
            
            for person in all_persons:
                person_keypoints = person['keypoints'][start_frame:end_frame]
                person_scores = person['scores'][start_frame:end_frame]
                
                # 세그먼트별 동적 점수 계산
                segment_movement = self._calculate_movement_intensity(person_keypoints)
                segment_score = person['composite_score'] * 0.3 + segment_movement * 0.7
                
                segment_persons.append({
                    'person_key': person['person_key'],
                    'keypoints': person_keypoints,
                    'scores': person_scores,
                    'segment_score': segment_score,
                    'original_composite': person['composite_score'],
                    'segment_movement': segment_movement
                })
            
            # 점수 기준 정렬 (전체 인물 유지)
            segment_persons.sort(key=lambda x: x['segment_score'], reverse=True)
            selected_persons = segment_persons  # 모든 인물 유지
            
            # 전체 인물 유지 (패딩 불필요)
            if not selected_persons:
                continue
            
            # 세그먼트 생성
            segment = {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'keypoint': np.array([p['keypoints'] for p in selected_persons]),
                'keypoint_score': np.array([p['scores'] for p in selected_persons]),
                'label': video_info['label'],
                'selected_persons': [
                    {
                        'person_key': p['person_key'],
                        'segment_score': p['segment_score'],
                        'original_composite': p['original_composite'],
                        'segment_movement': p['segment_movement']
                    } for p in selected_persons
                ]
            }
            
            segments.append(segment)
        
        return segments
    
    def _calculate_movement_intensity(self, keypoints):
        """세그먼트 내 움직임 강도 계산"""
        try:
            if keypoints.shape[0] < 2:
                return 0.0
            
            frame_diffs = np.diff(keypoints, axis=0)
            movement_magnitudes = np.sqrt(np.sum(frame_diffs**2, axis=2))
            avg_movement = np.mean(movement_magnitudes)
            
            return float(avg_movement)
            
        except Exception:
            return 0.0
    
    def process_batch_videos(self, video_list, output_dir, input_dir, training_stride=10, inference_stride=50, max_workers=2):
        """여러 비디오 배치 처리"""
        print(f"Processing {len(video_list)} videos with {max_workers} workers...")
        
        successful_videos_data = []
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_video = {
                executor.submit(self.process_single_video_to_segments, video, output_dir, input_dir, training_stride, inference_stride): video 
                for video in video_list
            }
            
            # 결과 수집
            for future in tqdm(as_completed(future_to_video), total=len(video_list), desc="Processing videos"):
                video = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        successful_videos_data.append(result)
                    else:
                        failed_count += 1
                except Exception as exc:
                    print(f'Video {video} generated exception: {exc}')
                    failed_count += 1
        
        print(f"\nBatch processing completed:")
        print(f"  Successful: {len(successful_videos_data)}")
        print(f"  Failed: {failed_count}")
        
        return successful_videos_data
    
    def _process_video_with_gpu(self, video_path, output_dir, input_dir, training_stride, inference_stride, assigned_device):
        """지정된 GPU로 비디오 처리 (멀티 GPU 지원)"""
        try:
            print(f"Processing {os.path.basename(video_path)} on {assigned_device}")
            
            # 임시로 UnifiedPoseProcessor 인스턴스 생성 (지정된 GPU로)
            temp_processor = UnifiedPoseProcessor(
                detector_config=self.detector_config,
                detector_checkpoint=self.detector_checkpoint,
                device=assigned_device,
                gpu_ids=[int(assigned_device.split(':')[1])] if 'cuda' in assigned_device else [],
                multi_gpu=False,  # 개별 워커에서는 단일 GPU 사용
                clip_len=self.clip_len,
                num_person=self.num_person,
                save_overlay=self.save_overlay,
                overlay_fps=self.overlay_fps
            )
            
            # 기존 메서드 호출
            return temp_processor.process_single_video_to_segments(
                video_path, output_dir, input_dir, training_stride, inference_stride
            )
            
        except Exception as e:
            print(f"Error processing {video_path} on {assigned_device}: {str(e)}")
            return None
    
    def create_unified_stgcn_data(self, video_results_list, output_dir, input_dir, train_split=0.7, val_split=0.2):
        """윈도우 기반 비디오 결과들을 통합하여 STGCN 학습용 데이터 생성"""
        
        print("Creating unified STGCN training data...")
        
        dataset_name = os.path.basename(input_dir.rstrip('/\\'))
        
        # 모든 윈도우 데이터 수집 및 STGCN 형식 변환
        all_stgcn_samples = []
        
        for video_result in video_results_list:
            video_name = video_result['video_name']
            label = video_result['label']
            label_folder = video_result['label_folder']
            
            # 비디오별 PKL 파일 저장
            self._save_video_pkl(video_result, output_dir, dataset_name)
            
            # 각 윈도우를 STGCN 샘플로 변환
            for window_data in video_result['windows']:
                stgcn_sample = self._convert_window_to_stgcn_format(
                    window_data, video_name, label, label_folder
                )
                if stgcn_sample:
                    all_stgcn_samples.append(stgcn_sample)
        
        print(f"Total window samples: {len(all_stgcn_samples)}")
        
        # 비디오 단위로 데이터 분할 및 새로운 구조로 저장
        train_segments, val_segments, test_segments = self._split_samples_by_video(
            all_stgcn_samples, video_results_list, train_split, val_split
        )
        
        # 새로운 출력 구조로 저장
        self._save_split_data_new_structure(
            train_segments, val_segments, test_segments, 
            output_dir, dataset_name, video_results_list
        )
        
        # 통합 PKL 파일들 저장
        self._save_unified_pkl_files(
            train_segments, val_segments, test_segments,
            output_dir, dataset_name
        )
        
        # temp 폴더 정리
        self._cleanup_temp_folder(output_dir, dataset_name)

        return len(train_segments), len(val_segments), len(test_segments)
    
    def _save_video_pkl(self, video_result, output_dir, dataset_name):
        """비디오별 PKL 파일 저장"""
        try:
            video_name = video_result['video_name']
            label_folder = video_result['label_folder']
            
            # temp 폴더에 저장
            video_pkl_dir = os.path.join(output_dir, dataset_name, 'temp', label_folder, video_name)
            os.makedirs(video_pkl_dir, exist_ok=True)
            
            video_pkl_path = os.path.join(video_pkl_dir, f"{video_name}_windows.pkl")
            
            with open(video_pkl_path, 'wb') as f:
                pickle.dump(video_result, f)
            
            print(f"Saved video PKL: {video_pkl_path}")
            
        except Exception as e:
            print(f"Error saving video PKL: {str(e)}")
    
    def _convert_window_to_stgcn_format(self, window_data, video_name, label, label_folder):
        """윈도우 데이터를 STGCN 형식으로 변환"""
        try:
            annotation = window_data['annotation']
            
            if 'persons' not in annotation or not annotation['persons']:
                return None
            
            # 모든 인물의 keypoint 데이터 수집
            all_keypoints = []
            all_scores = []
            
            # 순위순으로 정렬하여 처리
            sorted_persons = sorted(
                annotation['persons'].items(),
                key=lambda x: x[1]['rank']
            )
            
            for person_key, person_data in sorted_persons:
                person_annotation = person_data['annotation']
                keypoints = person_annotation['keypoint']  # (1, T, V, C)
                scores = person_annotation['keypoint_score']  # (1, T, V)
                
                # (1, T, V, C) -> (T, V, C)로 변환
                if keypoints.ndim == 4 and keypoints.shape[0] == 1:
                    keypoints = keypoints.squeeze(0)
                if scores.ndim == 3 and scores.shape[0] == 1:
                    scores = scores.squeeze(0)
                
                all_keypoints.append(keypoints)
                all_scores.append(scores)
            
            # (num_person, T, V, C) 형태로 변환
            final_keypoints = np.array(all_keypoints)
            final_scores = np.array(all_scores)
            
            # STGCN 형식으로 변환
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
            print(f"Error converting window to STGCN format: {str(e)}")
            return None
    
    def _split_samples_by_video(self, all_samples, video_results_list, train_split, val_split):
        """비디오 단위로 샘플을 분할 (비디오별로 모든 윈도우가 같은 split에 속함)"""
        try:
            # 비디오별로 그룹화
            fight_videos = []
            normal_videos = []
            
            for video_result in video_results_list:
                video_name = video_result['video_name']
                label = video_result['label']
                
                if label == 1:  # Fight
                    fight_videos.append(video_name)
                else:  # Normal
                    normal_videos.append(video_name)
            
            print(f"Fight videos: {len(fight_videos)}")
            print(f"Normal videos: {len(normal_videos)}")
            
            # 각 라벨별로 비디오를 분할
            def split_videos(videos, train_ratio, val_ratio):
                np.random.seed(42)
                np.random.shuffle(videos)
                
                total = len(videos)
                if total < 3:  # 비디오가 너무 적으면 모두 train에 할당
                    return videos, [], []
                
                train_size = int(total * train_ratio)
                val_size = int(total * val_ratio)
                
                train = videos[:train_size]
                val = videos[train_size:train_size + val_size]
                test = videos[train_size + val_size:]
                
                return train, val, test
            
            fight_train_videos, fight_val_videos, fight_test_videos = split_videos(fight_videos, train_split, val_split)
            normal_train_videos, normal_val_videos, normal_test_videos = split_videos(normal_videos, train_split, val_split)
            
            print(f"Split results:")
            print(f"  Train: Fight({len(fight_train_videos)}), Normal({len(normal_train_videos)})")
            print(f"  Val: Fight({len(fight_val_videos)}), Normal({len(normal_val_videos)})")
            print(f"  Test: Fight({len(fight_test_videos)}), Normal({len(normal_test_videos)})")
            
            # 비디오 분할에 따라 샘플들을 할당
            train_segments = []
            val_segments = []
            test_segments = []
            
            # 각 분할에 속하는 비디오들의 모든 윈도우를 해당 분할에 할당
            all_train_videos = fight_train_videos + normal_train_videos
            all_val_videos = fight_val_videos + normal_val_videos
            all_test_videos = fight_test_videos + normal_test_videos
            
            for sample in all_samples:
                video_name = sample['window_info']['video_name']
                
                if video_name in all_train_videos:
                    train_segments.append(sample)
                elif video_name in all_val_videos:
                    val_segments.append(sample)
                elif video_name in all_test_videos:
                    test_segments.append(sample)
            
            # 셔플
            np.random.shuffle(train_segments)
            np.random.shuffle(val_segments)
            np.random.shuffle(test_segments)
            
            print(f"Final sample counts:")
            print(f"  Train samples (windows): {len(train_segments)}")
            print(f"  Val samples (windows): {len(val_segments)}")
            print(f"  Test samples (windows): {len(test_segments)}")
            
            return train_segments, val_segments, test_segments
            
        except Exception as e:
            print(f"Error splitting samples by video: {str(e)}")
            return [], [], []
    
    
    def _save_split_data_new_structure(self, train_segments, val_segments, test_segments, 
                                     output_dir, dataset_name, video_results_list):
        """새로운 출력 구조로 분할 데이터 저장"""
        try:
            base_output_dir = os.path.join(output_dir, dataset_name)
            
            # 분할별로 처리
            splits_data = {
                'train': train_segments,
                'val': val_segments,
                'test': test_segments
            }
            
            for split_name, segments in splits_data.items():
                if not segments:
                    continue
                
                # Fight/Normal 별로 분류
                fight_segments = [s for s in segments if s['label'] == 1]
                normal_segments = [s for s in segments if s['label'] == 0]
                
                # Fight 폴더 처리
                if fight_segments:
                    fight_dir = os.path.join(base_output_dir, split_name, 'Fight')
                    os.makedirs(fight_dir, exist_ok=True)
                    self._move_segment_files_to_split(fight_segments, fight_dir, 'Fight', video_results_list)
                
                # Normal 폴더 처리
                if normal_segments:
                    normal_dir = os.path.join(base_output_dir, split_name, 'Normal')
                    os.makedirs(normal_dir, exist_ok=True)
                    self._move_segment_files_to_split(normal_segments, normal_dir, 'Normal', video_results_list)
        
        except Exception as e:
            print(f"Error saving split data: {str(e)}")
    
    def _move_segment_files_to_split(self, segments, target_dir, label_folder, video_results_list):
        """세그먼트 파일들을 해당 분할 폴더로 이동"""
        try:
            import shutil
            
            # 비디오별로 그룹화
            video_groups = defaultdict(list)
            for segment in segments:
                video_name = segment['window_info']['video_name']
                video_groups[video_name].append(segment)
            
            print(f"Moving {len(video_groups)} videos to {target_dir}")
            
            # 각 비디오별로 폴더 생성 및 파일 이동
            for video_name, video_segments in video_groups.items():
                video_target_dir = os.path.join(target_dir, video_name)
                os.makedirs(video_target_dir, exist_ok=True)
                
                # 해당 비디오의 원본 결과 찾기
                video_result = None
                for vr in video_results_list:
                    if vr['video_name'] == video_name and vr['label_folder'] == label_folder:
                        video_result = vr
                        break
                
                if not video_result:
                    print(f"Warning: Video result not found for {video_name}")
                    continue
                
                # temp 폴더 경로 수정
                base_output_dir = os.path.dirname(os.path.dirname(target_dir))  # output/{dataset_name}/
                temp_video_dir = os.path.join(base_output_dir, 'temp', label_folder, video_name)
                
                # 비디오별 PKL 파일 이동
                temp_pkl_path = os.path.join(temp_video_dir, f"{video_name}_windows.pkl")
                if os.path.exists(temp_pkl_path):
                    target_pkl_path = os.path.join(video_target_dir, f"{video_name}_windows.pkl")
                    shutil.move(temp_pkl_path, target_pkl_path)  # copy2 대신 move 사용
                    print(f"Moved PKL: {video_name}_windows.pkl")
                else:
                    print(f"Warning: PKL file not found at {temp_pkl_path}")
                
                # 조각 비디오 파일들 이동
                moved_count = 0
                for window_data in video_result['windows']:
                    segment_video_path = window_data.get('segment_video_path')
                    if segment_video_path and os.path.exists(segment_video_path):
                        target_video_path = os.path.join(video_target_dir, os.path.basename(segment_video_path))
                        shutil.move(segment_video_path, target_video_path)  # copy2 대신 move 사용
                        moved_count += 1
                
                print(f"Moved {moved_count} segment videos for {video_name}")
        
        except Exception as e:
            import traceback
            print(f"Error moving segment files: {str(e)}")
            traceback.print_exc()
    
    def _save_unified_pkl_files(self, train_segments, val_segments, test_segments, output_dir, dataset_name):
        """통합 PKL 파일들 저장"""
        try:
            base_output_dir = os.path.join(output_dir, dataset_name)
            
            splits_data = {
                'train': train_segments,
                'val': val_segments,
                'test': test_segments
            }
            
            for split_name, segments in splits_data.items():
                if not segments:
                    continue
                
                pkl_filename = f"{dataset_name}_{split_name}_windows.pkl"
                pkl_path = os.path.join(base_output_dir, pkl_filename)
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(segments, f)
                
                print(f"Saved unified PKL: {pkl_path} ({len(segments)} samples)")
        
        except Exception as e:
            print(f"Error saving unified PKL files: {str(e)}")
    
    def _cleanup_temp_folder(self, output_dir, dataset_name):
        """temp 폴더 정리"""
        try:
            import shutil
            
            temp_folder = os.path.join(output_dir, dataset_name, 'temp')
            
            if os.path.exists(temp_folder):
                print(f"Cleaning up temp folder: {temp_folder}")
                shutil.rmtree(temp_folder)
                print("Temp folder cleaned up successfully")
            else:
                print("Temp folder not found - nothing to clean up")
        
        except Exception as e:
            print(f"Error cleaning up temp folder: {str(e)}")
    
    def _cleanup_multiprocessing_resources(self):
        """멀티프로세싱 리소스 정리"""
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # 세마포어 정리
            for obj in mp.active_children():
                obj.terminate()
                obj.join(timeout=1)
                
        except Exception:
            pass