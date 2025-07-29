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
    
    def __init__(self, detector_config, detector_checkpoint, 
                 clip_len=100, num_person=5, save_overlay=True, overlay_fps=30):
        """
        Args:
            detector_config: RTMO 검출기 설정 파일
            detector_checkpoint: RTMO 검출기 체크포인트 (PTH 파일)
            clip_len: 세그먼트 길이 (프레임)
            num_person: 오버레이 표시할 최대 인물 수 (모든 인물은 저장됨)
            save_overlay: 오버레이 비디오 저장 여부
            overlay_fps: 오버레이 비디오 FPS
        """
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.clip_len = clip_len
        self.num_person = num_person
        self.save_overlay = save_overlay
        self.overlay_fps = overlay_fps
        
        # 멀티프로세싱 리소스 정리
        atexit.register(self._cleanup_multiprocessing_resources)
    
    def process_single_video_to_segments(self, video_path, output_dir, training_stride=10, inference_stride=50):
        """
        단일 비디오를 처리하여 세그먼트별 PKL 생성
        
        Returns:
            dict: {
                'training_segments': [...],  # stride=10 세그먼트들
                'inference_segments': [...], # stride=50 세그먼트들
                'video_info': {...}
            }
        """
        try:
            print(f"Processing video: {os.path.basename(video_path)}")
            
            # 1단계: 기본 포즈 추출 + 추적 (기존 스크립트 활용)
            temp_pkl_path = self._extract_basic_poses(video_path, output_dir)
            if not temp_pkl_path or not os.path.exists(temp_pkl_path):
                print(f"Failed to extract poses from {video_path}")
                return None
            
            # 2단계: 기본 데이터 로드
            with open(temp_pkl_path, 'rb') as f:
                basic_data = pickle.load(f)
            
            # 3단계: 세그먼트별 처리 (학습용 + 추론용)
            training_segments = self._create_segments(basic_data, training_stride)
            inference_segments = self._create_segments(basic_data, inference_stride)
            
            # 4단계: 개별 비디오 세그먼트 PKL 저장
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            segments_pkl_path = os.path.join(output_dir, f"{video_name}_segments.pkl")
            
            segments_data = {
                'video_info': basic_data['video_info'],
                'training_segments': training_segments,
                'inference_segments': inference_segments,
                'source_video': video_path
            }
            
            with open(segments_pkl_path, 'wb') as f:
                pickle.dump(segments_data, f)
            
            # 임시 파일 정리
            if os.path.exists(temp_pkl_path):
                os.remove(temp_pkl_path)
            
            print(f"Generated {len(training_segments)} training + {len(inference_segments)} inference segments")
            
            return segments_data
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
    
    def _extract_basic_poses(self, video_path, output_dir):
        """새로운 클래스를 사용하여 기본 포즈 추출"""
        try:
            # 실패 로거 초기화
            failure_log_path = os.path.join(output_dir, 'enhanced_failed_videos.txt')
            failure_logger = FailureLogger(failure_log_path)
            
            # Enhanced RTMO 포즈 추출기 초기화
            extractor = EnhancedRTMOPoseExtractor(
                config_path=self.detector_config,
                checkpoint_path=self.detector_checkpoint,
                device='cuda:0',
                save_overlay=self.save_overlay,
                num_person=self.num_person,
                overlay_fps=self.overlay_fps
            )
            
            # 단일 비디오 처리
            pkl_path = extractor.process_single_video(video_path, output_dir, failure_logger)
            
            return pkl_path
            
        except Exception as e:
            print(f"Exception in pose extraction: {str(e)}")
            return None
    
    def _create_segments(self, basic_data, stride):
        """기본 데이터에서 세그먼트 생성"""
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
    
    def process_batch_videos(self, video_list, output_dir, training_stride=10, inference_stride=50, max_workers=2):
        """여러 비디오 배치 처리"""
        print(f"Processing {len(video_list)} videos with {max_workers} workers...")
        
        successful_videos = []
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_video = {
                executor.submit(self.process_single_video_to_segments, video, output_dir, training_stride, inference_stride): video 
                for video in video_list
            }
            
            # 결과 수집
            for future in tqdm(as_completed(future_to_video), total=len(video_list), desc="Processing videos"):
                video = future_to_video[future]
                try:
                    result = future.result()
                    if result:
                        successful_videos.append(result)
                    else:
                        failed_count += 1
                except Exception as exc:
                    print(f'Video {video} generated exception: {exc}')
                    failed_count += 1
        
        print(f"\nBatch processing completed:")
        print(f"  Successful: {len(successful_videos)}")
        print(f"  Failed: {failed_count}")
        
        return successful_videos
    
    def create_unified_stgcn_data(self, segments_data_list, output_dir, train_split=0.7, val_split=0.2):
        """개별 세그먼트 데이터들을 통합하여 STGCN 학습용 데이터 생성"""
        
        print("Creating unified STGCN training data...")
        
        # 학습용과 추론용 세그먼트 분리 수집
        all_training_segments = []
        all_inference_segments = []
        
        for segments_data in segments_data_list:
            video_info = segments_data['video_info']
            
            # 학습용 세그먼트 변환
            for segment in segments_data['training_segments']:
                stgcn_sample = self._convert_segment_to_stgcn_format(segment, video_info, 'training')
                all_training_segments.append(stgcn_sample)
            
            # 추론용 세그먼트 변환
            for segment in segments_data['inference_segments']:
                stgcn_sample = self._convert_segment_to_stgcn_format(segment, video_info, 'inference')
                all_inference_segments.append(stgcn_sample)
        
        print(f"Total training segments: {len(all_training_segments)}")
        print(f"Total inference segments: {len(all_inference_segments)}")
        
        # 학습용 데이터 저장 (dense)
        self._split_and_save_data(
            all_training_segments, 
            os.path.join(output_dir, 'dense_training'),
            train_split, val_split, 'training'
        )
        
        # 추론용 데이터 저장 (sparse)
        self._split_and_save_data(
            all_inference_segments,
            os.path.join(output_dir, 'sparse_inference'), 
            train_split, val_split, 'inference'
        )
        
        return len(all_training_segments), len(all_inference_segments)
    
    def _convert_segment_to_stgcn_format(self, segment, video_info, mode):
        """세그먼트를 STGCN 형식으로 변환"""
        return {
            'frame_dir': f"{video_info['frame_dir']}_seg_{segment['start_frame']:04d}",
            'total_frames': self.clip_len,
            'img_shape': video_info['img_shape'],
            'original_shape': video_info['original_shape'],
            'label': segment['label'],
            'keypoint': segment['keypoint'],
            'keypoint_score': segment['keypoint_score'],
            'segment_info': {
                'start_frame': segment['start_frame'],
                'end_frame': segment['end_frame'],
                'mode': mode,
                'selected_persons': segment['selected_persons']
            }
        }
    
    def _split_and_save_data(self, samples, output_dir, train_split, val_split, mode):
        """데이터 분할 및 저장"""
        
        # 라벨 분포 확인
        label_counts = defaultdict(int)
        for sample in samples:
            label_counts[sample['label']] += 1
        
        print(f"\n{mode.upper()} data label distribution:")
        for label, count in label_counts.items():
            label_name = "Fight" if label == 1 else "NonFight"
            print(f"  {label_name}: {count}")
        
        # 랜덤 셔플 및 분할
        np.random.seed(42)
        np.random.shuffle(samples)
        
        total_samples = len(samples)
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        # 저장
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = [
            (train_samples, 'rwf2000_enhanced_sliding_train.pkl'),
            (val_samples, 'rwf2000_enhanced_sliding_val.pkl'),
            (test_samples, 'rwf2000_enhanced_sliding_test.pkl')
        ]
        
        for samples_data, filename in output_files:
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'wb') as f:
                pickle.dump(samples_data, f)
            print(f"Saved {len(samples_data)} segments to {output_path}")
    
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