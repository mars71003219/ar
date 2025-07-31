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
                 clip_len=100, num_person=5, save_overlay=True, overlay_fps=30,
                 # 포즈 추출 파라미터
                 score_thr=0.3, nms_thr=0.35, quality_threshold=0.3, min_track_length=10,
                 # ByteTracker 파라미터
                 track_high_thresh=0.6, track_low_thresh=0.1, track_max_disappeared=30, track_min_hits=3,
                 # 복합 점수 가중치
                 movement_weight=0.30, position_weight=0.35, interaction_weight=0.20, 
                 temporal_weight=0.10, persistence_weight=0.05):
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
            # 포즈 추출 파라미터
            score_thr: 포즈 검출 점수 임계값
            nms_thr: NMS 임계값
            quality_threshold: 트랙 품질 최소 임계값
            min_track_length: 유효한 트랙의 최소 길이
            # ByteTracker 파라미터
            track_high_thresh: ByteTracker 높은 임계값
            track_low_thresh: ByteTracker 낮은 임계값
            track_max_disappeared: 트랙이 사라지기 전까지 최대 프레임 수
            track_min_hits: 유효한 트랙으로 간주하기 위한 최소 히트 수
            # 복합 점수 가중치
            movement_weight: 움직임 점수 가중치
            position_weight: 위치 점수 가중치
            interaction_weight: 상호작용 점수 가중치
            temporal_weight: 시간적 일관성 가중치
            persistence_weight: 지속성 점수 가중치
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
        self.movement_weight = movement_weight
        self.position_weight = position_weight
        self.interaction_weight = interaction_weight
        self.temporal_weight = temporal_weight
        self.persistence_weight = persistence_weight
        
        # GPU 설정 초기화
        self._setup_gpu_environment()
        
        # GPU 설정 확인 로그는 개발시에만 필요하므로 제거
        
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
                    # GPU 설정 완료
                else:
                    self.device = 'cuda:0'
                    self.gpu_ids = [0]
            else:
                self.device = 'cpu'
                self.gpu_ids = []
        except Exception as e:
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
            # 비디오 처리 시작
            
            # 비디오 파일 검증
            if not os.path.exists(video_path):
                return self._create_failure_analysis(
                    video_path,
                    "FILE_NOT_FOUND",
                    "Video file does not exist at specified path",
                    {"checked_path": video_path, "directory_exists": os.path.exists(os.path.dirname(video_path))}
                )
            
            # 1단계: 전체 비디오에 대한 포즈 추정 (윈도우별 트래킹은 별도)
            all_pose_results = self._extract_full_video_poses(video_path, output_dir)
            
            # 포즈 추출 결과 확인
            if isinstance(all_pose_results, dict) and 'failure_stage' in all_pose_results:
                # 포즈 추출이 실패 분석을 반환한 경우
                return all_pose_results
            elif not all_pose_results:
                # 포즈 추출이 None이나 빈 리스트를 반환한 경우
                return self._create_failure_analysis(
                    video_path, 
                    "POSE_EXTRACTION_FAILED", 
                    "No poses detected in video - video may not contain people or RTMO model failed",
                    {"stage": "pose_extraction", "poses_detected": 0}
                )
            
            # 짧은 비디오 처리: 100프레임 미만인 경우 패딩
            if len(all_pose_results) < self.clip_len:
                all_pose_results = self._apply_temporal_padding(all_pose_results, self.clip_len)
            
            # 2단계: 윈도우별 ByteTrack + 점수 계산 (training mode에서는 작은 stride 사용)
            stride = training_stride  # 훈련용 데이터에서는 조밀한 윈도우 생성
            windows_data = self._process_windows_with_tracking(all_pose_results, video_path, output_dir, stride)
            
            # 윈도우 처리 결과 확인
            if isinstance(windows_data, dict) and 'failure_stage' in windows_data:
                # 윈도우 처리가 실패 분석을 반환한 경우
                return windows_data
            elif not windows_data:
                # 윈도우 처리가 None이나 빈 리스트를 반환한 경우
                return self._create_failure_analysis(
                    video_path,
                    "WINDOW_PROCESSING_FAILED", 
                    "No valid windows generated - insufficient tracking data or processing error",
                    {"stage": "window_processing", "total_frames": len(all_pose_results), "windows_generated": 0}
                )
            
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
            
            # 윈도우 생성 완료
            
            return video_result
            
        except Exception as e:
            import traceback
            video_name = os.path.basename(video_path)
            
            # 구체적인 실패 원인 분석
            failure_analysis = self._analyze_video_failure(video_path, e, traceback.format_exc())
            
            print(f"Video processing failed: {video_name}")
            print(f"  Failure Stage: {failure_analysis.get('failure_stage', 'UNKNOWN')}")
            print(f"  Root Cause: {failure_analysis.get('root_cause', 'Unknown error')}")
            
            return failure_analysis
    
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
                device=self.device,  # 설정된 디바이스 사용
                save_overlay=False,  # 전체 비디오 처리시에는 오버레이 생성 안함
                num_person=self.num_person,
                overlay_fps=self.overlay_fps,
                # 포즈 추출 파라미터 전달
                score_thr=self.score_thr,
                nms_thr=self.nms_thr,
                quality_threshold=self.quality_threshold,
                min_track_length=self.min_track_length,
                # ByteTracker 파라미터 전달
                track_high_thresh=self.track_high_thresh,
                track_low_thresh=self.track_low_thresh,
                track_max_disappeared=self.track_max_disappeared,
                track_min_hits=self.track_min_hits,
                # 복합 점수 가중치 전달
                weights=[self.movement_weight, self.position_weight, self.interaction_weight, 
                        self.temporal_weight, self.persistence_weight]
            )
            
            # 전체 비디오에 대한 포즈 추정만 수행
            pose_results = extractor.extract_poses_only(video_path, failure_logger)
            
            # 포즈 추출 결과 검증
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
        """윈도우별로 ByteTrack + 점수 계산 수행"""
        try:
            total_frames = len(all_pose_results)
            windows_data = []
            
            # stride를 사용한 슬라이딩 윈도우 생성
            for window_idx, start_frame in enumerate(range(0, total_frames - self.clip_len + 1, stride)):
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
                    # 윈도우 데이터가 실패 분석 결과인지 확인
                    if isinstance(window_data, dict) and 'failure_stage' in window_data:
                        # 윈도우 처리 중 실패 발생 - 실패 분석 결과를 반환
                        return window_data
                    else:
                        # 정상 윈도우 데이터
                        windows_data.append(window_data)
            
            # 유효한 윈도우가 하나도 없는 경우
            if len(windows_data) == 0:
                return self._create_failure_analysis(
                    video_path,
                    "NO_VALID_WINDOWS",
                    "No valid windows could be processed - all windows failed annotation or tracking",
                    {
                        "stage": "window_processing",
                        "total_frames": total_frames,
                        "total_windows_attempted": len(range(0, total_frames - self.clip_len + 1, stride)),
                        "valid_windows_generated": 0
                    }
                )
            
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
        """단일 윈도우 처리: ByteTrack + 점수 계산 + 조각 비디오 생성"""
        try:
            from enhanced_rtmo_bytetrack_pose_extraction import (
                ByteTracker, create_detection_results, assign_track_ids_from_bytetrack,
                create_enhanced_annotation, EnhancedRTMOPoseExtractor
            )
            
            # ByteTracker 초기화 (윈도우별로 독립)
            tracker = ByteTracker(
                high_thresh=self.track_high_thresh,
                low_thresh=self.track_low_thresh,
                max_disappeared=self.track_max_disappeared,
                min_hits=self.track_min_hits
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
            pose_model = init_model(self.detector_config, self.detector_checkpoint, device=self.device)
            
            # 윈도우별 점수 계산 및 어노테이션 생성
            annotation, status_message = create_enhanced_annotation(
                tracked_pose_results, video_path, pose_model,
                min_track_length=max(3, self.min_track_length // 10),  # 윈도우가 짧으므로 최소 길이 단축
                quality_threshold=self.quality_threshold,
                weights=[self.movement_weight, self.position_weight, self.interaction_weight, 
                        self.temporal_weight, self.persistence_weight]
            )
            
            if annotation is None:
                # 어노테이션 생성 실패 시 구체적인 원인을 포함한 실패 분석 반환
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
                    out_writer.write(frame)
            
            # 리소스 정리
            cap.release()
            out_writer.release()
            
            # 세그먼트 비디오 생성 완료
            return segment_video_path
            
        except Exception as e:
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
            return []
    
    def _apply_temporal_padding(self, pose_results, target_length):
        """
        짧은 비디오에 대해 시간적 패딩을 적용하여 최소 길이 확보
        
        Args:
            pose_results: 원본 포즈 결과 리스트
            target_length: 목표 길이 (일반적으로 clip_len=100)
        
        Returns:
            패딩이 적용된 포즈 결과 리스트
        """
        try:
            if len(pose_results) >= target_length:
                return pose_results
            
            current_length = len(pose_results)
            needed_frames = target_length - current_length
            
            # 패딩 수행 중
            
            # 패딩 전략: 마지막 프레임을 반복
            if current_length > 0:
                last_frame = pose_results[-1]
                padded_results = pose_results.copy()
                
                # 마지막 프레임을 필요한 만큼 반복
                for _ in range(needed_frames):
                    padded_results.append(last_frame)
                
                # 마지막 프레임 반복 패딩 사용
                return padded_results
            else:
                return pose_results
                
        except Exception as e:
            return pose_results
    
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
        # 배치 처리 시작
        
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
                    failed_count += 1
        
        # 배치 처리 완료/minimum leng
        
        return successful_videos_data
    
    def _process_video_with_gpu(self, video_path, output_dir, input_dir, training_stride, inference_stride, assigned_device):
        """지정된 GPU로 비디오 처리 (멀티 GPU 지원)"""
        try:
            # GPU별 비디오 처리 시작
            
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
            return None
    
    def create_unified_stgcn_data(self, video_results_list, output_dir, input_dir, train_split=0.7, val_split=0.2):
        """윈도우 기반 비디오 결과들을 통합하여 STGCN 학습용 데이터 생성"""
        
        # STGCN 훈련 데이터 생성 시작
        
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
        
        # 전체 윈도우 샘플 배열 생성 완료
        
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
        
        # 로그 및 출력 파일들을 최종 폴더로 이동
        self._organize_output_files(output_dir, dataset_name)
        
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
        """비디오 단위로 샘플을 분할 (언더바 앞부분이 같은 파일들은 같은 그룹으로 분할)"""
        try:
            # 언더바 앞부분을 기준으로 비디오 그룹화
            fight_video_groups = defaultdict(list)
            normal_video_groups = defaultdict(list)
            
            for video_result in video_results_list:
                video_name = video_result['video_name']
                label = video_result['label']
                
                # 언더바 앞부분 추출 (예: 0lHQ2f0d_001 -> 0lHQ2f0d)
                if '_' in video_name:
                    group_key = video_name.split('_')[0]
                else:
                    group_key = video_name  # 언더바가 없으면 전체 파일명을 그룹키로 사용
                
                if label == 1:  # Fight
                    fight_video_groups[group_key].append(video_name)
                else:  # Normal
                    normal_video_groups[group_key].append(video_name)
            
            print(f"Fight video groups: {len(fight_video_groups)} groups, {sum(len(videos) for videos in fight_video_groups.values())} videos")
            print(f"Normal video groups: {len(normal_video_groups)} groups, {sum(len(videos) for videos in normal_video_groups.values())} videos")
            
            # 그룹 정보 출력
            for group_key, videos in fight_video_groups.items():
                print(f"  Fight group '{group_key}': {len(videos)} videos - {videos[:3]}{'...' if len(videos) > 3 else ''}")
            for group_key, videos in normal_video_groups.items():
                print(f"  Normal group '{group_key}': {len(videos)} videos - {videos[:3]}{'...' if len(videos) > 3 else ''}")
            
            # 각 라벨별로 그룹을 분할 (그룹 단위로 분할)
            def split_video_groups(video_groups, train_ratio, val_ratio):
                # 그룹키 목록 생성
                group_keys = list(video_groups.keys())
                np.random.seed(42)
                np.random.shuffle(group_keys)
                
                total_groups = len(group_keys)
                if total_groups < 3:  # 그룹이 너무 적으면 모두 train에 할당
                    train_videos = []
                    for group_key in group_keys:
                        train_videos.extend(video_groups[group_key])
                    return train_videos, [], []
                
                train_size = int(total_groups * train_ratio)
                val_size = int(total_groups * val_ratio)
                
                # 그룹별로 분할
                train_group_keys = group_keys[:train_size]
                val_group_keys = group_keys[train_size:train_size + val_size]
                test_group_keys = group_keys[train_size + val_size:]
                
                # 각 분할에 속하는 모든 비디오 수집
                train_videos = []
                val_videos = []
                test_videos = []
                
                for group_key in train_group_keys:
                    train_videos.extend(video_groups[group_key])
                for group_key in val_group_keys:
                    val_videos.extend(video_groups[group_key])
                for group_key in test_group_keys:
                    test_videos.extend(video_groups[group_key])
                
                return train_videos, val_videos, test_videos
            
            fight_train_videos, fight_val_videos, fight_test_videos = split_video_groups(fight_video_groups, train_split, val_split)
            normal_train_videos, normal_val_videos, normal_test_videos = split_video_groups(normal_video_groups, train_split, val_split)
            
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
                    fight_dir = os.path.join(output_dir, dataset_name, split_name, 'Fight')
                    os.makedirs(fight_dir, exist_ok=True)
                    self._move_segment_files_to_split(fight_segments, fight_dir, 'Fight', video_results_list)
                
                # Normal 폴더 처리
                if normal_segments:
                    normal_dir = os.path.join(output_dir, dataset_name, split_name, 'Normal')
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
            
            print(f"Moving {len(video_groups)} videos from temp to {os.path.basename(target_dir)}")
            
            total_moved_pkl = 0
            total_moved_videos = 0
            failed_videos = []
            
            # 각 비디오별로 폴더 생성 및 파일 이동
            for video_name in video_groups.keys():
                try:
                    video_target_dir = os.path.join(target_dir, video_name)
                    os.makedirs(video_target_dir, exist_ok=True)
                    
                    # 해당 비디오의 원본 결과 찾기
                    video_result = None
                    for vr in video_results_list:
                        if vr['video_name'] == video_name and vr['label_folder'] == label_folder:
                            video_result = vr
                            break
                    
                    if not video_result:
                        print(f"  WARNING: Video result not found for {video_name}")
                        failed_videos.append(f"{video_name} (no result data)")
                        continue
                    
                    # temp 폴더 경로 계산
                    # target_dir: output/{dataset_name}/{split_name}/{label_folder}
                    dataset_output_dir = os.path.dirname(os.path.dirname(target_dir))  # output/{dataset_name}/
                    temp_video_dir = os.path.join(dataset_output_dir, 'temp', label_folder, video_name)
                    
                    if not os.path.exists(temp_video_dir):
                        print(f"  WARNING: Temp folder not found: {temp_video_dir}")
                        failed_videos.append(f"{video_name} (temp folder missing)")
                        continue
                    
                    # 1. 비디오별 PKL 파일 이동
                    temp_pkl_path = os.path.join(temp_video_dir, f"{video_name}_windows.pkl")
                    if os.path.exists(temp_pkl_path):
                        target_pkl_path = os.path.join(video_target_dir, f"{video_name}_windows.pkl")
                        shutil.move(temp_pkl_path, target_pkl_path)
                        print(f"  ✓ Moved PKL: {video_name}_windows.pkl")
                        total_moved_pkl += 1
                    else:
                        print(f"  WARNING: PKL file not found: {temp_pkl_path}")
                    
                    # 2. 조각 비디오 파일들 이동
                    moved_count = 0
                    for window_data in video_result['windows']:
                        segment_video_path = window_data.get('segment_video_path')
                        if segment_video_path and os.path.exists(segment_video_path):
                            target_video_path = os.path.join(video_target_dir, os.path.basename(segment_video_path))
                            try:
                                shutil.move(segment_video_path, target_video_path)
                                moved_count += 1
                            except Exception as move_error:
                                print(f"  WARNING: Failed to move {os.path.basename(segment_video_path)}: {move_error}")
                    
                    if moved_count > 0:
                        print(f"  ✓ Moved {moved_count} segment videos for {video_name}")
                        total_moved_videos += moved_count
                    else:
                        print(f"  ⚠ No segment videos found for {video_name}")
                    
                    # 3. 빈 temp 비디오 폴더 제거 시도
                    try:
                        if os.path.exists(temp_video_dir) and not os.listdir(temp_video_dir):
                            os.rmdir(temp_video_dir)
                            print(f"  ✓ Removed empty temp folder for {video_name}")
                    except Exception as cleanup_error:
                        print(f"  WARNING: Could not remove temp folder {temp_video_dir}: {cleanup_error}")
                
                except Exception as video_error:
                    print(f"  ERROR: Failed to process {video_name}: {video_error}")
                    failed_videos.append(f"{video_name} ({str(video_error)})")
                    continue
            
            # 이동 결과 요약
            print(f"File moving completed:")
            print(f"  ✓ Moved {total_moved_pkl} PKL files")
            print(f"  ✓ Moved {total_moved_videos} segment videos")
            if failed_videos:
                print(f"  ⚠ Failed videos ({len(failed_videos)}):")
                for failed in failed_videos[:5]:  # 최대 5개만 표시
                    print(f"    - {failed}")
                if len(failed_videos) > 5:
                    print(f"    ... and {len(failed_videos) - 5} more")
        
        except Exception as e:
            import traceback
            print(f"ERROR: Critical failure in file moving: {str(e)}")
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
        """temp 폴더 정리 - 강제 삭제"""
        try:
            import shutil
            
            base_output_dir = os.path.join(output_dir, dataset_name)
            print(f"Cleaning temp folders in: {base_output_dir}")
            
            temp_folders_removed = 0
            
            # 가능한 temp 폴더 위치들
            temp_locations = [
                os.path.join(base_output_dir, 'temp'),              # 메인 temp
                os.path.join(base_output_dir, 'train', 'temp'),     # train/temp
                os.path.join(base_output_dir, 'val', 'temp'),       # val/temp
                os.path.join(base_output_dir, 'test', 'temp')       # test/temp
            ]
            
            # 추가로 재귀적으로 temp 폴더 찾기
            try:
                for root, dirs, files in os.walk(base_output_dir):
                    for dir_name in dirs:
                        if dir_name == 'temp':
                            temp_path = os.path.join(root, dir_name)
                            if temp_path not in temp_locations:
                                temp_locations.append(temp_path)
            except Exception:
                pass
            
            for temp_path in temp_locations:
                if os.path.exists(temp_path):
                    try:
                        print(f"  Removing temp folder: {temp_path}")
                        
                        # 강제 삭제 시도
                        shutil.rmtree(temp_path, ignore_errors=True)
                        
                        # 삭제 확인
                        if not os.path.exists(temp_path):
                            print(f"    ✓ Successfully removed")
                            temp_folders_removed += 1
                        else:
                            print(f"    ⚠ Folder still exists, trying alternative method...")
                            # 대안 방법: 개별 파일 삭제
                            self._force_remove_directory(temp_path)
                            if not os.path.exists(temp_path):
                                print(f"    ✓ Successfully removed with alternative method")
                                temp_folders_removed += 1
                            else:
                                print(f"    ✗ Failed to remove completely")
                    except Exception as e:
                        print(f"    ✗ Error removing {temp_path}: {e}")
            
            if temp_folders_removed > 0:
                print(f"  ✓ Removed {temp_folders_removed} temp folders")
            else:
                print(f"  No temp folders found to remove")
                
        except Exception as e:
            print(f"ERROR: Failed to cleanup temp folders: {e}")
    
    def _force_remove_directory(self, dir_path):
        """강제로 디렉토리와 모든 내용을 삭제"""
        try:
            import stat
            
            def handle_remove_readonly(func, path, exc):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            
            shutil.rmtree(dir_path, onerror=handle_remove_readonly)
        except Exception:
            # 최후의 수단: 개별 파일 삭제
            try:
                for root, dirs, files in os.walk(dir_path, topdown=False):
                    for name in files:
                        try:
                            file_path = os.path.join(root, name)
                            os.chmod(file_path, stat.S_IWRITE)
                            os.remove(file_path)
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                os.rmdir(dir_path)
            except Exception:
                pass
    
    def _analyze_video_failure(self, video_path, exception, full_traceback):
        """비디오 처리 실패 원인을 상세히 분석"""
        video_name = os.path.basename(video_path)
        failure_stage = "UNKNOWN"
        root_cause = "Unknown error during processing"
        detailed_info = {}
        
        try:
            # 1. 비디오 파일 기본 검사
            if not os.path.exists(video_path):
                failure_stage = "FILE_NOT_FOUND"
                root_cause = "Video file does not exist"
                detailed_info = {
                    'checked_path': video_path,
                    'directory_exists': os.path.exists(os.path.dirname(video_path)),
                    'parent_directory': os.path.dirname(video_path)
                }
            else:
                # 2. 비디오 파일 접근성 및 속성 검사
                try:
                    import cv2
                    file_size = os.path.getsize(video_path)
                    detailed_info['file_size_bytes'] = file_size
                    
                    if file_size == 0:
                        failure_stage = "EMPTY_FILE"
                        root_cause = "Video file is empty (0 bytes)"
                    else:
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            detailed_info.update({
                                'frame_count': frame_count,
                                'fps': fps,
                                'resolution': f"{width}x{height}",
                                'duration_seconds': frame_count / fps if fps > 0 else 0
                            })
                            cap.release()
                            
                            if frame_count == 0:
                                failure_stage = "NO_FRAMES"
                                root_cause = "Video file contains no frames"
                            elif fps <= 0:
                                failure_stage = "INVALID_FPS"
                                root_cause = "Video has invalid FPS metadata"
                            elif width <= 0 or height <= 0:
                                failure_stage = "INVALID_RESOLUTION"
                                root_cause = "Video has invalid resolution metadata"
                            else:
                                # 비디오 파일은 정상, 처리 단계에서 실패
                                failure_stage, root_cause = self._analyze_processing_failure(exception, full_traceback)
                        else:
                            failure_stage = "CODEC_ERROR"
                            root_cause = "Cannot decode video - codec issue or corrupted file"
                except Exception as video_error:
                    failure_stage = "FILE_ACCESS_ERROR" 
                    root_cause = f"Error accessing video file: {str(video_error)}"
                    detailed_info['access_error'] = str(video_error)
            
            # 3. 에러 메시지에서 추가 단서 수집
            error_message = str(exception).lower()
            
            # GPU/CUDA 관련 에러 확인
            if any(keyword in error_message for keyword in ['cuda', 'gpu', 'device']):
                if failure_stage == "UNKNOWN":
                    failure_stage = "GPU_ERROR"
                    root_cause = "CUDA/GPU related error"
                detailed_info['gpu_related'] = True
            
            # 메모리 관련 에러 확인  
            if any(keyword in error_message for keyword in ['memory', 'out of memory', 'allocation']):
                if failure_stage == "UNKNOWN":
                    failure_stage = "MEMORY_ERROR"
                    root_cause = "Insufficient memory"
                detailed_info['memory_related'] = True
            
            # 포즈 추출 관련 에러 확인
            if any(keyword in full_traceback.lower() for keyword in ['pose', 'rtmo', 'extract_poses']):
                if failure_stage == "UNKNOWN":
                    failure_stage = "POSE_EXTRACTION_FAILED"
                    root_cause = "RTMO pose detection failed - no persons detected or model error"
                detailed_info['pose_extraction_failed'] = True
            
            # 트래킹 관련 에러 확인
            if any(keyword in full_traceback.lower() for keyword in ['track', 'bytetrack', 'tracking']):
                if failure_stage == "UNKNOWN":
                    failure_stage = "TRACKING_FAILED"
                    root_cause = "Person tracking failed - insufficient poses or tracking error"
                detailed_info['tracking_failed'] = True
            
            # 윈도우 처리 관련 에러 확인
            if any(keyword in full_traceback.lower() for keyword in ['window', 'annotation', 'segment']):
                if failure_stage == "UNKNOWN":
                    failure_stage = "WINDOW_PROCESSING_FAILED"
                    root_cause = "Window processing failed - insufficient data for annotation"
                detailed_info['window_processing_failed'] = True
                
        except Exception as analysis_error:
            detailed_info['analysis_error'] = str(analysis_error)
        
        # 해결 방법 제안
        solution_map = {
            'FILE_NOT_FOUND': 'Check file path and ensure video file exists',
            'EMPTY_FILE': 'Re-download or restore the video file',
            'CODEC_ERROR': 'Convert video to MP4/H264 format',
            'NO_FRAMES': 'Check video integrity and re-encode if corrupted',
            'INVALID_FPS': 'Re-encode video with valid metadata',
            'INVALID_RESOLUTION': 'Re-encode video with valid resolution',
            'GPU_ERROR': 'Check GPU availability, memory, and CUDA installation',
            'MEMORY_ERROR': 'Reduce batch size or free up system/GPU memory',
            'POSE_EXTRACTION_FAILED': 'Ensure video contains people, check RTMO model',
            'TRACKING_FAILED': 'Adjust tracking parameters or improve pose detection quality',
            'WINDOW_PROCESSING_FAILED': 'Check minimum track length and annotation requirements',
            'FILE_ACCESS_ERROR': 'Check file permissions and system access rights'
        }
        
        return {
            'video_path': video_path,
            'video_name': video_name,
            'failure_stage': failure_stage,
            'root_cause': root_cause,
            'suggested_solution': solution_map.get(failure_stage, 'Review error logs for specific details'),
            'detailed_info': detailed_info,
            'error_type': type(exception).__name__,
            'error_message': str(exception),
            'full_traceback': full_traceback,
            'timestamp': self._get_current_timestamp()
        }
    
    def _analyze_processing_failure(self, exception, full_traceback):
        """처리 단계에서의 실패 원인 분석"""
        error_msg = str(exception).lower()
        traceback_lower = full_traceback.lower()
        
        # 특정 함수나 모듈에서의 실패 확인
        if '_extract_full_video_poses' in traceback_lower:
            return "POSE_EXTRACTION_FAILED", "RTMO pose detection failed - no persons detected in video"
        elif '_process_windows_with_tracking' in traceback_lower:
            return "WINDOW_PROCESSING_FAILED", "Window-based tracking processing failed"
        elif 'bytetrack' in traceback_lower:
            return "TRACKING_FAILED", "ByteTracker algorithm failed during person tracking"
        elif 'annotation' in traceback_lower:
            return "ANNOTATION_FAILED", "Failed to create enhanced annotation - insufficient tracking data"
        elif 'segment_video' in traceback_lower:
            return "SEGMENT_VIDEO_FAILED", "Failed to create segment video output"
        else:
            return "PROCESSING_ERROR", f"Processing failed with {type(exception).__name__}: {str(exception)}"
    
    def _organize_output_files(self, output_dir, dataset_name):
        """로그 및 출력 파일들을 최종 폴더로 정리"""
        try:
            import shutil
            import glob
            
            base_output_dir = os.path.join(output_dir, dataset_name)
            
            # logs 폴더 생성
            logs_dir = os.path.join(base_output_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            print(f"Organizing log files to: {logs_dir}")
            
            moved_count = 0
            
            # 직접적으로 알려진 로그 파일들 이동
            log_extensions = ['.log', '.json', '.txt']
            log_patterns = ['failed_videos_', 'processing_errors_', 'processing_summary_', 'enhanced_failed_videos']
            
            # 1. 메인 출력 디렉토리에서 로그 파일들 찾기
            for file_name in os.listdir(output_dir):
                if file_name.startswith('.'):
                    continue
                    
                file_path = os.path.join(output_dir, file_name)
                if not os.path.isfile(file_path):
                    continue
                
                # 로그 파일인지 확인
                is_log_file = False
                
                # 확장자로 확인
                for ext in log_extensions:
                    if file_name.endswith(ext):
                        is_log_file = True
                        break
                
                # 패턴으로 확인
                if not is_log_file:
                    for pattern in log_patterns:
                        if pattern in file_name:
                            is_log_file = True
                            break
                
                if is_log_file:
                    try:
                        target_path = os.path.join(logs_dir, file_name)
                        
                        # 중복 파일명 처리
                        counter = 1
                        original_target = target_path
                        while os.path.exists(target_path):
                            name, ext = os.path.splitext(original_target)
                            target_path = f"{name}_{counter:02d}{ext}"
                            counter += 1
                        
                        shutil.move(file_path, target_path)
                        print(f"  ✓ Moved: {file_name}")
                        moved_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to move {file_name}: {e}")
            
            # 2. 데이터셋 디렉토리에서도 로그 파일들 찾기
            if os.path.exists(base_output_dir):
                for file_name in os.listdir(base_output_dir):
                    if file_name.startswith('.') or file_name == 'logs':
                        continue
                        
                    file_path = os.path.join(base_output_dir, file_name)
                    if not os.path.isfile(file_path):
                        continue
                    
                    # 로그 파일인지 확인
                    is_log_file = False
                    
                    # 확장자로 확인
                    for ext in log_extensions:
                        if file_name.endswith(ext):
                            is_log_file = True
                            break
                    
                    # 패턴으로 확인
                    if not is_log_file:
                        for pattern in log_patterns:
                            if pattern in file_name:
                                is_log_file = True
                                break
                    
                    if is_log_file:
                        try:
                            target_path = os.path.join(logs_dir, file_name)
                            
                            # 중복 파일명 처리
                            counter = 1
                            original_target = target_path
                            while os.path.exists(target_path):
                                name, ext = os.path.splitext(original_target)
                                target_path = f"{name}_{counter:02d}{ext}"
                                counter += 1
                            
                            shutil.move(file_path, target_path)
                            print(f"  ✓ Moved: {file_name}")
                            moved_count += 1
                        except Exception as e:
                            print(f"  ✗ Failed to move {file_name}: {e}")
            
            # 3. 하위 폴더에서도 로그 파일 검색 (train, val, test 폴더 등)
            if os.path.exists(base_output_dir):
                for root, dirs, files in os.walk(base_output_dir):
                    # logs 폴더는 건너뛰기
                    if 'logs' in root:
                        continue
                        
                    for file_name in files:
                        if file_name.startswith('.'):
                            continue
                            
                        # 로그 파일인지 확인
                        is_log_file = False
                        
                        # 확장자로 확인
                        for ext in log_extensions:
                            if file_name.endswith(ext):
                                is_log_file = True
                                break
                        
                        # 패턴으로 확인
                        if not is_log_file:
                            for pattern in log_patterns:
                                if pattern in file_name:
                                    is_log_file = True
                                    break
                        
                        if is_log_file:
                            file_path = os.path.join(root, file_name)
                            try:
                                target_path = os.path.join(logs_dir, file_name)
                                
                                # 중복 파일명 처리
                                counter = 1
                                original_target = target_path
                                while os.path.exists(target_path):
                                    name, ext = os.path.splitext(original_target)
                                    target_path = f"{name}_{counter:02d}{ext}"
                                    counter += 1
                                
                                shutil.move(file_path, target_path)
                                print(f"  ✓ Moved: {file_name}")
                                moved_count += 1
                            except Exception as e:
                                print(f"  ✗ Failed to move {file_name}: {e}")
            
            if moved_count > 0:
                print(f"  ✓ Total moved: {moved_count} files to logs/")
                # README 파일 생성
                self._create_logs_readme(logs_dir, dataset_name, moved_count)
            else:
                print(f"  No log files found to move")
                
        except Exception as e:
            print(f"ERROR: Failed to organize output files: {e}")
    
    def _create_logs_readme(self, logs_dir, dataset_name, files_count):
        """logs 폴더에 설명 파일 생성"""
        try:
            readme_path = os.path.join(logs_dir, 'README.md')
            
            readme_content = f"""# Processing Logs - {dataset_name}

This directory contains all log files and processing information for the {dataset_name} dataset.

## File Types

### Error Logs
- `processing_errors_*.log` - Detailed error logs with timestamps
- `enhanced_failed_videos.txt` - Enhanced RTMO processing failures
- `failed_videos_*.json` - Structured failure information with diagnosis

### Processing Summary
- `processing_summary_*.json` - Overall processing statistics and metrics

### Other Files
- `*.log` - General log files
- `*.txt` - Text-based logs and reports
- `*.json` - Structured data and configuration files

## File Count
Total files organized: {files_count}

## Generated
Generated at: {self._get_current_timestamp()}
"""
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print(f"  ✓ Created logs/README.md")
            
        except Exception as e:
            print(f"  ⚠ Failed to create README: {e}")
    
    def _get_current_timestamp(self):
        """현재 시간을 ISO 포맷으로 반환"""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except Exception:
            return None
    
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
    
    def merge_existing_pkl_files(self, processed_data_dir, output_dir, train_split=0.7, val_split=0.2):
        """
        기존에 처리된 temp 폴더 구조에서 PKL 파일들을 수집하여 train/val/test로 분할하고 통합 PKL 생성
        
        Args:
            processed_data_dir: 처리된 데이터가 있는 디렉토리 (temp 폴더가 포함된)
            output_dir: 출력 디렉토리
            train_split: 학습 데이터 비율
            val_split: 검증 데이터 비율
        
        Returns:
            tuple: (train_count, val_count, test_count)
        """
        try:
            print("=" * 70)
            print(" PKL Merge Mode - Collecting existing processed data")
            print("=" * 70)
            
            # 데이터셋명 추출
            dataset_name = os.path.basename(processed_data_dir.rstrip('/\\'))
            temp_dir = os.path.join(processed_data_dir, 'temp')
            
            if not os.path.exists(temp_dir):
                print(f"Error: temp directory not found at {temp_dir}")
                return 0, 0, 0
            
            print(f"Dataset: {dataset_name}")
            print(f"Temp directory: {temp_dir}")
            
            # temp 폴더에서 비디오별 PKL 파일 수집
            video_results_list = []
            all_stgcn_samples = []
            
            # Fight와 Normal 폴더에서 PKL 파일 검색
            for label_folder in ['Fight', 'Normal']:
                label_dir = os.path.join(temp_dir, label_folder)
                if not os.path.exists(label_dir):
                    print(f"Warning: {label_folder} directory not found in temp")
                    continue
                
                print(f"Processing {label_folder} videos...")
                label = 1 if label_folder == 'Fight' else 0
                
                # 각 비디오 폴더에서 PKL 파일 로드
                for video_name in os.listdir(label_dir):
                    video_dir = os.path.join(label_dir, video_name)
                    if not os.path.isdir(video_dir):
                        continue
                    
                    pkl_file = os.path.join(video_dir, f"{video_name}_windows.pkl")
                    if not os.path.exists(pkl_file):
                        print(f"Warning: PKL file not found for {video_name}")
                        continue
                    
                    try:
                        with open(pkl_file, 'rb') as f:
                            video_result = pickle.load(f)
                        
                        # 기본 정보 보완
                        if 'video_name' not in video_result:
                            video_result['video_name'] = video_name
                        if 'label' not in video_result:
                            video_result['label'] = label
                        if 'label_folder' not in video_result:
                            video_result['label_folder'] = label_folder
                        if 'dataset_name' not in video_result:
                            video_result['dataset_name'] = dataset_name
                        
                        video_results_list.append(video_result)
                        
                        # 각 윈도우를 STGCN 샘플로 변환
                        if 'windows' in video_result:
                            for window_data in video_result['windows']:
                                stgcn_sample = self._convert_window_to_stgcn_format(
                                    window_data, video_name, label, label_folder
                                )
                                if stgcn_sample:
                                    all_stgcn_samples.append(stgcn_sample)
                        
                        print(f"  Loaded: {video_name} ({len(video_result.get('windows', []))} windows)")
                        
                    except Exception as e:
                        print(f"Error loading PKL for {video_name}: {str(e)}")
                        continue
            
            if not video_results_list:
                print("Error: No valid PKL files found")
                return 0, 0, 0
            
            print(f"\nTotal videos loaded: {len(video_results_list)}")
            print(f"Total window samples: {len(all_stgcn_samples)}")
            
            # 비디오 단위로 데이터 분할
            print("\nSplitting data by video groups...")
            train_segments, val_segments, test_segments = self._split_samples_by_video(
                all_stgcn_samples, video_results_list, train_split, val_split
            )
            
            # 새로운 출력 구조로 저장
            print("Moving files to split directories...")
            self._save_split_data_new_structure(
                train_segments, val_segments, test_segments, 
                output_dir, dataset_name, video_results_list
            )
            
            # 통합 PKL 파일들 저장
            print("Creating unified PKL files...")
            self._save_unified_pkl_files(
                train_segments, val_segments, test_segments,
                output_dir, dataset_name
            )
            
            # 로그 및 출력 파일들을 최종 폴더로 이동
            print("Organizing output files...")
            self._organize_output_files(output_dir, dataset_name)
            
            # temp 폴더 정리
            print("Cleaning up temp folder...")
            self._cleanup_temp_folder(output_dir, dataset_name)
            
            print(f"\nMerge completed:")
            print(f"  Training samples: {len(train_segments):,}")
            print(f"  Validation samples: {len(val_segments):,}")
            print(f"  Test samples: {len(test_segments):,}")
            print(f"  Total windows: {len(train_segments) + len(val_segments) + len(test_segments):,}")
            
            return len(train_segments), len(val_segments), len(test_segments)
            
        except Exception as e:
            print(f"Error in merge mode: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0, 0, 0
    
    def _create_failure_analysis(self, video_path, failure_stage, root_cause, detailed_info=None):
        """실패 분석 결과를 생성하는 헬퍼 메서드"""
        video_name = os.path.basename(video_path)
        
        # 해결 방법 제안
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
            'video_path': video_path,
            'video_name': video_name,
            'failure_stage': failure_stage,
            'root_cause': root_cause,
            'suggested_solution': solution_map.get(failure_stage, 'Review error logs for specific details'),
            'detailed_info': detailed_info or {},
            'error_type': failure_stage,
            'error_message': root_cause,
            'timestamp': self._get_current_timestamp(),
            'full_traceback': None  # 예외가 아닌 경우는 traceback 없음
        }