#!/usr/bin/env python3
"""
Enhanced RTMO Pose Extractor - 핵심 포즈 추출 클래스

기존 rtmo_gcn_pipeline의 EnhancedRTMOPoseExtractor를 이전한 버전입니다.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

try:
    from mmpose.apis import inference_bottomup, init_model
    MMPOSE_AVAILABLE = True
except ImportError as e:
    print(f"MMPose import error: {e}")
    MMPOSE_AVAILABLE = False


class EnhancedRTMOPoseExtractor:
    """개선된 RTMO 포즈 추출기 클래스"""
    
    def __init__(self, config_file, checkpoint_file, device='cuda:0', 
                 gpu_ids=None,
                 score_thr=0.3, nms_thr=0.35,
                 track_high_thresh=0.6, track_low_thresh=0.1,
                 track_max_disappeared=30, track_min_hits=3,
                 quality_threshold=0.3, min_track_length=10,
                 weights=None):
        """
        Args:
            config_file: RTMO 설정 파일 경로
            checkpoint_file: RTMO 체크포인트 파일 경로
            device: 추론에 사용할 디바이스
            gpu_ids: 사용할 GPU ID 리스트
            score_thr: 포즈 검출 점수 임계값
            nms_thr: NMS 임계값
            track_high_thresh: ByteTracker 높은 임계값
            track_low_thresh: ByteTracker 낮은 임계값
            track_max_disappeared: 트랙 최대 소실 프레임
            track_min_hits: 트랙 최소 히트 수
            quality_threshold: 품질 임계값
            min_track_length: 최소 트랙 길이
            weights: 복합점수 가중치
        """
        self.config_path = config_file
        self.checkpoint_path = checkpoint_file
        self.device = device
        self.gpu_ids = gpu_ids or []
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        
        # 트래킹 설정
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_max_disappeared = track_max_disappeared
        self.track_min_hits = track_min_hits
        self.quality_threshold = quality_threshold
        self.min_track_length = min_track_length
        self.weights = weights or [0.45, 0.10, 0.30, 0.10, 0.05]
        
        # 모델 초기화
        self.pose_model = None
        if MMPOSE_AVAILABLE:
            print(f"Initializing RTMO model: {config_file}")
            self.pose_model = init_model(config_file, checkpoint_file, device=device)
            self._configure_model()
            print("RTMO model initialized successfully")
        else:
            print("Warning: MMPose not available, pose extraction will not work")
    
    def _configure_model(self):
        """모델 설정 적용"""
        if not self.pose_model:
            return
            
        if hasattr(self.pose_model.cfg, 'model'):
            if hasattr(self.pose_model.cfg.model, 'test_cfg'):
                self.pose_model.cfg.model.test_cfg.score_thr = self.score_thr
                self.pose_model.cfg.model.test_cfg.nms_thr = self.nms_thr
            else:
                self.pose_model.cfg.model.test_cfg = dict(score_thr=self.score_thr, nms_thr=self.nms_thr)
        
        if hasattr(self.pose_model, 'head') and hasattr(self.pose_model.head, 'test_cfg'):
            self.pose_model.head.test_cfg.score_thr = self.score_thr
            self.pose_model.head.test_cfg.nms_thr = self.nms_thr
    
    def extract_poses_only(self, video_path, failure_logger=None):
        """전체 비디오에 대해 포즈 추정만 수행 (트래킹 제외)"""
        if not MMPOSE_AVAILABLE or not self.pose_model:
            print("Error: MMPose not available or model not initialized")
            return None
            
        try:
            print(f"Extracting poses from: {video_path}")
            
            # CUDA 메모리 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if failure_logger:
                    failure_logger.log_failure(video_path, "Cannot open video file")
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            pose_results = []
            
            # 포즈 추정만 수행 (트래킹 없음)
            print(f"Running pose estimation on {total_frames} frames...")
            pbar = tqdm(total=total_frames, desc="Extracting poses")

            while True:
                success, frame = cap.read()
                if not success:
                    break

                # 포즈 추정만 수행
                batch_pose_results = inference_bottomup(self.pose_model, frame)
                pose_result = batch_pose_results[0]
                
                pose_results.append(pose_result)
                pbar.update(1)

            cap.release()
            pbar.close()
            
            if not pose_results:
                return None
            
            print(f"Extracted poses from {len(pose_results)} frames")
            return pose_results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if failure_logger:
                failure_logger.log_failure(video_path, f"Pose extraction error: {str(e)}")
            return None
    
    def extract_single_frame_poses(self, frame):
        """단일 프레임에서 포즈 추정"""
        if not MMPOSE_AVAILABLE or not self.pose_model:
            return None
            
        try:
            batch_pose_results = inference_bottomup(self.pose_model, frame)
            return batch_pose_results[0]
        except Exception as e:
            print(f"Error extracting poses from single frame: {str(e)}")
            return None
    
    def _convert_pose_data_sample(self, pose_data_sample):
        """PoseDataSample 객체를 표준 형식으로 변환"""
        try:
            # MMPose의 PoseDataSample 형식인지 확인
            if hasattr(pose_data_sample, 'pred_instances'):
                instances = pose_data_sample.pred_instances
                converted_poses = []
                
                # 각 인스턴스를 표준 형식으로 변환
                if hasattr(instances, 'keypoints') and hasattr(instances, 'bboxes'):
                    # torch tensor이면 cpu로 이동, numpy array이면 그대로 사용
                    if hasattr(instances.keypoints, 'cpu'):
                        keypoints = instances.keypoints.cpu().numpy()
                    else:
                        keypoints = np.array(instances.keypoints)
                    
                    if hasattr(instances.bboxes, 'cpu'):
                        bboxes = instances.bboxes.cpu().numpy()
                    else:
                        bboxes = np.array(instances.bboxes)
                    
                    # keypoint_scores 처리 (visibility scores)
                    keypoint_scores = None
                    if hasattr(instances, 'keypoint_scores'):
                        if hasattr(instances.keypoint_scores, 'cpu'):
                            keypoint_scores = instances.keypoint_scores.cpu().numpy()
                        else:
                            keypoint_scores = np.array(instances.keypoint_scores)
                    
                    # bbox_scores 처리
                    bbox_scores = None
                    if hasattr(instances, 'bbox_scores'):
                        if hasattr(instances.bbox_scores, 'cpu'):
                            bbox_scores = instances.bbox_scores.cpu().numpy()
                        else:
                            bbox_scores = np.array(instances.bbox_scores)
                    
                    for i in range(len(keypoints)):
                        person_keypoints = keypoints[i]  # 원본 keypoints
                        
                        # keypoints가 (17, 2) 형태인 경우 (17, 3)으로 확장
                        if person_keypoints.shape[-1] == 2:
                            # visibility scores 추가
                            if keypoint_scores is not None and i < len(keypoint_scores):
                                # keypoint_scores가 있으면 사용
                                visibility = keypoint_scores[i]
                                person_keypoints = np.concatenate([
                                    person_keypoints, 
                                    visibility.reshape(-1, 1)
                                ], axis=1)
                            else:
                                # keypoint_scores가 없으면 기본값 1.0 사용
                                visibility = np.ones((person_keypoints.shape[0], 1))
                                person_keypoints = np.concatenate([person_keypoints, visibility], axis=1)
                        
                        # bbox score 결정
                        if bbox_scores is not None and i < len(bbox_scores):
                            bbox_score = float(bbox_scores[i])
                        else:
                            bbox_score = 1.0
                        
                        # bbox에 score 추가 (ByteTracker 형식)
                        bbox_with_score = np.append(bboxes[i], bbox_score)
                        
                        converted_pose = {
                            'keypoints': person_keypoints,  # (17, 3) 형태
                            'bbox': bbox_with_score,        # (5,) 형태 [x1, y1, x2, y2, score]
                            'score': bbox_score
                        }
                        converted_poses.append(converted_pose)
                
                return converted_poses
            else:
                print("Unsupported pose data format")
                return []
                
        except Exception as e:
            print(f"Error converting pose data sample: {str(e)}")
            return []
    
    def _compute_iou_simple(self, box1, box2):
        """간단한 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def cleanup(self):
        """리소스 정리"""
        if self.pose_model:
            del self.pose_model
            self.pose_model = None
        
        # CUDA 메모리 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass