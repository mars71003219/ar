#!/usr/bin/env python3
"""
Enhanced RTMO Pose Extractor - 핵심 포즈 추출 클래스
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

try:
    from mmpose.apis import inference_bottomup, init_model
    from mmpose.registry import VISUALIZERS
except ImportError as e:
    print(f"MMPose import error: {e}")

from .tracker import ByteTracker


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
        
        # 모델을 생성자에서 한 번만 초기화
        print(f"Initializing RTMO model: {config_file}")
        self.pose_model = init_model(config_file, checkpoint_file, device=device)
        
        # 모델 설정 적용
        self._configure_model()
        print("RTMO model initialized successfully")
    
    def _configure_model(self):
        """모델 설정 적용"""
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
        try:
            print(f"Extracting poses from: {video_path}")
            
            # CUDA 메모리 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # 이미 초기화된 모델 사용 (매번 로드하지 않음)
            pose_model = self.pose_model
        
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
                batch_pose_results = inference_bottomup(pose_model, frame)
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
    
    def apply_tracking_to_poses(self, poses_data: list, start_frame: int, end_frame: int, window_idx: int):
        """저장된 포즈 데이터에 트래킹 및 복합점수 적용"""
        try:
            if not poses_data or len(poses_data) == 0:
                return None
            
            print(f"    Debug: Window {window_idx} processing {len(poses_data)} frames")
            
            # ByteTracker 초기화
            tracker = ByteTracker(
                high_thresh=self.track_high_thresh,
                low_thresh=self.track_low_thresh,
                max_disappeared=self.track_max_disappeared,
                min_hits=self.track_min_hits
            )
            
            # 트래킹 적용
            tracked_poses = []
            for frame_idx, pose_result in enumerate(poses_data):
                try:
                    # PoseDataSample 객체를 표준 형식으로 변환
                    converted_pose = self._convert_pose_data_sample(pose_result)
                    
                    # ByteTracker에 전달할 detection 생성
                    detections = np.array([p['bbox'] for p in converted_pose if 'bbox' in p]) if converted_pose else np.empty((0, 5))
                    
                    # 트래커 업데이트 및 활성 트랙 가져오기
                    tracks = tracker.update(detections)
                    
                    if converted_pose and len(converted_pose) > 0:
                        # 트래킹 결과를 포즈 데이터에 매핑
                        tracked_frame = self._map_tracks_to_poses(converted_pose, tracks)
                        tracked_poses.append(tracked_frame)
                    else:
                        # 포즈가 없는 프레임도 빈 리스트로 추가
                        tracked_poses.append([])
                        
                except Exception as e:
                    print(f"      Error in frame {frame_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    tracked_poses.append([])
            
            # 복합점수 계산 및 어노테이션 생성
            annotation = self._generate_simple_annotation(tracked_poses)
            
            if not annotation or 'persons' not in annotation or not annotation['persons']:
                return None
            
            # 윈도우 결과 구성 - 레거시 형식과 동일하게
            window_result = {
                'window_idx': window_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'annotation': annotation,
                'tracking_applied': True,
                'frame_count': len(tracked_poses)
            }
            
            return window_result
            
        except Exception as e:
            print(f"Error applying tracking to poses: {str(e)}")
            return None
    
    def _convert_pose_data_sample(self, pose_data_sample):
        """PoseDataSample 객체를 표준 형식으로 변환 - 레거시 코드와 동일"""
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
    
    def _generate_simple_annotation(self, tracked_poses):
        """간단한 어노테이션 생성 - 레거시 코드와 동일한 로직"""
        try:
            if not tracked_poses:
                return None
            
            # 트랙 ID별로 데이터 수집
            track_data = {}
            
            for frame_idx, frame_poses in enumerate(tracked_poses):
                if frame_poses:
                    for person in frame_poses:
                        if 'track_id' in person:
                            track_id = person['track_id']
                            if track_id not in track_data:
                                track_data[track_id] = []
                            
                            track_data[track_id].append({
                                'frame_idx': frame_idx,
                                'keypoints': person.get('keypoints', []),
                                'bbox': person.get('bbox', []),
                                'score': person.get('score', 0.0)
                            })
            
            if not track_data:
                return None
            
            # 각 트랙에 대해 복합 점수 계산 (간단화)
            persons = {}
            for track_id, frames in track_data.items():
                if len(frames) >= self.min_track_length:
                    # 평균 점수 계산 (안전한 스칼라 변환)
                    scores = []
                    for f in frames:
                        score = f['score']
                        if isinstance(score, np.ndarray):
                            score = float(score.item()) if score.size == 1 else float(score.mean())
                        scores.append(float(score))
                    avg_score = np.mean(scores)
                    
                    # 키포인트 데이터 구성
                    num_frames = len(tracked_poses)
                    keypoints_data = np.zeros((1, num_frames, 17, 2), dtype=np.float32)
                    scores_data = np.zeros((1, num_frames, 17), dtype=np.float32)
                    
                    for frame_data in frames:
                        frame_idx = frame_data['frame_idx']
                        # 키포인트 존재 여부를 안전하게 확인
                        keypoints = frame_data.get('keypoints', [])
                        if self._has_valid_keypoints(keypoints):
                            try:
                                kpts = np.array(keypoints)
                                
                                # 키포인트 형태 확인 및 처리
                                if kpts.ndim == 1 and len(kpts) >= 51:
                                    # 평면화된 경우 (51,) -> (17, 3)으로 변형
                                    kpts = kpts.reshape(-1, 3)
                                elif kpts.ndim == 2:
                                    # (17, 2) 또는 (17, 3) 형태인 경우 - 이미 _convert_pose_data_sample에서 변환됨
                                    if kpts.shape[0] >= 17 and kpts.shape[1] >= 2:
                                        # (17, 2) 또는 (17, 3) 모두 허용
                                        if kpts.shape[1] == 2:
                                            # (17, 2) 형태인 경우 confidence를 1.0으로 추가
                                            confidence = np.ones((kpts.shape[0], 1))
                                            kpts = np.concatenate([kpts, confidence], axis=1)
                                        # (17, 3) 형태면 그대로 사용
                                    else:
                                        print(f"Invalid keypoint shape for track {track_id}, frame {frame_idx}: {kpts.shape}")
                                        continue
                                elif kpts.ndim == 3:
                                    # (1, 17, 3) 형태인 경우 squeeze
                                    kpts = kpts.squeeze(0)
                                else:
                                    print(f"Unsupported keypoint dimension for track {track_id}, frame {frame_idx}: {kpts.ndim}")
                                    continue
                                
                                if self._safe_array_check(kpts, 17):
                                    # 안전한 인덱싱 및 배열 크기 검증
                                    try:
                                        # 배열 크기 체크
                                        kpts_subset = kpts[:17]
                                        if kpts_subset.ndim >= 2 and kpts_subset.shape[1] >= 3:
                                            keypoints_data[0, frame_idx, :17, 0] = kpts_subset[:, 0]
                                            keypoints_data[0, frame_idx, :17, 1] = kpts_subset[:, 1]
                                            scores_data[0, frame_idx, :17] = kpts_subset[:, 2]
                                        else:
                                            print(f"Invalid keypoint subset shape for track {track_id}, frame {frame_idx}: {kpts_subset.shape}")
                                    except (IndexError, ValueError) as idx_err:
                                        print(f"Index error for track {track_id}, frame {frame_idx}: {str(idx_err)}")
                                        continue
                            except Exception as e:
                                print(f"Error processing keypoints for track {track_id}, frame {frame_idx}: {str(e)}")
                                continue
                    
                    persons[track_id] = {
                        'keypoint': keypoints_data,
                        'keypoint_score': scores_data,
                        'num_keypoints': 17,
                        'track_id': track_id,
                        'composite_score': float(avg_score)
                    }
            
            if not persons:
                return None
            
            return {
                'persons': persons,
                'metadata': {
                    'num_persons': len(persons),
                    'frame_count': len(tracked_poses),
                    'quality_threshold': self.quality_threshold
                }
            }
            
        except Exception as e:
            print(f"Error generating annotation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _safe_array_check(self, arr, min_length):
        """배열 크기를 안전하게 체크"""
        try:
            if isinstance(arr, (list, tuple)):
                return len(arr) >= min_length
            elif isinstance(arr, np.ndarray):
                return arr.shape[0] >= min_length
            else:
                return False
        except Exception:
            return False
    
    def _has_valid_keypoints(self, keypoints):
        """키포인트가 유효한지 안전하게 확인"""
        try:
            if keypoints is None:
                return False
            
            if isinstance(keypoints, (list, tuple)):
                return len(keypoints) > 0
            elif isinstance(keypoints, np.ndarray):
                return keypoints.size > 0 and not np.all(keypoints == 0)
            else:
                return bool(keypoints) if keypoints is not None else False
        except Exception:
            return False
    
    def _calculate_simple_movement_score(self, track_data):
        """간단한 움직임 점수 계산"""
        try:
            if len(track_data) < 2:
                return 0.0
            
            movement_sum = 0.0
            valid_movements = 0
            
            sorted_data = sorted(track_data, key=lambda x: x['frame_idx'])
            
            for i in range(1, len(sorted_data)):
                prev_kpts = sorted_data[i-1]['keypoints']
                curr_kpts = sorted_data[i]['keypoints']
                
                if prev_kpts.shape == curr_kpts.shape:
                    # 주요 관절점들의 움직임 계산 (어깨, 엉덩이, 손목, 발목)
                    key_joints = [5, 6, 9, 10, 11, 12, 15, 16]  # COCO 인덱스
                    
                    joint_movements = []
                    for joint_idx in key_joints:
                        if joint_idx < len(prev_kpts) and joint_idx < len(curr_kpts):
                            try:
                                # 키포인트 차원 확인 (2D or 3D)
                                if prev_kpts[joint_idx].shape[0] >= 2 and curr_kpts[joint_idx].shape[0] >= 2:
                                    # 신뢰도 확인 (3차원인 경우만)
                                    if (prev_kpts[joint_idx].shape[0] >= 3 and curr_kpts[joint_idx].shape[0] >= 3):
                                        # 3차원 (x, y, confidence) 형태
                                        if prev_kpts[joint_idx][2] > 0.3 and curr_kpts[joint_idx][2] > 0.3:
                                            movement = np.linalg.norm(curr_kpts[joint_idx][:2] - prev_kpts[joint_idx][:2])
                                            joint_movements.append(movement)
                                    else:
                                        # 2차원 (x, y) 형태 - 신뢰도 확인 없이 바로 계산
                                        movement = np.linalg.norm(curr_kpts[joint_idx][:2] - prev_kpts[joint_idx][:2])
                                        joint_movements.append(movement)
                            except (IndexError, ValueError) as idx_err:
                                # 인덱스 오류나 배열 오류 시 해당 관절점 건너뛰기
                                continue
                    
                    if joint_movements:
                        movement_sum += np.mean(joint_movements)
                        valid_movements += 1
            
            return movement_sum / valid_movements if valid_movements > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating movement score: {str(e)}")
            return 0.0
    
    def _map_tracks_to_poses(self, pose_result: list, tracks: list) -> list:
        """트래킹 결과를 포즈 데이터에 매핑"""
        try:
            if not tracks:
                return pose_result
            
            tracked_poses = []
            
            # 각 트랙에 대해 가장 가까운 포즈 찾기
            for track in tracks:
                try:
                    track_bbox = track.to_bbox()
                    track_center = [(track_bbox[0] + track_bbox[2]) / 2, 
                                   (track_bbox[1] + track_bbox[3]) / 2]
                    
                    best_match = None
                    best_distance = float('inf')
                    
                    for person in pose_result:
                        if 'bbox' in person:
                            person_bbox = person['bbox'][:4]  # [x1, y1, x2, y2]만 사용
                            person_center = [(person_bbox[0] + person_bbox[2]) / 2,
                                           (person_bbox[1] + person_bbox[3]) / 2]
                            
                            # 중심점 간 거리 계산
                            distance = np.sqrt((track_center[0] - person_center[0])**2 + 
                                             (track_center[1] - person_center[1])**2)
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_match = person.copy()
                    
                    if best_match is not None:
                        # 트랙 ID 추가
                        best_match['track_id'] = int(track.track_id) if hasattr(track, 'track_id') else -1
                        # numpy 배열을 안전하게 리스트로 변환
                        if isinstance(track_bbox, np.ndarray):
                            best_match['track_bbox'] = track_bbox.tolist()
                        else:
                            best_match['track_bbox'] = list(track_bbox)
                        tracked_poses.append(best_match)
                        
                except Exception as e:
                    print(f"        Error processing track {track.track_id}: {str(e)}")
                    continue
            
            return tracked_poses
            
        except Exception as e:
            print(f"Error mapping tracks to poses: {str(e)}")
            return pose_result