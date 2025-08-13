#!/usr/bin/env python3
"""
RTMO + Enhanced ByteTracker Pipeline
RTMO 포즈 추정과 향상된 ByteTracker를 결합한 파이프라인
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from pathlib import Path

try:
    from mmpose.apis import init_model, inference_bottomup
    from mmpose.structures import PoseDataSample
except ImportError as e:
    print(f"MMPose import error: {e}")
    print("Please install MMPose properly")
    raise

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.enhanced_byte_tracker import EnhancedByteTracker
from configs.rtmo_tracker_config import RTMOTrackerConfig
from utils.bbox_utils import clip_bbox


class RTMOTrackingPipeline:
    """
    RTMO 포즈 추정 + Enhanced ByteTracker 통합 파이프라인
    """
    
    def __init__(self, 
                 rtmo_config: str,
                 rtmo_checkpoint: str,
                 device: str = 'cuda:0',
                 tracker_config: Optional[Dict] = None):
        """
        Args:
            rtmo_config: RTMO 설정 파일 경로
            rtmo_checkpoint: RTMO 체크포인트 파일 경로
            device: 추론 디바이스
            tracker_config: 트래커 설정 (None이면 기본 RTMO 설정 사용)
        """
        self.device = device
        
        # RTMO 모델 초기화
        print(f"Loading RTMO model from {rtmo_checkpoint}")
        self.rtmo_model = init_model(rtmo_config, rtmo_checkpoint, device=device)
        
        # Enhanced ByteTracker 초기화
        if tracker_config is None:
            tracker_config = RTMOTrackerConfig.get_config_dict()
        
        self.tracker = EnhancedByteTracker(
            obj_score_thrs=tracker_config['obj_score_thrs'],
            init_track_thr=tracker_config['init_track_thr'],
            weight_iou_with_det_scores=tracker_config['weight_iou_with_det_scores'],
            match_iou_thrs=tracker_config['match_iou_thrs'],
            num_tentatives=tracker_config['num_tentatives'],
            num_frames_retain=tracker_config['num_frames_retain']
        )
        
        self.frame_count = 0
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'average_tracks_per_frame': 0.0
        }
    
    def process_frame(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        단일 프레임 처리
        
        Args:
            image: 입력 이미지 (H, W, 3)
            
        Returns:
            (tracks, pose_result): 트랙 리스트와 포즈 결과
        """
        self.frame_count += 1
        
        # RTMO로 포즈 추정
        pose_results = inference_bottomup(self.rtmo_model, image)
        
        if not pose_results:
            # 검출 결과가 없으면 빈 detection으로 tracker 업데이트
            detections = np.empty((0, 5))
        else:
            # 포즈 결과에서 detection 추출
            detections = self._extract_detections_from_pose_result(pose_results[0])
        
        # 이미지 범위로 bbox 클리핑
        if len(detections) > 0:
            detections[:, :4] = clip_bbox(detections[:, :4], image.shape)
        
        # ByteTracker로 추적
        tracks = self.tracker.update(detections, image.shape)
        
        # 통계 업데이트
        self._update_stats(len(detections), len(tracks))
        
        return tracks, pose_results[0] if pose_results else None
    
    def _extract_detections_from_pose_result(self, pose_result: PoseDataSample) -> np.ndarray:
        """
        포즈 결과에서 detection 형태로 변환
        
        Args:
            pose_result: MMPose PoseDataSample
            
        Returns:
            detections: shape (N, 5) - [x1, y1, x2, y2, score]
        """
        # MMPose 버전 호환성 처리
        if hasattr(pose_result, 'pred_instances'):
            pred_instances = pose_result.pred_instances
        elif hasattr(pose_result, '_pred_instances'):
            pred_instances = pose_result._pred_instances
        else:
            return np.empty((0, 5))
        
        if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
            return np.empty((0, 5))
        
        # 바운딩 박스 추출
        if hasattr(pred_instances.bboxes, 'cpu'):
            bboxes = pred_instances.bboxes.cpu().numpy()
        else:
            bboxes = np.array(pred_instances.bboxes)
        
        # 점수 추출
        if hasattr(pred_instances, 'bbox_scores'):
            if hasattr(pred_instances.bbox_scores, 'cpu'):
                scores = pred_instances.bbox_scores.cpu().numpy()
            else:
                scores = np.array(pred_instances.bbox_scores)
        elif hasattr(pred_instances, 'scores'):
            if hasattr(pred_instances.scores, 'cpu'):
                scores = pred_instances.scores.cpu().numpy()
            else:
                scores = np.array(pred_instances.scores)
        else:
            # 점수가 없으면 1.0으로 설정
            scores = np.ones(len(bboxes))
        
        # detection 형태로 결합
        detections = np.column_stack([bboxes, scores])
        
        return detections
    
    def assign_track_ids_to_pose_result(self, pose_result: PoseDataSample, 
                                       tracks: List) -> PoseDataSample:
        """
        포즈 결과에 트랙 ID 할당
        
        Args:
            pose_result: MMPose PoseDataSample
            tracks: 트랙 리스트
            
        Returns:
            트랙 ID가 할당된 포즈 결과
        """
        if not pose_result or not hasattr(pose_result, 'pred_instances'):
            return pose_result
        
        pred_instances = pose_result.pred_instances
        
        if not hasattr(pred_instances, 'bboxes') or len(pred_instances.bboxes) == 0:
            pred_instances.track_ids = torch.tensor([], dtype=torch.long)
            return pose_result
        
        # 바운딩 박스 추출
        if hasattr(pred_instances.bboxes, 'cpu'):
            pose_bboxes = pred_instances.bboxes.cpu().numpy()
        else:
            pose_bboxes = np.array(pred_instances.bboxes)
        
        # IoU 기반 매칭으로 트랙 ID 할당
        track_ids = self._assign_track_ids_by_iou(pose_bboxes, tracks)
        
        # 트랙 ID를 텐서로 변환하여 할당
        pred_instances.track_ids = torch.tensor(track_ids, dtype=torch.long)
        
        return pose_result
    
    def _assign_track_ids_by_iou(self, pose_bboxes: np.ndarray, 
                                tracks: List, 
                                iou_threshold: float = 0.5) -> List[int]:
        """
        IoU 기반으로 포즈 바운딩 박스에 트랙 ID 할당
        
        Args:
            pose_bboxes: 포즈 바운딩 박스들 shape (N, 4)
            tracks: 트랙 리스트
            iou_threshold: IoU 임계값
            
        Returns:
            트랙 ID 리스트 (매칭 안된 경우 -1)
        """
        from utils.bbox_utils import compute_iou
        
        track_ids = []
        
        for pose_bbox in pose_bboxes:
            best_track_id = -1
            best_iou = 0.0
            
            for track in tracks:
                track_bbox = track.to_xyxy()
                iou = compute_iou(pose_bbox, track_bbox)
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_track_id = track.track_id
            
            track_ids.append(best_track_id)
        
        return track_ids
    
    def _update_stats(self, num_detections: int, num_tracks: int):
        """통계 정보 업데이트"""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += num_detections
        self.stats['total_tracks'] = max(self.stats['total_tracks'], 
                                        self.tracker.stats['total_tracks'])
        
        if self.stats['total_frames'] > 0:
            total_active = sum([len(self.tracker.tracked_tracks) for _ in range(self.stats['total_frames'])])
            self.stats['average_tracks_per_frame'] = total_active / self.stats['total_frames']
    
    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        stats = self.stats.copy()
        stats.update(self.tracker.get_stats())
        return stats
    
    def reset(self):
        """파이프라인 상태 리셋"""
        self.tracker.reset()
        self.frame_count = 0
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'average_tracks_per_frame': 0.0
        }
    
    def __str__(self):
        return (f"RTMOTrackingPipeline(frames={self.frame_count}, "
                f"tracks={len(self.tracker.tracked_tracks)})")
    
    def __repr__(self):
        return self.__str__()