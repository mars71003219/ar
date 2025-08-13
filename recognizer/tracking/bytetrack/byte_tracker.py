"""
ByteTracker 메인 구현

mmtracking의 ByteTrack 알고리즘을 recognizer 시스템에 맞게 구현했습니다.
pose_estimation과 scoring 사이에서 정확한 트래킹 처리를 제공합니다.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..base import BaseTracker
from ...core.data_structures import PersonPose, FramePoses
from .models.track import STrack, TrackState
from .core.kalman_filter import KalmanFilter
from .utils.bbox_utils import convert_bbox_to_z, iou_distance
from .utils.matching import associate_detections_to_trackers, linear_assignment


@dataclass
class ByteTrackerConfig:
    """ByteTracker 설정"""
    track_thresh: float = 0.5          # 트랙 생성 임계값
    high_thresh: float = 0.6           # 높은 신뢰도 임계값
    match_thresh: float = 0.8          # 매칭 임계값
    frame_rate: int = 30               # 프레임 레이트
    track_buffer: int = 30             # 트랙 버퍼 크기
    min_box_area: float = 10           # 최소 박스 면적
    mot20: bool = False                # MOT20 dataset 여부


class ByteTrackerCore:
    """ByteTracker 핵심 알고리즘"""
    
    def __init__(self, config: ByteTrackerConfig):
        self.config = config
        self.kalman_filter = KalmanFilter()
        
        # 트랙 관리
        self.tracked_stracks: List[STrack] = []    # 활성 트랙
        self.lost_stracks: List[STrack] = []       # 잃어버린 트랙
        self.removed_stracks: List[STrack] = []    # 제거된 트랙
        
        # 프레임 카운터
        self.frame_id = 0
        
    def reset(self):
        """트래커 상태 초기화"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_track_count()
        
    def update(self, detections: List[Dict[str, Any]]) -> List[STrack]:
        """
        프레임별 트래킹 업데이트
        
        Args:
            detections: 검출 결과 리스트
                각 검출은 {'bbox': [x1,y1,x2,y2], 'score': float, 'keypoints': np.ndarray} 형태
                
        Returns:
            활성 트랙 리스트
        """
        self.frame_id += 1
        
        # 검출 결과 필터링 및 STrack 생성
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # 신뢰도별 검출 분리
        high_det, low_det = self._separate_detections(detections)
        
        # 기존 트랙 예측
        strack_pool = self._joint_stracks(self.tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        # 1단계: 높은 신뢰도 검출과 활성 트랙 매칭
        dists = iou_distance(strack_pool, high_det)
        matches, u_track, u_detection = associate_detections_to_trackers(
            high_det, strack_pool, self.config.match_thresh
        )
        
        # 매칭된 트랙 업데이트
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_det[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # 2단계: 매칭되지 않은 트랙과 낮은 신뢰도 검출 매칭
        r_tracked_stracks = [strack_pool[i] for i in u_track 
                           if strack_pool[i].state == TrackState.Tracked]
        
        if len(r_tracked_stracks) > 0 and len(low_det) > 0:
            dists = iou_distance(r_tracked_stracks, low_det)
            matches, u_track_remain, u_detection_second = associate_detections_to_trackers(
                low_det, r_tracked_stracks, 0.5
            )
            
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = low_det[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
        
        # 매칭되지 않은 트랙 처리
        for it in u_track:
            track = strack_pool[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # 새로운 트랙 생성
        detections_new = [high_det[i] for i in u_detection]
        if len(detections_new) > 0:
            new_stracks = self._init_track(detections_new, self.frame_id)
            activated_stracks.extend(new_stracks)
        
        # 트랙 제거 조건 확인
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame() > self.config.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        # 트랙 리스트 업데이트
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # 중복 제거
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        return [track for track in self.tracked_stracks if track.is_activated]
    
    def _separate_detections(self, detections: List[Dict[str, Any]]) -> tuple:
        """검출을 신뢰도별로 분리"""
        high_det = []
        low_det = []
        
        for det in detections:
            # 박스 크기 필터링
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.config.min_box_area:
                continue
                
            # STrack 객체 생성
            strack = STrack(
                bbox=bbox,
                score=det['score'],
                keypoints=det.get('keypoints', None)
            )
            
            if det['score'] >= self.config.track_thresh:
                high_det.append(strack)
            else:
                low_det.append(strack)
                
        return high_det, low_det
    
    def _init_track(self, detections: List[STrack], frame_id: int) -> List[STrack]:
        """새로운 트랙 초기화"""
        new_stracks = []
        for det in detections:
            if det.score >= self.config.high_thresh:
                det.activate(self.kalman_filter, frame_id)
                new_stracks.append(det)
        return new_stracks
    
    @staticmethod
    def _joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        """두 트랙 리스트 합치기"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res
    
    @staticmethod
    def _sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
        """첫 번째 리스트에서 두 번째 리스트 제거"""
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
    
    @staticmethod
    def _remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack]) -> tuple:
        """중복 트랙 제거"""
        pdist = iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        
        for p, q in zip(pairs[0], pairs[1]):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        
        return resa, resb


class ByteTrackerWrapper(BaseTracker):
    """recognizer 시스템과 연동하는 ByteTracker 래퍼"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: ByteTracker 설정 딕셔너리
        """
        if config is None:
            config = {}
            
        self.config = ByteTrackerConfig(**config)
        self.tracker = ByteTrackerCore(self.config)
        
    def track_frame_poses(self, frame_poses: FramePoses) -> FramePoses:
        """
        pose_estimation → tracking → scoring 파이프라인에서 트래킹 처리
        
        Args:
            frame_poses: 포즈 추정 결과
            
        Returns:
            트래킹이 적용된 포즈 데이터
        """
        if not frame_poses.poses:
            return frame_poses
        
        # PersonPose를 검출 형태로 변환
        detections = []
        for pose in frame_poses.poses:
            detection = {
                'bbox': pose.bbox,
                'score': pose.confidence,
                'keypoints': pose.keypoints
            }
            detections.append(detection)
        
        # 트래킹 수행
        active_tracks = self.tracker.update(detections)
        
        # 결과를 FramePoses 형태로 변환
        tracked_poses = []
        for track in active_tracks:
            # 트래킹된 정보로 PersonPose 업데이트
            tracked_pose = PersonPose(
                person_id=track.track_id,
                bbox=track.tlbr.tolist(),
                keypoints=track.keypoints,
                confidence=track.score
            )
            tracked_poses.append(tracked_pose)
        
        # 새로운 FramePoses 생성
        tracked_frame_poses = FramePoses(
            frame_idx=frame_poses.frame_idx,
            poses=tracked_poses,
            timestamp=frame_poses.timestamp,
            metadata={
                **frame_poses.metadata,
                'tracking_info': {
                    'total_tracks': len(active_tracks),
                    'frame_id': self.tracker.frame_id
                }
            }
        )
        
        return tracked_frame_poses
    
    def reset(self):
        """트래커 상태 초기화"""
        self.tracker.reset()
        
    def get_track_info(self) -> Dict[str, Any]:
        """트래킹 정보 반환"""
        return {
            'frame_id': self.tracker.frame_id,
            'tracked_count': len(self.tracker.tracked_stracks),
            'lost_count': len(self.tracker.lost_stracks),
            'removed_count': len(self.tracker.removed_stracks),
            'total_tracks': STrack.track_id_count
        }