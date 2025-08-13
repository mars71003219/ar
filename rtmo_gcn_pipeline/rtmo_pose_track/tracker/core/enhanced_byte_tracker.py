#!/usr/bin/env python3
"""
Enhanced ByteTracker implementation
MMTracking의 ByteTracker를 기반으로 성능 개선된 버전
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.track import Track, TrackState
from utils.matching import associate_detections_to_trackers, weighted_iou_association
from utils.bbox_utils import compute_ious


class EnhancedByteTracker:
    """
    향상된 ByteTracker 구현
    
    주요 개선사항:
    1. MMTracking의 설정값 활용
    2. 더 정교한 상태 관리
    3. 향상된 매칭 알고리즘
    4. 성능 최적화
    """
    
    def __init__(self, 
                 obj_score_thrs: Dict[str, float] = None,
                 init_track_thr: float = 0.7,
                 weight_iou_with_det_scores: bool = True,
                 match_iou_thrs: Dict[str, float] = None,
                 num_tentatives: int = 3,
                 num_frames_retain: int = 30):
        """
        Args:
            obj_score_thrs: {'high': 0.6, 'low': 0.1} - detection 점수 임계값
            init_track_thr: 새 트랙 초기화 임계값
            weight_iou_with_det_scores: IoU에 detection 점수 가중치 적용 여부
            match_iou_thrs: {'high': 0.1, 'low': 0.5, 'tentative': 0.3} - 매칭 IoU 임계값들
            num_tentatives: 트랙 확정을 위한 연속 프레임 수
            num_frames_retain: 트랙 유지 최대 프레임 수
        """
        # MMTracking ByteTracker 설정값 사용
        self.obj_score_thrs = obj_score_thrs or {'high': 0.6, 'low': 0.1}
        self.init_track_thr = init_track_thr
        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs or {'high': 0.1, 'low': 0.5, 'tentative': 0.3}
        self.num_tentatives = num_tentatives
        self.num_frames_retain = num_frames_retain
        
        # 트랙 관리
        self.tracked_tracks: List[Track] = []    # 확정된 활성 트랙들
        self.lost_tracks: List[Track] = []       # 일시적으로 잃어버린 트랙들  
        self.removed_tracks: List[Track] = []    # 완전히 제거된 트랙들
        
        # 프레임 정보
        self.frame_id = 0
        
        # 통계
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'new_tracks': 0,
            'lost_tracks': 0,
            'removed_tracks': 0
        }
    
    def update(self, detections: np.ndarray, img_shape: Optional[Tuple] = None) -> List[Track]:
        """
        ByteTracker 메인 업데이트 로직
        
        Args:
            detections: shape (N, 5) - [x1, y1, x2, y2, score]
            img_shape: 이미지 크기 (H, W)
            
        Returns:
            활성 트랙들의 리스트
        """
        self.frame_id += 1
        
        # 1단계: 모든 트랙에 대해 예측 수행
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()
        
        # 입력 검증
        if len(detections) == 0:
            # 검출이 없을 때 트랙 상태만 관리
            self._manage_tracks()
            return self._get_output_tracks()
        
        # 2단계: detection을 high/low 점수로 분리
        high_det_inds = detections[:, 4] >= self.obj_score_thrs['high']
        low_det_inds = ((detections[:, 4] >= self.obj_score_thrs['low']) & 
                       (detections[:, 4] < self.obj_score_thrs['high']))
        
        high_dets = detections[high_det_inds]
        low_dets = detections[low_det_inds]
        
        # 3단계: High confidence detection과 confirmed tracks 1차 매칭
        confirmed_tracks = [t for t in self.tracked_tracks if t.is_activated]
        
        if len(confirmed_tracks) > 0 and len(high_dets) > 0:
            # 트랙들의 예측 bbox 추출
            track_bboxes = np.array([t.to_xyxy() for t in confirmed_tracks])
            
            if self.weight_iou_with_det_scores:
                matches, u_detection, u_track = weighted_iou_association(
                    high_dets[:, :4], track_bboxes, high_dets[:, 4],
                    iou_threshold=self.match_iou_thrs['high'])
            else:
                matches, u_detection, u_track = associate_detections_to_trackers(
                    high_dets[:, :4], track_bboxes,
                    iou_threshold=self.match_iou_thrs['high'])
        else:
            matches = np.empty((0, 2), dtype=int)
            u_detection = np.arange(len(high_dets))
            u_track = np.arange(len(confirmed_tracks))
        
        # 매칭된 트랙들 업데이트
        for m in matches:
            track = confirmed_tracks[m[1]]
            track.update(high_dets[m[0], :4], high_dets[m[0], 4])
            track.frame_id = self.frame_id
        
        # 4단계: 매칭되지 않은 high detections와 unconfirmed tracks 매칭
        unconfirmed_tracks = [t for t in self.tracked_tracks if not t.is_activated]
        
        if len(unconfirmed_tracks) > 0 and len(u_detection) > 0:
            unconfirmed_bboxes = np.array([t.to_xyxy() for t in unconfirmed_tracks])
            unmatched_high_dets = high_dets[u_detection]
            
            matches2, u_detection2, u_unconfirmed = associate_detections_to_trackers(
                unmatched_high_dets[:, :4], unconfirmed_bboxes,
                iou_threshold=self.match_iou_thrs['tentative'])
            
            # 매칭된 unconfirmed 트랙들 업데이트
            for m in matches2:
                track = unconfirmed_tracks[m[1]]
                track.update(unmatched_high_dets[m[0], :4], unmatched_high_dets[m[0], 4])
                track.frame_id = self.frame_id
            
            # 남은 unmatched detections 업데이트
            u_detection = u_detection[u_detection2]
        
        # 5단계: 매칭되지 않은 confirmed tracks 중 최근 lost된 것들과 low detections 매칭
        lost_tracks_for_low = []
        for i in u_track:
            track = confirmed_tracks[i]
            if track.time_since_update == 1:  # 방금 lost된 트랙만
                track.mark_lost()
                lost_tracks_for_low.append(track)
            else:
                # 이미 오래된 lost 트랙은 lost_tracks로 이동
                self.tracked_tracks.remove(track)
                track.mark_lost()
                self.lost_tracks.append(track)
        
        # Low confidence detection과 매칭
        if len(lost_tracks_for_low) > 0 and len(low_dets) > 0:
            lost_bboxes = np.array([t.to_xyxy() for t in lost_tracks_for_low])
            
            matches3, u_low_detection, u_lost = associate_detections_to_trackers(
                low_dets[:, :4], lost_bboxes,
                iou_threshold=self.match_iou_thrs['low'])
            
            # 매칭된 lost 트랙들 재활성화
            for m in matches3:
                track = lost_tracks_for_low[m[1]]
                track.re_activate(low_dets[m[0], :4], low_dets[m[0], 4], self.frame_id)
            
            # 매칭되지 않은 lost 트랙들을 lost_tracks로 이동
            for i in u_lost:
                track = lost_tracks_for_low[i]
                self.tracked_tracks.remove(track)
                self.lost_tracks.append(track)
        else:
            # Low detection 매칭이 없으면 모든 lost 트랙을 lost_tracks로 이동
            for track in lost_tracks_for_low:
                self.tracked_tracks.remove(track)
                self.lost_tracks.append(track)
        
        # 6단계: 매칭되지 않은 high confidence detections로 새 트랙 생성
        for i in u_detection:
            det = high_dets[i]
            if det[4] >= self.init_track_thr:
                new_track = Track(det[:4], det[4])
                new_track.frame_id = self.frame_id
                self.tracked_tracks.append(new_track)
                self.stats['new_tracks'] += 1
                self.stats['total_tracks'] += 1
        
        # 7단계: 트랙 상태 관리 및 정리
        self._manage_tracks()
        
        # 8단계: 통계 업데이트
        self._update_stats()
        
        return self._get_output_tracks()
    
    def _manage_tracks(self):
        """트랙 상태 관리 및 정리"""
        # lost_tracks에서 너무 오래된 트랙들 제거
        tracks_to_remove = []
        for track in self.lost_tracks:
            if track.time_since_update > self.num_frames_retain:
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.lost_tracks.remove(track)
            track.mark_removed()
            self.removed_tracks.append(track)
        
        # tracked_tracks에서 비활성화된 트랙들 정리
        tracks_to_remove = []
        for track in self.tracked_tracks:
            if (track.state == TrackState.LOST and 
                track.time_since_update > self.num_frames_retain):
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.tracked_tracks.remove(track)
            track.mark_removed()
            self.removed_tracks.append(track)
    
    def _get_output_tracks(self) -> List[Track]:
        """출력할 트랙들 반환 (활성화된 트랙들만)"""
        output_tracks = []
        
        for track in self.tracked_tracks:
            # 충분한 hits를 가지거나 최근에 업데이트된 트랙만 출력
            if (track.is_activated or 
                (track.hits >= self.num_tentatives and track.time_since_update < 1)):
                output_tracks.append(track)
        
        return output_tracks
    
    def _update_stats(self):
        """통계 정보 업데이트"""
        self.stats['active_tracks'] = len([t for t in self.tracked_tracks if t.is_activated])
        self.stats['lost_tracks'] = len(self.lost_tracks)
        self.stats['removed_tracks'] = len(self.removed_tracks)
    
    def get_stats(self) -> Dict:
        """현재 통계 정보 반환"""
        return self.stats.copy()
    
    def reset(self):
        """트래커 상태 리셋"""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        
        self.frame_id = 0
        Track.reset_count()
        
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'new_tracks': 0,
            'lost_tracks': 0,
            'removed_tracks': 0
        }
    
    def __len__(self):
        """전체 트랙 수"""
        return len(self.tracked_tracks) + len(self.lost_tracks)
    
    def __str__(self):
        return (f"EnhancedByteTracker(frame={self.frame_id}, "
                f"tracked={len(self.tracked_tracks)}, "
                f"lost={len(self.lost_tracks)}, "
                f"removed={len(self.removed_tracks)})")
    
    def __repr__(self):
        return self.__str__()