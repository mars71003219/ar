"""
ByteTracker 메인 구현

mmtracking의 ByteTrack 알고리즘을 recognizer 시스템에 맞게 구현했습니다.
pose_estimation과 scoring 사이에서 정확한 트래킹 처리를 제공합니다.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from ...utils.import_utils import safe_import_pose_structures
except ImportError:
    try:
        from utils.import_utils import safe_import_pose_structures
    except ImportError:
        def safe_import_pose_structures():
            try:
                from utils.data_structure import PersonPose, FramePoses
                return PersonPose, FramePoses
            except ImportError:
                from ...utils.data_structure import PersonPose, FramePoses
                return PersonPose, FramePoses

try:
    from tracking.base import BaseTracker, TrackedObject
except ImportError:
    from ..base import BaseTracker, TrackedObject

PersonPose, FramePoses = safe_import_pose_structures()
from .models.track import STrack, TrackState
from .core.kalman_filter import KalmanFilter
from .utils.bbox_utils import convert_bbox_to_z, iou_distance
from .utils.matching import associate_detections_to_trackers, associate_detections_to_trackers_hybrid, linear_assignment


@dataclass
class ByteTrackerConfig:
    """ByteTracker 설정"""
    track_thresh: float = 0.3          # 트랙 생성 임계값 (낮춤)
    high_thresh: float = 0.5           # 높은 신뢰도 임계값 (낮춤)
    match_thresh: float = 0.7          # 매칭 임계값 (높임)
    frame_rate: int = 30               # 프레임 레이트
    track_buffer: int = 90             # 트랙 버퍼 크기 (늘림)
    min_box_area: float = 100          # 최소 박스 면적 (늘림)
    mot20: bool = False                # MOT20 dataset 여부
    low_thresh: float = 0.1            # 낮은 신뢰도 임계값
    match_thresh_second: float = 0.5   # 2단계 매칭 임계값
    # 하이브리드 매칭 설정 (싸움 동작 대응)
    use_hybrid_matching: bool = True   # 하이브리드 매칭 활성화
    iou_weight: float = 0.6           # IoU 거리 가중치
    keypoint_weight: float = 0.4      # 키포인트 거리 가중치


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
        if self.frame_id <= 5:  # 처음 5프레임만 로깅
            logging.info(f"  high_det={len(high_det)}, low_det={len(low_det)}")
            if len(high_det) > 0:
                scores = [det.score for det in high_det]
                logging.info(f"  high_det scores: {scores}")
        
        # 기존 트랙 예측
        strack_pool = self._joint_stracks(self.tracked_stracks, self.lost_stracks)
        if self.frame_id <= 5:  # 처음 5프레임만 로깅
            logging.info(f"  strack_pool size: {len(strack_pool)} (tracked: {len(self.tracked_stracks)}, lost: {len(self.lost_stracks)})")
        STrack.multi_predict(strack_pool)
        
        # 1단계: 높은 신뢰도 검출과 활성 트랙 매칭 (하이브리드 매칭 적용)
        if self.config.use_hybrid_matching:
            matches, u_detection, u_track = associate_detections_to_trackers_hybrid(
                high_det, strack_pool, self.config.match_thresh,
                use_hybrid=True,
                iou_weight=self.config.iou_weight,
                keypoint_weight=self.config.keypoint_weight
            )
        else:
            dists = iou_distance(strack_pool, high_det)
            matches, u_detection, u_track = associate_detections_to_trackers(
                high_det, strack_pool, self.config.match_thresh
            )
        if self.frame_id <= 5:  # 처음 5프레임만 로깅
            logging.info(f"  association result: matches={len(matches)}, u_track={len(u_track)}, u_detection={len(u_detection)}")
            logging.info(f"  u_detection indices: {u_detection}")
        
        # 매칭된 트랙 업데이트
        for itracked, idet in matches:
            try:
                # 인덱스 범위 확인
                if itracked >= len(strack_pool) or idet >= len(high_det):
                    logging.warning(f"Index out of range: itracked={itracked}, idet={idet}, pool_len={len(strack_pool)}, det_len={len(high_det)}")
                    continue
                    
                track = strack_pool[itracked]
                det = high_det[idet]
                
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            except (IndexError, AttributeError) as e:
                logging.warning(f"Error updating track {itracked}: {str(e)}")
                continue
        
        # 2단계: 매칭되지 않은 트랙과 낮은 신뢰도 검출 매칭
        r_tracked_stracks = []
        for i in u_track:
            try:
                if i < len(strack_pool) and strack_pool[i].state == TrackState.Tracked:
                    r_tracked_stracks.append(strack_pool[i])
            except (IndexError, AttributeError) as e:
                logging.warning(f"Error accessing track {i}: {str(e)}")
                continue
        
        if len(r_tracked_stracks) > 0 and len(low_det) > 0:
            # 2단계도 하이브리드 매칭 적용 (더 관대한 임계값)
            if self.config.use_hybrid_matching:
                matches, u_detection_second, u_track_remain = associate_detections_to_trackers_hybrid(
                    low_det, r_tracked_stracks, self.config.match_thresh_second,
                    use_hybrid=True,
                    iou_weight=self.config.iou_weight,
                    keypoint_weight=self.config.keypoint_weight
                )
            else:
                dists = iou_distance(r_tracked_stracks, low_det)
                matches, u_detection_second, u_track_remain = associate_detections_to_trackers(
                    low_det, r_tracked_stracks, self.config.match_thresh_second
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
            try:
                if it < len(strack_pool):
                    track = strack_pool[it]
                    if track.state != TrackState.Lost:
                        track.mark_lost()
                        lost_stracks.append(track)
                else:
                    logging.warning(f"Track index {it} out of range for pool size {len(strack_pool)}")
            except (IndexError, AttributeError) as e:
                logging.warning(f"Error processing unmatched track {it}: {str(e)}")
                continue
        
        # 새로운 트랙 생성
        detections_new = [high_det[i] for i in u_detection]
        if self.frame_id <= 5:  # 처음 5프레임만 로깅
            logging.info(f"  creating new tracks: u_detection={u_detection}, detections_new={len(detections_new)}")
        if len(detections_new) > 0:
            new_stracks = self._init_track(detections_new, self.frame_id)
            activated_stracks.extend(new_stracks)
            if self.frame_id <= 5:  # 처음 5프레임만 로깅
                logging.info(f"  created {len(new_stracks)} new tracks")
        
        # 적극적인 트랙 제거 조건 (ID 일관성 개선)
        for track in self.lost_stracks[:]:  # 복사본으로 순회
            should_remove = False
            
            # 기본 시간 기반 제거 (더 짧은 버퍼)
            time_since_lost = self.frame_id - track.end_frame()
            if time_since_lost > min(self.config.track_buffer // 2, 30):  # 기존 90 -> 45 또는 30
                should_remove = True
                
            # 품질 기반 제거 (더 엄격하게)
            elif hasattr(track, 'hits') and track.tracklet_len > 5:
                hit_ratio = track.hits / track.tracklet_len
                if hit_ratio < 0.4:  # 40% 미만 히트율
                    should_remove = True
                    
            # 매우 짧은 트랙 제거
            elif track.tracklet_len < 3 and time_since_lost > 10:
                should_remove = True
                
            # 오래된 활성 트랙도 품질 검사
            if not should_remove and track.state == TrackState.Tracked:
                if hasattr(track, 'time_since_update') and track.time_since_update > 20:
                    should_remove = True
                    
            if should_remove:
                track.mark_removed()
                removed_stracks.append(track)
                self.lost_stracks.remove(track)  # 즉시 제거
        
        # 트랙 리스트 업데이트
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # 트랙 수 제한 (ID 폭발 방지)
        max_total_tracks = 20  # 전체 트랙 수 제한
        total_tracks = len(self.tracked_stracks) + len(self.lost_stracks)
        
        if total_tracks > max_total_tracks:
            # 오래된 lost_tracks부터 제거
            self.lost_stracks.sort(key=lambda x: x.end_frame())
            excess_count = total_tracks - max_total_tracks
            
            for i in range(min(excess_count, len(self.lost_stracks))):
                track = self.lost_stracks[i]
                track.mark_removed()
                removed_stracks.append(track)
            
            # 제거된 트랙들을 lost_stracks에서 삭제
            self.lost_stracks = self.lost_stracks[excess_count:]
            self.removed_stracks.extend(removed_stracks[-excess_count:] if excess_count > 0 else [])
        
        # 중복 제거
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        active_tracks = [track for track in self.tracked_stracks if track.is_activated]
        
        # ID 관리 모니터링 (매 10프레임마다 + 처음 5프레임)
        should_log = (self.frame_id <= 5) or (self.frame_id % 10 == 0)
        if should_log:
            logging.info(f"ByteTracker frame {self.frame_id}: input detections={len(detections)}, output tracks={len(active_tracks)}")
            logging.info(f"  high_det={len(high_det) if 'high_det' in locals() else 0}, low_det={len(low_det) if 'low_det' in locals() else 0}")
            logging.info(f"  activated={len(activated_stracks)}, refind={len(refind_stracks)}")
            
            # 트랙 수가 많을 때 경고
            total_tracks = len(self.tracked_stracks) + len(self.lost_stracks)
            if total_tracks > 10:
                logging.warning(f"  HIGH TRACK COUNT: total={total_tracks} (tracked={len(self.tracked_stracks)}, lost={len(self.lost_stracks)})")
                if len(removed_stracks) > 0:
                    logging.info(f"  removed {len(removed_stracks)} tracks this frame")
            
            if len(active_tracks) > 0:
                track_ids = [track.track_id for track in active_tracks]
                logging.info(f"  active track IDs: {track_ids}")
                
        return active_tracks
    
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
        if frame_id <= 5:  # 처음 5프레임만 로깅
            logging.info(f"_init_track: {len(detections)} detections, high_thresh={self.config.high_thresh}")
        for i, det in enumerate(detections):
            if frame_id <= 5:
                logging.info(f"  det {i}: score={det.score}, >= high_thresh? {det.score >= self.config.high_thresh}")
            if det.score >= self.config.high_thresh:
                try:
                    det.activate(self.kalman_filter, frame_id)
                    new_stracks.append(det)
                    if frame_id <= 5:
                        logging.info(f"  det {i}: activated successfully, is_activated={det.is_activated}")
                except Exception as e:
                    if frame_id <= 5:
                        logging.error(f"  det {i}: activation failed: {str(e)}")
        if frame_id <= 5:
            logging.info(f"_init_track result: {len(new_stracks)} new tracks")
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
    
    def __init__(self, config=None):
        """
        Args:
            config: ByteTracker 설정 (TrackingConfig 또는 딕셔너리)
        """
        if config is None:
            config = {}
        
        # TrackingConfig 객체를 ByteTrackerConfig로 변환
        if hasattr(config, '__dict__'):
            # dataclass 객체인 경우
            config_dict = {
                'track_thresh': getattr(config, 'track_thresh', 0.3),
                'high_thresh': getattr(config, 'track_high_thresh', 0.5),
                'match_thresh': getattr(config, 'match_thresh', 0.7),
                'frame_rate': getattr(config, 'frame_rate', 30),
                'track_buffer': getattr(config, 'track_buffer', 90),
                'min_box_area': getattr(config, 'min_box_area', 100),
                'low_thresh': getattr(config, 'track_low_thresh', 0.1),
                'match_thresh_second': getattr(config, 'match_thresh_second', 0.5),
                'mot20': getattr(config, 'mot20', False),
                # 하이브리드 매칭 설정 추가
                'use_hybrid_matching': getattr(config, 'use_hybrid_matching', True),
                'iou_weight': getattr(config, 'iou_weight', 0.6),
                'keypoint_weight': getattr(config, 'keypoint_weight', 0.4)
            }
        else:
            # 딕셔너리인 경우
            config_dict = config
            
        self.config = ByteTrackerConfig(**config_dict)
        self.tracker = ByteTrackerCore(self.config)
    
    def reset_tracker(self):
        """트래커 상태 완전 초기화 - 새 비디오 처리시 호출"""
        self.tracker.reset()
        
    def track_frame_poses(self, frame_poses: FramePoses) -> FramePoses:
        """
        pose_estimation → tracking → scoring 파이프라인에서 트래킹 처리
        
        Args:
            frame_poses: 포즈 추정 결과
            
        Returns:
            트래킹이 적용된 포즈 데이터
        """
        if not frame_poses.persons:
            return frame_poses
        
        # PersonPose를 검출 형태로 변환
        detections = []
        for pose in frame_poses.persons:
            detection = {
                'bbox': pose.bbox,
                'score': pose.score,
                'keypoints': pose.keypoints
            }
            detections.append(detection)
        
        # 트래킹 수행
        active_tracks = self.tracker.update(detections)
        
        # 결과를 FramePoses 형태로 변환
        tracked_poses = []
        for track in active_tracks:
            # 트래킹 품질 점수 계산
            quality_score = self._calculate_track_quality(track)
            
            # 트래킹된 정보로 PersonPose 업데이트
            tracked_pose = PersonPose(
                person_id=track.track_id,
                bbox=track.tlbr.tolist(),
                keypoints=track.keypoints,
                score=track.score
            )
            tracked_pose.track_id = track.track_id
            tracked_pose.bbox_score = track.score
            tracked_pose.composite_score = quality_score
            tracked_poses.append(tracked_pose)
        
        # 새로운 FramePoses 생성
        tracked_frame_poses = FramePoses(
            frame_idx=frame_poses.frame_idx,
            persons=tracked_poses,
            timestamp=frame_poses.timestamp,
            image_shape=frame_poses.image_shape,
            metadata={
                **frame_poses.metadata,
                'tracking_info': {
                    'total_tracks': len(active_tracks),
                    'frame_id': self.tracker.frame_id
                }
            }
        )
        
        return tracked_frame_poses
    
    def _calculate_track_quality(self, track) -> float:
        """트랙 품질 점수 계산"""
        if not hasattr(track, 'hits') or not hasattr(track, 'tracklet_len'):
            return track.score
        
        # 기본 신뢰도
        base_score = track.score
        
        # 트랙 지속성 (hits / tracklet_len)
        if track.tracklet_len > 0:
            persistence_ratio = track.hits / track.tracklet_len
        else:
            persistence_ratio = 1.0
        
        # 트랙 길이 보너스 (길수록 안정적)
        length_bonus = min(1.0, track.tracklet_len / 10.0)  # 10프레임 기준으로 정규화
        
        # 최근 업데이트 페널티 (너무 오래된 트랙은 점수 감소)
        if hasattr(track, 'time_since_update'):
            recency_factor = max(0.1, 1.0 - (track.time_since_update / 5.0))  # 5프레임 기준
        else:
            recency_factor = 1.0
        
        # 복합 점수 계산
        composite_score = (
            base_score * 0.4 +                    # 40% 기본 신뢰도
            persistence_ratio * 0.3 +             # 30% 지속성
            length_bonus * 0.2 +                  # 20% 길이 보너스
            recency_factor * 0.1                  # 10% 최근성
        )
        
        return min(1.0, max(0.0, composite_score))
    
    def initialize_tracker(self) -> bool:
        """트래커 초기화 (BaseTracker 추상메서드 구현)"""
        try:
            self.tracker.reset()
            return True
        except Exception:
            return False
    
    def update(self, detections: List[PersonPose]) -> List[TrackedObject]:
        """트래킹 업데이트 (BaseTracker 추상메서드 구현)"""
        # PersonPose를 딕셔너리 형태로 변환
        detection_dicts = []
        for person in detections:
            detection_dicts.append({
                'bbox': person.bbox,
                'score': person.score,
                'keypoints': person.keypoints
            })
        
        # ByteTracker 업데이트
        active_tracks = self.tracker.update(detection_dicts)
        
        # STrack을 TrackedObject로 변환
        tracked_objects = []
        for track in active_tracks:
            tracked_obj = TrackedObject(
                track_id=track.track_id,
                bbox=track.bbox,
                keypoints=track.keypoints if hasattr(track, 'keypoints') else np.zeros((17, 3)),
                score=track.score,
                age=track.tracklet_len,
                hits=track.hits
            )
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
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