"""
STrack 클래스 - ByteTracker의 핵심 트랙 객체

mmtracking의 ByteTrack 구현을 참고하여 pose tracking에 최적화했습니다.
"""

import numpy as np
from typing import List, Optional, Tuple
from ..core.kalman_filter import KalmanFilter
from ..utils.bbox_utils import convert_bbox_to_z, convert_x_to_bbox


class TrackState:
    """트랙 상태"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single Track 클래스 - ByteTracker의 핵심 트랙 객체"""
    
    shared_kalman = KalmanFilter()
    track_id_count = 0

    def __init__(self, bbox: List[float], score: float, keypoints: Optional[np.ndarray] = None):
        """
        Args:
            bbox: [x1, y1, x2, y2] 형태의 바운딩 박스
            score: 검출 신뢰도
            keypoints: [17, 3] 형태의 키포인트 (optional)
        """
        # 트랙 정보
        self.track_id = None
        self.is_activated = False
        
        # 검출 정보
        self.score = score
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.keypoints = keypoints if keypoints is not None else np.zeros((17, 3))
        
        # 칼만 필터 상태
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        
        # 트래킹 상태
        self.state = TrackState.New
        self.frame_id = 0
        self.tracklet_len = 0
        self.start_frame = 0
        
        # 히스토리
        self.alpha = 0.9  # 피처 스무딩 파라미터
        self.smooth_feat = None
        self.curr_feat = None
        
        # 추가 메타데이터  
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
    def predict(self):
        """칼만 필터를 사용한 상태 예측"""
        if self.kalman_filter is not None:
            if (self.state != TrackState.Tracked) and (self.state != TrackState.Lost):
                return
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
    
    @staticmethod
    def multi_predict(stracks: List['STrack']):
        """여러 트랙에 대한 일괄 예측"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """트랙 활성화"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # 바운딩 박스를 칼만 필터 상태로 변환
        self.mean, self.covariance = self.kalman_filter.initiate(convert_bbox_to_z(self.bbox))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """트랙 재활성화"""
        # 칼만 필터 업데이트
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, convert_bbox_to_z(new_track.bbox)
        )
        
        # 정보 업데이트
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.keypoints = new_track.keypoints
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.hits += 1
        self.time_since_update = 0
        
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: 'STrack', frame_id: int):
        """트랙 업데이트"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.hits += 1
        self.time_since_update = 0

        # 칼만 필터 업데이트
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, convert_bbox_to_z(new_track.bbox)
        )
        
        # 정보 업데이트
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.keypoints = new_track.keypoints
        self.state = TrackState.Tracked
        self.is_activated = True

    def mark_lost(self):
        """트랙을 잃어버림으로 표시"""
        self.state = TrackState.Lost
        self.time_since_update += 1

    def mark_removed(self):
        """트랙을 제거됨으로 표시"""
        self.state = TrackState.Removed

    @staticmethod
    def next_id():
        """다음 트랙 ID 생성"""
        STrack.track_id_count += 1
        return STrack.track_id_count

    def end_frame(self):
        """트랙의 마지막 프레임"""
        return self.frame_id

    @staticmethod
    def reset_track_count():
        """트랙 카운터 리셋"""
        STrack.track_id_count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """바운딩 박스를 [x, y, w, h] 형태로 반환"""
        if self.mean is None:
            return convert_bbox_to_z(self.bbox)[:4]
        
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """바운딩 박스를 [x1, y1, x2, y2] 형태로 반환"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyah(self) -> np.ndarray:
        """바운딩 박스를 [center_x, center_y, aspect_ratio, height] 형태로 반환"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_dict(self) -> dict:
        """트랙 정보를 딕셔너리로 변환"""
        return {
            'track_id': self.track_id,
            'bbox': self.tlbr.tolist(),
            'score': self.score,
            'keypoints': self.keypoints.tolist() if isinstance(self.keypoints, np.ndarray) else self.keypoints,
            'state': self.state,
            'frame_id': self.frame_id,
            'tracklet_len': self.tracklet_len,
            'hits': self.hits,
            'age': self.age,
            'time_since_update': self.time_since_update
        }

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame()})"