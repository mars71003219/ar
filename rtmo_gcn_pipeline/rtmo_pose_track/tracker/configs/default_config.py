#!/usr/bin/env python3
"""
Default configuration for Enhanced ByteTracker
MMTracking의 기본 설정값을 참고하여 구성
"""


class DefaultTrackerConfig:
    """Enhanced ByteTracker 기본 설정"""
    
    # Detection score thresholds (MMTracking 기본값)
    obj_score_thrs = {
        'high': 0.6,  # 첫 번째 매칭을 위한 높은 임계값
        'low': 0.1    # 두 번째 매칭을 위한 낮은 임계값
    }
    
    # 새 트랙 초기화 임계값 (MMTracking 기본값)
    init_track_thr = 0.7
    
    # IoU에 detection 점수 가중치 적용 여부
    weight_iou_with_det_scores = True
    
    # 매칭을 위한 IoU 임계값들 (MMTracking 기본값)
    match_iou_thrs = {
        'high': 0.1,        # confirmed tracks와 high detection 매칭
        'low': 0.5,         # lost tracks와 low detection 매칭  
        'tentative': 0.3    # unconfirmed tracks와 detection 매칭
    }
    
    # 트랙 확정을 위한 연속 프레임 수
    num_tentatives = 3
    
    # 트랙 유지 최대 프레임 수 (MMTracking 기본값: num_frames_retain)
    num_frames_retain = 30
    
    # Kalman filter 설정
    kalman_config = {
        'center_only': False,  # 전체 바운딩 박스 추적
        'dt': 1.0             # 시간 간격
    }
    
    # 성능 최적화 설정
    performance_config = {
        'enable_parallel': False,       # 병렬 처리 활성화 (추후 구현)
        'cache_predictions': True,      # 예측 결과 캐싱
        'optimize_memory': True         # 메모리 최적화
    }
    
    # 디버깅 및 로깅 설정
    debug_config = {
        'enable_logging': False,
        'log_level': 'INFO',
        'save_intermediate_results': False,
        'visualization': False
    }
    
    @classmethod
    def get_config_dict(cls):
        """설정을 딕셔너리로 반환"""
        return {
            'obj_score_thrs': cls.obj_score_thrs,
            'init_track_thr': cls.init_track_thr,
            'weight_iou_with_det_scores': cls.weight_iou_with_det_scores,
            'match_iou_thrs': cls.match_iou_thrs,
            'num_tentatives': cls.num_tentatives,
            'num_frames_retain': cls.num_frames_retain,
            'kalman_config': cls.kalman_config,
            'performance_config': cls.performance_config,
            'debug_config': cls.debug_config
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """딕셔너리에서 설정 생성"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config