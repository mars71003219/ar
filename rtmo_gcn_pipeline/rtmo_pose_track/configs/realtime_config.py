#!/usr/bin/env python3
"""
Real-time CCTV Fight Detection Configuration
실시간 CCTV 폭력 탐지 시스템 설정
"""

import os
from typing import Dict, Any, List, Optional


class RealtimeConfig:
    """실시간 CCTV 탐지 시스템 설정 클래스"""
    
    def __init__(self):
        # 기본 경로 설정
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # === 소스 설정 ===
        self.source_config = {
            # RTSP 스트림 예시: 'rtsp://admin:password@192.168.1.100:554/stream'
            # 웹캠: 0, 1, 2, ...
            # 비디오 파일: '/path/to/video.mp4'
            'source': 0,
            
            # 스트림 처리 설정
            'stream_config': {
                'buffer_size': 30,           # 프레임 버퍼 크기
                'reconnect_attempts': 5,     # 재연결 시도 횟수
                'timeout_seconds': 10,       # 타임아웃 (초)
                'target_fps': 15,            # 목표 FPS
                'frame_skip': 2              # 프레임 스킵 (성능 최적화)
            }
        }
        
        # === 모델 설정 ===
        self.model_config = {
            # RTMO 포즈 추정 모델
            'detector_config': os.path.join(
                self.base_dir, 
                '../../../mmpose/projects/rtmo/rtmo/body_2d_keypoint/'
                'rtmo_m_16xb16-600e_body8-halpe26-256x192.py'
            ),
            'detector_checkpoint': os.path.join(
                self.base_dir,
                '../../../checkpoints/rtmo-m_16xb16-600e_body8-halpe26-256x192-2abe5ca1_20231219.pth'
            ),
            
            # STGCN++ 행동 분류 모델
            'action_config': os.path.join(
                self.base_dir,
                '../../../mmaction2/configs/skeleton/stgcnpp/'
                'stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py'
            ),
            'action_checkpoint': os.path.join(
                self.base_dir,
                '../../../checkpoints/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.pth'
            ),
            
            # 디바이스 설정
            'device': 'cuda:0'  # 'cuda:0', 'cuda:1', 'cpu'
        }
        
        # === 추론 설정 ===
        self.inference_config = {
            # 윈도우 설정
            'clip_len': 100,                    # 윈도우 크기 (프레임)
            'inference_stride': 50,             # 추론 간격 (프레임)
            'max_persons': 4,                   # 최대 인원 수
            
            # 분류 설정
            'classification_threshold': 0.5,    # Fight/NonFight 임계값
            
            # 포즈 추정 설정
            'score_thr': 0.3,                  # 포즈 신뢰도 임계값
            'nms_thr': 0.3,                    # NMS 임계값
        }
        
        # === 실시간 의사결정 설정 ===
        self.decision_config = {
            # 기존 비디오 처리 조건의 실시간 적응
            'consecutive_threshold': 3,         # 연속 Fight 윈도우 임계값
            'fight_ratio_threshold': 0.4,       # Fight 비율 임계값 (슬라이딩 윈도우 기반)
            
            # 실시간 특화 설정
            'sliding_window_size': 20,          # 비율 계산용 슬라이딩 윈도우 크기
            'cooldown_period': 30,              # 연속 알림 방지 쿨다운 (초)
            'confidence_decay': 0.1,            # 쿨다운 중 신뢰도 감쇄율
            'temporal_weight': 0.8,             # 시간적 가중치
            
            # 적응적 임계값
            'enable_adaptive_threshold': True,   # 적응적 임계값 활성화
            'threshold_adjustment_factor': 0.05, # 임계값 조정 인수
            
            # 시간대별 가중치
            'night_time_weight': 1.2,          # 야간 시간대 가중치 (22:00-06:00)
            'day_time_weight': 1.0,            # 주간 시간대 가중치
            
            # 알람 해제 조건
            'min_recovery_time': 15,            # 최소 회복 시간 (초)
            'stability_threshold': 0.8,         # 안정성 임계값
            'normal_streak_required': 5,        # 연속 정상 윈도우 필요 수
        }
        
        # === 추적 설정 ===
        self.tracking_config = {
            'track_high_thresh': 0.6,          # 높은 신뢰도 임계값
            'track_low_thresh': 0.1,           # 낮은 신뢰도 임계값
            'track_max_disappeared': 30,       # 최대 사라진 프레임 수
            'track_min_hits': 3,               # 최소 탐지 횟수
            
            # 품질 제어
            'quality_threshold': 0.3,
            'min_track_length': 10,
        }
        
        # === 점수 계산 설정 ===
        self.scoring_config = {
            # 가중치
            'movement_weight': 0.3,
            'position_weight': 0.2,
            'posture_weight': 0.2,
            'interaction_weight': 0.15,
            'temporal_weight': 0.1,
            'appearance_weight': 0.05,
            
            # 임계값
            'high_movement_threshold': 50.0,
            'center_region_ratio': 0.6,
            'interaction_distance_threshold': 100.0,
        }
        
        # === 이벤트 처리 설정 ===
        self.event_config = {
            # 알림 설정
            'enable_console_alerts': True,
            'enable_file_logging': True,
            'enable_webhook_alerts': False,
            
            # 로그 설정
            'log_file_path': 'realtime_events.log',
            'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
            
            # 웹훅 설정 (선택사항)
            'webhook_url': None,
            'webhook_timeout': 5,
            
            # 이벤트 필터링
            'min_confidence_for_alert': 0.7,
            'suppress_normal_events': True,
        }
        
        # === 성능 설정 ===
        self.performance_config = {
            # 메모리 관리
            'max_frame_buffer': 1000,
            'cleanup_interval': 300,  # 초
            
            # 처리 최적화
            'enable_multithreading': True,
            'max_worker_threads': 4,
            
            # 모니터링
            'stats_update_interval': 10,  # 초
            'enable_profiling': False,
        }
        
        # === 출력 설정 ===
        self.output_config = {
            # 저장 경로
            'output_dir': os.path.join(self.base_dir, 'realtime_output'),
            'save_events': True,
            'save_statistics': True,
            
            # 비디오 저장 (선택사항)
            'save_video_segments': False,
            'segment_duration': 30,  # 초
            
            # 이미지 저장 (선택사항)  
            'save_detection_images': False,
            'image_save_interval': 60,  # 초
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """전체 설정 반환"""
        return {
            'source': self.source_config['source'],
            'stream_config': self.source_config['stream_config'],
            **self.model_config,
            **self.inference_config,
            'track_config': self.tracking_config,
            'scorer_config': self.scoring_config,
            'decision_config': self.decision_config,
            'event_config': self.event_config,
            'performance_config': self.performance_config,
            'output_config': self.output_config,
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """설정 업데이트"""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                # 최상위 설정으로 추가
                setattr(self, key, value)
    
    def get_rtsp_config(self, rtsp_url: str, **kwargs) -> Dict[str, Any]:
        """RTSP 전용 설정 생성"""
        config = self.get_full_config()
        config['source'] = rtsp_url
        
        # RTSP 최적화 설정
        config['stream_config'].update({
            'reconnect_attempts': 10,
            'timeout_seconds': 15,
            'buffer_size': 10,  # RTSP는 작은 버퍼 사용
        })
        
        # 추가 설정 적용
        if kwargs:
            config.update(kwargs)
        
        return config
    
    def get_webcam_config(self, camera_index: int = 0, **kwargs) -> Dict[str, Any]:
        """웹캠 전용 설정 생성"""
        config = self.get_full_config()
        config['source'] = camera_index
        
        # 웹캠 최적화 설정
        config['stream_config'].update({
            'reconnect_attempts': 3,
            'timeout_seconds': 5,
            'target_fps': 30,
            'frame_skip': 1,
        })
        
        # 추가 설정 적용
        if kwargs:
            config.update(kwargs)
        
        return config
    
    def get_video_file_config(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """비디오 파일 전용 설정 생성"""
        config = self.get_full_config()
        config['source'] = video_path
        
        # 비디오 파일 최적화 설정
        config['stream_config'].update({
            'reconnect_attempts': 1,
            'timeout_seconds': 30,
            'frame_skip': 0,  # 파일은 모든 프레임 처리
        })
        
        # 추가 설정 적용
        if kwargs:
            config.update(kwargs)
        
        return config
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """설정 유효성 검증"""
        if config is None:
            config = self.get_full_config()
        
        errors = []
        
        # 필수 모델 파일 확인
        if not os.path.exists(config.get('detector_config', '')):
            errors.append("RTMO detector config file not found")
        
        if not os.path.exists(config.get('detector_checkpoint', '')):
            errors.append("RTMO detector checkpoint file not found")
        
        if not os.path.exists(config.get('action_config', '')):
            errors.append("STGCN++ action config file not found")
        
        if not os.path.exists(config.get('action_checkpoint', '')):
            errors.append("STGCN++ action checkpoint file not found")
        
        # 소스 유효성 확인
        source = config.get('source')
        if isinstance(source, str):
            if source.startswith('rtsp://'):
                # RTSP URL 형식 간단 검증
                if not ('://' in source and '@' in source):
                    errors.append("Invalid RTSP URL format")
            elif not os.path.exists(source):
                errors.append(f"Video file not found: {source}")
        elif not isinstance(source, int) or source < 0:
            errors.append("Invalid camera index")
        
        # 수치 범위 검증
        if config.get('classification_threshold', 0.5) < 0 or config.get('classification_threshold', 0.5) > 1:
            errors.append("Classification threshold must be between 0 and 1")
        
        if config.get('clip_len', 100) < 10:
            errors.append("Clip length must be at least 10 frames")
        
        if config.get('inference_stride', 50) < 1:
            errors.append("Inference stride must be at least 1")
        
        return errors


# 사전 정의된 설정 프리셋
PRESET_CONFIGS = {
    'high_accuracy': {
        'classification_threshold': 0.7,
        'score_thr': 0.5,
        'clip_len': 120,
        'inference_stride': 30,
        # 의사결정 설정 (보수적)
        'decision_config': {
            'consecutive_threshold': 5,
            'fight_ratio_threshold': 0.5,
            'sliding_window_size': 30,
            'cooldown_period': 45,
            'confidence_decay': 0.05,
            'min_recovery_time': 20,
            'normal_streak_required': 7,
        }
    },
    
    'high_speed': {
        'frame_skip': 3,
        'target_fps': 10,
        'clip_len': 80,
        'inference_stride': 60,
        'max_persons': 3,
        # 의사결정 설정 (빠른 반응)
        'decision_config': {
            'consecutive_threshold': 2,
            'fight_ratio_threshold': 0.3,
            'sliding_window_size': 15,
            'cooldown_period': 20,
            'confidence_decay': 0.15,
            'min_recovery_time': 10,
            'normal_streak_required': 3,
        }
    },
    
    'balanced': {
        'classification_threshold': 0.5,
        'frame_skip': 2,
        'target_fps': 15,
        'clip_len': 100,
        'inference_stride': 50,
        # 의사결정 설정 (균형)
        'decision_config': {
            'consecutive_threshold': 3,
            'fight_ratio_threshold': 0.4,
            'sliding_window_size': 20,
            'cooldown_period': 30,
            'confidence_decay': 0.1,
            'min_recovery_time': 15,
            'normal_streak_required': 5,
        }
    },
    
    'debug': {
        'debug_mode': True,
        'enable_profiling': True,
        'log_level': 'DEBUG',
        'enable_console_alerts': True,
        'save_detection_images': True,
        # 의사결정 설정 (디버그)
        'decision_config': {
            'consecutive_threshold': 2,
            'fight_ratio_threshold': 0.3,
            'sliding_window_size': 10,
            'cooldown_period': 10,
            'enable_adaptive_threshold': False,  # 디버그에서는 고정 임계값
        }
    }
}


def get_config(preset: str = 'balanced', **overrides) -> Dict[str, Any]:
    """
    설정 생성 헬퍼 함수
    
    Args:
        preset: 프리셋 이름 ('high_accuracy', 'high_speed', 'balanced', 'debug')
        **overrides: 추가 설정 오버라이드
    
    Returns:
        완전한 설정 딕셔너리
    """
    config_manager = RealtimeConfig()
    base_config = config_manager.get_full_config()
    
    # 프리셋 적용
    if preset in PRESET_CONFIGS:
        base_config.update(PRESET_CONFIGS[preset])
    
    # 오버라이드 적용
    if overrides:
        base_config.update(overrides)
    
    return base_config


# 사용 예시
if __name__ == "__main__":
    # 기본 설정
    config_manager = RealtimeConfig()
    
    # RTSP 카메라 설정
    rtsp_config = config_manager.get_rtsp_config(
        'rtsp://admin:password@192.168.1.100:554/stream'
    )
    
    # 웹캠 설정  
    webcam_config = config_manager.get_webcam_config(0)
    
    # 고정확도 프리셋 설정
    high_acc_config = get_config('high_accuracy', device='cuda:1')
    
    # 설정 검증
    errors = config_manager.validate_config(rtsp_config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")
    
    print(f"Total config keys: {len(rtsp_config)}")