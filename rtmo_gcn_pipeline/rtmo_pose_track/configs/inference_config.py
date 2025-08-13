#!/usr/bin/env python3
"""
추론 전용 설정
"""

import os
import sys
from pathlib import Path

# 현재 파일의 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 설정 파일에서 사용 가능한 헬퍼 함수들
def get_workspace_path(relative_path: str) -> str:
    """워크스페이스 기준 경로 반환"""
    return f"/workspace/{relative_path.lstrip('/')}"

def get_data_path(relative_path: str) -> str:
    """데이터 디렉토리 기준 경로 반환"""
    return f"/aivanas/raw/surveillance/action/violence/action_recognition/data/{relative_path.lstrip('/')}"

def get_output_path(relative_path: str) -> str:
    """출력 디렉토리 기준 경로 반환"""
    return f"/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/{relative_path.lstrip('/')}"

class InferenceConfig:
    """추론 전용 설정"""
    
    # 모드 설정
    mode = 'inference'
    
    # 경로 설정
    # input_dir = get_data_path('RWF-2001')
    # output_dir = get_output_path('test_visualizer')
    input_dir = get_data_path('UBI_FIGHTS/videos')
    output_dir = get_output_path('UBI_FIGHTS')
    # input_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/raw_videos"
    # output_dir = "/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/hanbit"
    
    # 포즈 추정 관련 설정 (기존 코드 재사용)
    detector_config = get_workspace_path('mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py')
    detector_checkpoint = get_workspace_path('mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth')
    
    # MMAction2 모델 설정
    action_config = get_workspace_path('mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py')
    action_checkpoint = get_workspace_path('mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_stable/best_acc_top1_epoch_14.pth')
    
    # GPU 설정
    gpu = '0,1'
    
    # 포즈 추출 임계값
    score_thr = 0.3
    nms_thr = 0.35
    
    # 트래킹 설정
    track_high_thresh = 0.6
    track_low_thresh = 0.1
    track_max_disappeared = 30
    track_min_hits = 3
    quality_threshold = 0.15
    min_track_length = 10
    
    # 복합 점수 가중치
    movement_weight = 0.40
    position_weight = 0.15
    interaction_weight = 0.35
    temporal_weight = 0.08
    persistence_weight = 0.02
    
    # 윈도우 설정
    clip_len = 100
    inference_stride = 50  # 추론시에는 더 넓은 스트라이드 사용 (학습 시와 다르게 설정 가능)
    training_stride = 10   # 참고용 학습 시 스트라이드
    
    # 연속 이벤트 설정
    consecutive_event_threshold = 3 # 연속 3개 윈도우가 fight여야 최종 fight 판정
    
    # Focus person 설정 (visualizer용)
    focus_person = 4  # 상위 4명에 대해 색상 표시
    
    # 분류 임계값
    classification_threshold = 0.5  # 분류 확률 임계값
    
    # 디버그 설정
    debug_mode = False  # 디버그 로그 비활성화
    debug_single_video = False # 테스트를 위해 단일 비디오만 처리
    
    @classmethod
    def get_weights(cls):
        """복합 점수 가중치를 리스트로 반환"""
        return [
            cls.movement_weight,
            cls.position_weight,
            cls.interaction_weight,
            cls.temporal_weight,
            cls.persistence_weight
        ]
    
    @classmethod
    def validate_config(cls):
        """설정 값 검증"""
        errors = []
        
        # 경로 검증
        if not os.path.exists(cls.detector_config):
            errors.append(f"Detector config file not found: {cls.detector_config}")
        
        if not os.path.exists(cls.detector_checkpoint):
            errors.append(f"Detector checkpoint file not found: {cls.detector_checkpoint}")
            
        if not os.path.exists(cls.action_config):
            errors.append(f"Action config file not found: {cls.action_config}")
        
        if not os.path.exists(cls.action_checkpoint):
            errors.append(f"Action checkpoint file not found: {cls.action_checkpoint}")
        
        # 가중치 합계 검증
        weights_sum = sum(cls.get_weights())
        if abs(weights_sum - 1.0) > 0.01:
            errors.append(f"Weights sum should be 1.0, got {weights_sum}")
        
        # 연속 이벤트 임계값 검증
        if cls.consecutive_event_threshold < 1:
            errors.append(f"consecutive_event_threshold should be >= 1, got {cls.consecutive_event_threshold}")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """현재 설정 출력"""
        print("=" * 70)
        print(" Inference Configuration")
        print("=" * 70)
        print(f"Mode: {cls.mode}")
        print(f"Input Directory: {cls.input_dir}")
        print(f"Output Directory: {cls.output_dir}")
        print()
        
        print("Model Settings:")
        print(f"  Pose Config: {os.path.basename(cls.detector_config)}")
        print(f"  Pose Checkpoint: {os.path.basename(cls.detector_checkpoint)}")
        print(f"  Action Config: {os.path.basename(cls.action_config)}")
        print(f"  Action Checkpoint: {os.path.basename(cls.action_checkpoint)}")
        print(f"  GPU: {cls.gpu}")
        print()
        
        print("Pose Detection Thresholds:")
        print(f"  Score Threshold: {cls.score_thr}")
        print(f"  NMS Threshold: {cls.nms_thr}")
        print()
        
        print("Tracking Settings:")
        print(f"  High Threshold: {cls.track_high_thresh}")
        print(f"  Low Threshold: {cls.track_low_thresh}")
        print(f"  Max Disappeared: {cls.track_max_disappeared}")
        print(f"  Min Hits: {cls.track_min_hits}")
        print(f"  Quality Threshold: {cls.quality_threshold}")
        print(f"  Min Track Length: {cls.min_track_length}")
        print()
        
        print("Window Settings:")
        print(f"  Clip Length: {cls.clip_len} frames")
        print(f"  Inference Stride: {cls.inference_stride} (used for inference)")
        print(f"  Training Stride: {cls.training_stride} (reference only)")
        print(f"  Window Overlap: {max(0, cls.clip_len - cls.inference_stride)} frames")
        print()
        
        print("Inference Settings:")
        print(f"  Consecutive Event Threshold: {cls.consecutive_event_threshold}")
        print(f"  Classification Threshold: {cls.classification_threshold}")
        print(f"  Focus Person Count: {cls.focus_person}")
        print(f"  Debug Mode: {cls.debug_mode}")
        print()
        
        print("Composite Score Weights:")
        print(f"  Movement: {cls.movement_weight}")
        print(f"  Position: {cls.position_weight}")
        print(f"  Interaction: {cls.interaction_weight}")
        print(f"  Temporal: {cls.temporal_weight}")
        print(f"  Persistence: {cls.persistence_weight}")
        print(f"  Total: {sum(cls.get_weights()):.3f}")
        print()
        
        print("Output Structure:")
        print(f"  {cls.output_dir}/")
        print("  ├── windows/")
        print("  │   ├── Fight/")
        print("  │   └── NonFight/")
        print("  ├── results/")
        print("  │   ├── window_results.json")
        print("  │   ├── video_results.json")
        print("  │   └── performance_metrics.json")
        print("  └── visualizations/")
        print("      ├── Fight/")
        print("      └── NonFight/")
        print("=" * 70)