#!/usr/bin/env python3
"""
기본 설정 파일 - 폭력 감지 데이터 처리 파이프라인
모든 설정 값을 여기서 관리합니다.
"""

import os

class DefaultConfig:
    """기본 설정 클래스"""
    
    # 모드 설정
    mode = 'full'  # ['full', 'merge']
    
    # 경로 설정
    input_dir = '/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000'
    output_dir = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output'
    
    # Resume 설정
    resume = False
    
    # 포즈 추출 관련 설정
    detector_config = '/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py'
    detector_checkpoint = '/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'
    
    # GPU 설정
    gpu = '0,1'  # GPU ID들 (쉼표로 구분) 또는 'cpu'
    
    # 처리 파라미터
    clip_len = 100  # 세그먼트 클립 길이 (프레임)
    num_person = 4  # 오버레이에 표시할 최대 인원수 (모든 인물은 저장됨)
    training_stride = 10  # 밀집 학습 세그먼트용 스트라이드
    inference_stride = 50  # 희소 추론 세그먼트용 스트라이드
    max_workers = 32  # 최대 병렬 워커 수
    
    # 오버레이 설정
    save_overlay = True  # 포즈 오버레이 비디오 저장 여부
    overlay_fps = 30  # 오버레이 비디오 FPS
    
    # 포즈 추출 임계값 설정
    score_thr = 0.3  # 포즈 검출 점수 임계값
    nms_thr = 0.35  # NMS 임계값
    quality_threshold = 0.3  # 트랙 품질 최소 임계값
    min_track_length = 10  # 유효한 트랙의 최소 길이
    
    # ByteTracker 설정
    track_high_thresh = 0.6  # ByteTracker 높은 임계값
    track_low_thresh = 0.1  # ByteTracker 낮은 임계값
    track_max_disappeared = 30  # 트랙이 삭제되기 전 최대 소실 프레임
    track_min_hits = 3  # 유효한 트랙으로 간주되는 최소 히트 수
    
    # 윈도우 처리 설정
    min_success_rate = 0.1  # 윈도우 처리에 필요한 최소 성공률 (0.1 = 10%)
    
    # 복합 점수 가중치 설정
    movement_weight = 0.45  # 움직임 점수 가중치
    position_weight = 0.10  # 위치 점수 가중치
    interaction_weight = 0.30  # 상호작용 점수 가중치
    temporal_weight = 0.10  # 시간적 일관성 가중치
    persistence_weight = 0.05  # 지속성 점수 가중치
    
    # 데이터 분할
    train_split = 0.7  # 학습 데이터 비율
    val_split = 0.2  # 검증 데이터 비율
    
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
        
        # 가중치 합계 검증
        weights_sum = sum(cls.get_weights())
        if abs(weights_sum - 1.0) > 0.01:
            errors.append(f"Weights sum should be 1.0, got {weights_sum}")
        
        # 분할 비율 검증
        if cls.train_split + cls.val_split >= 1.0:
            errors.append(f"train_split + val_split should be < 1.0, got {cls.train_split + cls.val_split}")
        
        # 값 범위 검증
        if not (0.0 <= cls.score_thr <= 1.0):
            errors.append(f"score_thr should be in [0.0, 1.0], got {cls.score_thr}")
        
        if not (0.0 <= cls.nms_thr <= 1.0):
            errors.append(f"nms_thr should be in [0.0, 1.0], got {cls.nms_thr}")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """현재 설정 출력"""
        print("=" * 70)
        print(" Configuration Settings")
        print("=" * 70)
        print(f"Mode: {cls.mode}")
        print(f"Input Directory: {cls.input_dir}")
        print(f"Output Directory: {cls.output_dir}")
        print(f"Resume: {cls.resume}")
        print()
        
        print("Model Settings:")
        print(f"  Config: {os.path.basename(cls.detector_config)}")
        print(f"  Checkpoint: {os.path.basename(cls.detector_checkpoint)}")
        print(f"  GPU: {cls.gpu}")
        print()
        
        print("Processing Parameters:")
        print(f"  Clip Length: {cls.clip_len} frames")
        print(f"  Training Stride: {cls.training_stride}")
        print(f"  Inference Stride: {cls.inference_stride}")
        print(f"  Max Workers: {cls.max_workers}")
        print(f"  Save Overlay: {cls.save_overlay}")
        print()
        
        print("Pose Detection Thresholds:")
        print(f"  Score Threshold: {cls.score_thr}")
        print(f"  NMS Threshold: {cls.nms_thr}")
        print(f"  Quality Threshold: {cls.quality_threshold}")
        print(f"  Min Track Length: {cls.min_track_length}")
        print()
        
        print("ByteTracker Settings:")
        print(f"  High Threshold: {cls.track_high_thresh}")
        print(f"  Low Threshold: {cls.track_low_thresh}")
        print(f"  Max Disappeared: {cls.track_max_disappeared}")
        print(f"  Min Hits: {cls.track_min_hits}")
        print()
        
        print("Composite Score Weights:")
        print(f"  Movement: {cls.movement_weight}")
        print(f"  Position: {cls.position_weight}")
        print(f"  Interaction: {cls.interaction_weight}")
        print(f"  Temporal: {cls.temporal_weight}")
        print(f"  Persistence: {cls.persistence_weight}")
        print(f"  Total: {sum(cls.get_weights()):.3f}")
        print()
        
        print("Data Split:")
        print(f"  Train: {cls.train_split}")
        print(f"  Val: {cls.val_split}")
        print(f"  Test: {1.0 - cls.train_split - cls.val_split:.1f}")
        print("=" * 70)