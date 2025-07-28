#!/usr/bin/env python3
"""
End-to-End Inference Pipeline Configuration
엔드투엔드 추론 파이프라인 설정
"""

import os
import os.path as osp

# 기본 경로 설정
BASE_DIR = "/workspace"
MMPOSE_DIR = osp.join(BASE_DIR, "mmpose")
MMACTION_DIR = osp.join(BASE_DIR, "mmaction2")

# 학습된 모델 경로
POSE_CONFIG = osp.join(MMPOSE_DIR, "configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py")
POSE_CHECKPOINT = osp.join(MMPOSE_DIR, "checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth")

GCN_CONFIG = osp.join(MMACTION_DIR, "configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py")
GCN_CHECKPOINT = osp.join(MMACTION_DIR, "work_dirs/enhanced_fight_stgcn_v1/best_acc_top1_epoch_24.pth")

# 기본 입력/출력 경로
DEFAULT_INPUT_DIR = "/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000/test"
DEFAULT_OUTPUT_DIR = "/workspace/rtmo_gcn_pipeline/inference_pipeline/inference_results"

# 파이프라인 설정
INFERENCE_CONFIG = {
    # 하드웨어 설정
    'device': 'cuda:0',
    'max_workers': 4,
    'batch_size': 8,
    
    # 모델 설정
    'sequence_length': 30,
    'pose_score_threshold': 0.3,
    'confidence_threshold': 0.5,
    'consecutive_fight_threshold': 3,  # 연속 Fight 예측 임계값
    'num_person': 2,                   # 최대 추적 인원 수
    
    # Fight-우선 트래킹 설정 (5영역 방식)
    'region_weights': {
        'center': 1.0,         # 중앙 영역 (가장 중요)
        'top_left': 0.7,       # 좌상단
        'top_right': 0.7,      # 우상단  
        'bottom_left': 0.6,    # 좌하단
        'bottom_right': 0.6    # 우하단
    },
    
    # 복합 점수 가중치 (Fight 탐지 최적화)
    'composite_weights': {
        'position': 0.20,      # 위치 점수 (20%)
        'movement': 0.35,      # 움직임 점수 (35%) - Fight의 핵심 특징
        'interaction': 0.30,   # 상호작용 점수 (30%) - 인물간 상호작용 중시
        'detection': 0.10,     # 검출 신뢰도 (10%)
        'consistency': 0.05    # 시간적 일관성 (5%)
    },
    
    # 비디오 처리 설정
    'frame_extraction': {
        'fps': None,           # None = 원본 FPS 사용
        'max_frames': None,     # 최대 프레임 수 (30초 * 30fps)
        'resize': (640, 480),  # 표준 해상도
    },
    
    # 오버레이 설정
    'overlay': {
        'enabled': True,
        'joint_color': (0, 255, 0),      # 관절 색상 (녹색)
        'skeleton_color': (255, 0, 0),   # 스켈레톤 색상 (빨간색)
        'text_color': (255, 255, 255),   # 텍스트 색상 (흰색)
        'font_scale': 0.4,               # 폰트 크기 (작게)
        'thickness': 1,
        'point_radius': 1,               # 관절 포인트 반지름
        'show_window_prediction': True   # 윈도우 예측 결과 표시
    },
    
    # 성능 최적화
    'optimization': {
        'memory_pool_size': 100,
        'cache_enabled': True,
        'parallel_processing': True,
        'gpu_memory_fraction': 0.8
    }
}

# COCO 17 키포인트 연결 정의 (시각화용)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# 키포인트 이름 (COCO 17-point)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def validate_config():
    """설정 파일 유효성 검사"""
    missing_files = []
    
    required_files = [
        ('Pose Config', POSE_CONFIG),
        ('Pose Checkpoint', POSE_CHECKPOINT),
        ('GCN Config', GCN_CONFIG),
        ('GCN Checkpoint', GCN_CHECKPOINT)
    ]
    
    for name, path in required_files:
        if not osp.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("필수 파일이 누락되었습니다:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("모든 필수 파일이 존재합니다.")
    return True

def check_gpu_availability():
    """GPU 사용 가능 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device(INFERENCE_CONFIG['device'])
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(device)}")
            return True
        else:
            print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            INFERENCE_CONFIG['device'] = 'cpu'
            return False
    except ImportError:
        print("PyTorch가 설치되지 않았습니다.")
        return False

if __name__ == "__main__":
    print("파이프라인 설정 검증 중...")
    config_valid = validate_config()
    gpu_available = check_gpu_availability()
    
    if config_valid:
        print("설정 검증 완료!")
    else:
        print("설정 검증 실패!")
        exit(1)