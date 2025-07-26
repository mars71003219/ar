#!/usr/bin/env python3
"""
Optimized Violence Pipeline Configuration
최적화된 폭력 분류 파이프라인 설정 파일
"""

import os
import os.path as osp

# 기본 경로 설정
BASE_DIR = "/home/gaonpf/hsnam/mmlabs"
MMPOSE_DIR = osp.join(BASE_DIR, "mmpose")
MMACTION_DIR = osp.join(BASE_DIR, "mmaction2")

# 모델 설정
POSE_CONFIG = osp.join(MMPOSE_DIR, "configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py")
POSE_CHECKPOINT = osp.join(MMPOSE_DIR, "checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth")

GCN_CONFIG = osp.join(MMACTION_DIR, "configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_rwf2000_finetune_0.py")
GCN_CHECKPOINT = osp.join(MMACTION_DIR, "work_dirs/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_rwf2000_finetune_0/best_acc_top1_epoch_23.pth")

# 입력 데이터 경로
TEST_VIDEO_DIR = "/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000/test"
TRAIN_VIDEO_DIR = "/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000/train"

# 출력 경로
OUTPUT_DIR = "/workspace/rtmo_gcn_pipeline/optimized_results"

# 파이프라인 설정
PIPELINE_CONFIG = {
    # 하드웨어 설정
    'device': 'cuda:0',
    'max_workers': 4,
    'batch_size': 8,
    
    # 모델 설정
    'sequence_length': 30,
    'pose_score_threshold': 0.3,
    'nms_threshold': 0.65,
    
    # Fight-우선 트래킹 설정 (전체 4분할 + 중앙 방식)
    'region_weights': {
        'center': 1.0,         # 중앙 영역 (가장 중요)
        'top_left': 0.7,       # 좌상단
        'top_right': 0.7,      # 우상단  
        'bottom_left': 0.6,    # 좌하단
        'bottom_right': 0.6    # 우하단
    },
    
    # 복합 점수 가중치
    'composite_weights': {
        'position': 0.3,
        'movement': 0.25,
        'interaction': 0.25,
        'detection': 0.1,
        'consistency': 0.1
    },
    
    # 추론 설정
    'window_overlap': 0.5,  # 50% 오버랩
    'confidence_threshold': 0.5,
    'continuity_threshold': 3,
    
    # 성능 최적화
    'memory_pool_size': 100,
    'cache_enabled': True,
    'parallel_pose_extraction': True,
    'gpu_memory_fraction': 0.8
}

# 검증을 위한 경로 체크 함수
def validate_paths():
    """설정된 경로들의 유효성 검사"""
    paths_to_check = [
        ('Pose Config', POSE_CONFIG),
        ('Pose Checkpoint', POSE_CHECKPOINT),
        ('GCN Config', GCN_CONFIG),
        ('GCN Checkpoint', GCN_CHECKPOINT)
    ]
    
    missing_paths = []
    for name, path in paths_to_check:
        if not osp.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("❌ 다음 파일들을 찾을 수 없습니다:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    
    print("✅ 모든 모델 파일이 존재합니다.")
    return True

# GPU 메모리 체크 함수
def check_gpu_memory():
    """GPU 메모리 상태 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device(PIPELINE_CONFIG['device'])
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            
            print(f"🖥️ GPU 메모리 상태:")
            print(f"   - 총 메모리: {total_memory / 1024**3:.2f} GB")
            print(f"   - 할당된 메모리: {allocated_memory / 1024**3:.2f} GB")
            print(f"   - 예약된 메모리: {reserved_memory / 1024**3:.2f} GB")
            print(f"   - 사용 가능한 메모리: {(total_memory - allocated_memory) / 1024**3:.2f} GB")
            
            return True
        else:
            print("❌ CUDA를 사용할 수 없습니다.")
            return False
    except Exception as e:
        print(f"❌ GPU 메모리 체크 실패: {e}")
        return False

if __name__ == "__main__":
    print("🔧 파이프라인 설정 검증 중...")
    print(f"📁 기본 디렉토리: {BASE_DIR}")
    print(f"🎯 출력 디렉토리: {OUTPUT_DIR}")
    
    # 경로 검증
    validate_paths()
    
    # GPU 메모리 체크
    check_gpu_memory()
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 출력 디렉토리 생성 완료: {OUTPUT_DIR}")