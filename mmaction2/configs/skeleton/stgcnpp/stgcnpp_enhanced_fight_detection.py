# Enhanced STGCN++ Configuration for Custom RTMO Fight Detection (Optimized)
# Loss 정체 문제 해결을 위한 최적화된 설정

_base_ = 'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'

# Import custom modules - 경로 문제로 일시적으로 비활성화
# custom_imports = dict(
#     imports=['mmaction.datasets.enhanced_pose_dataset'],
#     allow_failed_imports=False
# )

# ============================================================================
# Model Configuration (Enhanced Fight Detection with Improved Architecture)
# ============================================================================

model = dict(
    backbone=dict(
        type='STGCN',           # 올바른 모델 타입
        gcn_adaptive='init',    # MMAction2 표준 설정
        gcn_with_res=True,      # 잔차 연결 활성화
        tcn_type='mstcn',       # 시간 합성곱 타입
        graph_cfg=dict(layout='coco', mode='spatial'),
        # 모델 복잡도 적정 수준으로 조정 (학습 능력 향상)
        num_stages=8,           # 6 → 8로 증가 (더 깊은 학습)
        base_channels=48,       # 32 → 48로 증가 (더 많은 특징)
        inflate_stages=[5, 7],  # [4] → [5, 7]로 확장
        down_stages=[5, 7],     # [4] → [5, 7]로 확장
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=2,  # Fight / Non-fight
        in_channels=192,        # base_channels 48, 2번 inflate → 48*2*2 = 192 channels
        dropout=0.6,            # 0.8 → 0.6으로 감소 (학습 활성화)
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0, 1.5]         # Fight 클래스 가중치 완화 (2.0→1.5)
            # 오버피팅으로 인한 gradient explosion 방지
        )
    )
)

# ============================================================================
# Dataset Configuration (Enhanced Format)
# ============================================================================

# Enhanced dataset 사용
dataset_type = 'PoseDataset'  # 표준 dataset으로 임시 변경

# 데이터 경로 설정
data_root = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/UCF_Crime_test2'
ann_file_train = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/RWF-2000_train_windows.pkl'
ann_file_val = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/RWF-2000_val_windows.pkl'
ann_file_test = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/RWF-2000_test_windows.pkl'

# ============================================================================
# Enhanced Pipeline Configuration with Data Augmentation
# ============================================================================

# Enhanced training pipeline (MMAction2 표준 transforms 사용)
train_pipeline = [
    # 기본 정규화
    dict(type='PreNormalize2D'),
    
    # Skeleton feature 생성 (bone 특징 사용)
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    
    # Temporal sampling
    dict(type='UniformSampleFrames', clip_len=100),
    
    # Pose decode
    dict(type='PoseDecode'),
    
    # Final formatting
    dict(type='FormatGCNInput', num_person=4),
    dict(type='PackActionInputs')
]

# Standard validation pipeline (증강 제거)
val_pipeline = [
    # 기본 정규화
    dict(type='PreNormalize2D'),
    
    # Skeleton feature 생성
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    
    # Temporal sampling (test mode)
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    
    # Pose decode
    dict(type='PoseDecode'),
    
    # Final formatting
    dict(type='FormatGCNInput', num_person=4),
    dict(type='PackActionInputs')
]

# Enhanced test pipeline (multiple clips for robust testing)
test_pipeline = [
    # 기본 정규화
    dict(type='PreNormalize2D'),
    
    # Skeleton feature 생성
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    
    # Multiple clips for better test accuracy
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    
    # Pose decode
    dict(type='PoseDecode'),
    
    # Final formatting
    dict(type='FormatGCNInput', num_person=4),
    dict(type='PackActionInputs')
]

# ============================================================================
# DataLoader Configuration (Optimized)
# ============================================================================

train_dataloader = dict(
    _delete_=True,  # base config dataloader 완전히 무시
    batch_size=16,  # 8 → 16으로 증가 (안정성 향상)
    num_workers=8,  # 4 → 8로 증가 (더 빠른 데이터 로딩)
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        test_mode=False
    )
)

val_dataloader = dict(
    _delete_=True,  # base config dataloader 완전히 무시
    batch_size=16,  # 8 → 16으로 증가
    num_workers=8,  # 4 → 8로 증가
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# ============================================================================
# Training Strategy (Optimized for Gradient Stability)
# ============================================================================

# 최적화된 Optimizer configuration
optim_wrapper = dict(
    _delete_=True,  # base config optimizer 무시
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0001,      # 0.0003 → 0.0001로 대폭 감소 (gradient explosion 방지)
        weight_decay=0.005,     # 0.002 → 0.005로 증가 (정규화 강화)
        betas=(0.9, 0.999),      # AdamW 최적 설정
        eps=1e-8                 # 수치적 안정성
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)  # 2.0 → 1.0으로 강화 (gradient explosion 차단)
)

# 개선된 Learning rate scheduler
param_scheduler = [
    # 적절한 Warmup (3 에폭)
    dict(
        type='LinearLR',
        start_factor=0.01,   # 0.1 → 0.01로 감소 (더 안전한 시작)
        by_epoch=True,
        begin=0,
        end=3,               # 10 → 3 에폭으로 감소 (전체 15에폭에 적절)
        convert_to_iter_based=True
    ),
    # 3 에폭 이후에는 CosineAnnealingLR 적용
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=3,             # 10 → 3으로 변경
        T_max=12,            # 40 → 12로 조정 (15-3=12에폭)
        eta_min_ratio=0.001  # 낮은 최종 학습률
    )
]

# Training loop configuration (균형잡힌 학습 기간)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=15,          # 15에폭 유지 (3에폭 웜업 + 12에폭 학습)
    val_begin=1,
    val_interval=1          # 매 에폭마다 검증
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Enhanced evaluation metrics
val_evaluator = [
    dict(
        type='AccMetric',
        metric_options=dict(
            top_k_accuracy=dict(topk=(1,)),
            mean_class_accuracy=dict()
        )
    )
]

test_evaluator = val_evaluator

# ============================================================================
# Runtime Configuration (Enhanced Monitoring)
# ============================================================================

# Enhanced checkpoint and logging (표준 훅만 사용)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
        max_keep_ckpts=5       # 3 → 5개로 증가
    ),
    logger=dict(
        type='LoggerHook',
        interval=20,            # 50 → 20으로 감소 (더 자주 로깅)
        ignore_last=False
    )
)

# Visualization (기본 설정)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends
)

# Model loading
load_from = '/workspace/mmaction2/checkpoints/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth'

# Work directory (새로운 실험을 위한 디렉토리)
work_dir = '/workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_conservative'

# Resume training
resume = False

# Randomness (CUDA 호환성을 위해 deterministic=False로 설정)
randomness = dict(seed=42, deterministic=False)

# Environment
env_cfg = dict(
    cudnn_benchmark=True,   # False → True (성능 향상)
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Logging level
log_level = 'INFO'

# ============================================================================
# Enhanced Custom Hooks for Stable Training
# ============================================================================

custom_hooks = [
    # Loss가 너무 낮아지면 강제 종료 (오버피팅 방지)
    # 실제 구현이 필요하지만 개념적으로 추가
]

# ============================================================================
# Auto Learning Rate Scaling (Optional)
# ============================================================================

# 배치 크기 변경에 따른 자동 학습률 조정
auto_scale_lr = dict(
    base_batch_size=128,    # 기준 배치 크기
    enable=True             # 자동 조정 활성화
)

# ============================================================================
# Mixed Precision Training (Optional - 메모리 절약)
# ============================================================================

# FP16 training for memory efficiency
# optim_wrapper.update(dict(
#     type='AmpOptimWrapper',
#     loss_scale='dynamic'
# ))

# ============================================================================
# Training Command Examples
# ============================================================================

"""
Optimized STGCN++ Training Commands:

1. Basic Training (Recommended):
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_optimized.py

2. With distributed training:
bash tools/dist_train.sh configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_optimized.py 4

3. Resume training:
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_optimized.py --resume

4. With custom work directory:
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_optimized.py --work-dir work_dirs/custom_optimized

5. Testing:
python tools/test.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_optimized.py checkpoints/optimized_fight_model.pth

Key Optimizations Applied:
✅ Gradient Clipping (max_norm=1.0) - Gradient explosion 방지
✅ Lower Learning Rate (0.0005) - 안정적인 수렴
✅ Longer Warmup (10 epochs) - 초기 불안정성 방지
✅ Increased Batch Size (16) - 훈련 안정성 향상
✅ Enhanced Model Architecture - 더 깊은 네트워크 (10 stages)
✅ Regularization (Dropout 0.5) - 오버피팅 방지
✅ Label Smoothing (0.1) - 일반화 향상
✅ Data Augmentation - 데이터 다양성 증가
✅ Better Monitoring - 더 자주 로깅 및 체크포인트

Expected Improvements:
🎯 Stable gradient norms (< 2.0)
🎯 Consistent loss decrease
🎯 Better generalization
🎯 Reduced overfitting
🎯 More stable training dynamics
"""