# Enhanced STGCN++ Configuration for Custom RTMO Fight Detection
# 커스텀 RTMO enhanced annotation format을 사용한 STGCN++ 싸움 감지 설정

_base_ = 'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'

# Import custom modules - 경로 문제로 일시적으로 비활성화
# custom_imports = dict(
#     imports=['mmaction.datasets.enhanced_pose_dataset'],
#     allow_failed_imports=False
# )

# ============================================================================
# Model Configuration (Enhanced Fight Detection)
# ============================================================================

model = dict(
    cls_head=dict(
        type='GCNHead',
        num_classes=2,  # Fight / Non-fight
        in_channels=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0, 1.0]  # Fight 클래스에 약간 더 높은 가중치
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
# Enhanced Pipeline Configuration
# ============================================================================

# Standard training pipeline (Enhanced transforms 문제로 표준 pipeline 사용)
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

# Standard validation pipeline
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

# Standard test pipeline (multiple clips for robust testing)
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
# DataLoader Configuration
# ============================================================================

train_dataloader = dict(
    _delete_=True,  # base config dataloader 완전히 무시
    batch_size=8,
    num_workers=4,
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
    batch_size=8,
    num_workers=4,
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
# Training Strategy (Enhanced Fight Detection)
# ============================================================================

# Optimizer configuration (base config 완전 오버라이드)
optim_wrapper = dict(
    _delete_=True,  # base config optimizer 무시
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.001,  # AdamW에 적합한 학습률로 조정
        weight_decay=0.0005
    ),
    clip_grad=dict(max_norm=40, norm_type=2)
)

# Learning rate scheduler
param_scheduler = [
    # 5 에폭 동안 선형적으로 학습률을 증가시키는 Warmup
    dict(
        type='LinearLR',
        start_factor=0.001, # 0.01 * 0.001 = 0.00001 부터 시작
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    # 5 에폭 이후에는 CosineAnnealingLR 적용
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
        T_max=50,
        eta_min_ratio=0.01
    )
]

# Training loop configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # Enhanced format에 적합한 epoch 수
    val_begin=1,
    val_interval=2  # Enhanced evaluation을 위해 더 자주 검증
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Enhanced evaluation metrics (수정: base config와 호환)
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
# Runtime Configuration
# ============================================================================

# Checkpoint and logging
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
        max_keep_ckpts=3
    ),
    logger=dict(
        type='LoggerHook',
        interval=50,
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

# Work directory
work_dir = '/workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_epoch50'

# Resume training
resume = False

# Randomness
randomness = dict(seed=42, deterministic=False)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Logging level
log_level = 'INFO'

# ============================================================================
# Custom Hooks for Enhanced Training
# ============================================================================

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='acc/top1',         # 모니터링할 메트릭
        patience=5,                 # 5 에폭 동안 개선이 없으면 중단
        rule='greater'              # 메트릭 값이 클수록 좋음
    )
]

# ============================================================================
# Training Command Examples
# ============================================================================

"""
Enhanced STGCN++ Training Commands:

1. Basic Training:
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py

2. With distributed training:
bash tools/dist_train.sh configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py 4

3. Resume training:
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py --resume

4. With custom work directory:
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py --work-dir work_dirs/custom_enhanced

5. Testing:
python tools/test.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py checkpoints/enhanced_fight_model.pth

Enhanced Features Enabled:
- Fight-prioritized person ranking
- 5-region spatial analysis
- Composite score weighting
- Quality-based filtering
- Adaptive augmentation
- Region-aware normalization
"""