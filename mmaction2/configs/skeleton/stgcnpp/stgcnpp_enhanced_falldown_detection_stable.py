# Enhanced STGCN++ Configuration for Custom RTMO Fight Detection (Stable)

_base_ = 'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'

# ============================================================================
# Model Configuration (Stable Fight Detection)
# ============================================================================

model = dict(
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial'),
        num_stages=4,  # 6 -> 4로 더 단순화 (학습 안정성 향상)
        base_channels=64,  # 48 -> 64로 증가 (표현력 향상)
        inflate_stages=[2],    # 단일 inflate로 단순화
        down_stages=[2],       # 단일 down으로 단순화
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=2,
        in_channels=128, # base_channels 64, 1번 inflate -> 64*2 = 128 channels
        dropout=0.3,     # 0.5 -> 0.3로 감소 (과적합 방지)
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.2, 1.0]  # NonFight:Fight = 45.4:54.6 분포에 맞춰 조정
        )
    )
)

# ============================================================================
# Dataset Configuration
# ============================================================================
dataset_type = 'PoseDataset'
data_root = '/workspace/recognizer/test_data'
ann_file_train = '/workspace/recognizer/output/falldown_aihub/stage3_dataset/unknown_s0.2_n0.65_bytetrack_h0.3_l0.1_t0.2_split0.7-0.2-0.1/train.pkl'
ann_file_val = '/workspace/recognizer/output/falldown_aihub/stage3_dataset/unknown_s0.2_n0.65_bytetrack_h0.3_l0.1_t0.2_split0.7-0.2-0.1/val.pkl'
ann_file_test = '/workspace/recognizer/output/falldown_aihub/stage3_dataset/unknown_s0.2_n0.65_bytetrack_h0.3_l0.1_t0.2_split0.7-0.2-0.1/test.pkl'

# ============================================================================
# Pipeline Configuration
# ============================================================================

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=4),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=4),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=4),
    dict(type='PackActionInputs')
]

# ============================================================================
# DataLoader Configuration
# ============================================================================

train_dataloader = dict(
    _delete_=True,
    batch_size=32,  # 16 -> 32로 증가 (학습 안정성 향상)
    num_workers=8,
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
    _delete_=True,
    batch_size=32,  # 16 -> 32로 증가 (검증 안정성 향상)
    num_workers=8,
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
# Training Strategy (Optimized for Stability)
# ============================================================================

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,   # 0.00001 -> 0.0001로 10배 증가 (학습 가능 수준)
        weight_decay=0.001,  # 0.005 -> 0.001로 감소
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    clip_grad=dict(max_norm=2.0, norm_type=2)  # 1.0 -> 2.0으로 증가
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  # 0.01 -> 0.1로 증가 (더 높은 시작점)
        by_epoch=True,
        begin=0,
        end=3,   # 5 -> 3으로 감소 (짧은 warmup)
        convert_to_iter_based=True
    ),
    dict(
        type='MultiStepLR',  # CosineAnnealingLR -> MultiStepLR로 변경
        by_epoch=True,
        begin=3,
        milestones=[15, 25],  # step decay at epoch 15, 25
        gamma=0.5
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,   # 15 -> 30으로 증가 (충분한 학습 시간)
    val_begin=1,
    val_interval=2   # 1 -> 2로 변경 (2에폭마다 검증)
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# Evaluation Configuration
# ============================================================================

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

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,  # 5 -> 10으로 변경 (체크포인트 간격 증가)
        save_best='auto',
        max_keep_ckpts=5
    ),
    logger=dict(
        type='LoggerHook',
        interval=20,
        ignore_last=False
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends
)

load_from = '/workspace/mmaction2/checkpoints/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth'
work_dir = '/workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-l_falldown_aihub_stable'
resume = False
randomness = dict(seed=42, deterministic=False)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
auto_scale_lr = dict(base_batch_size=128, enable=True)
