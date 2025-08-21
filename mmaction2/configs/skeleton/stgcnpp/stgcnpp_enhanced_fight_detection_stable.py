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
        num_stages=6,  # 8 -> 6으로 감소 (모델 단순화)
        base_channels=48,
        inflate_stages=[3, 5],  # num_stages에 맞춰 조정
        down_stages=[3, 5],    # num_stages에 맞춰 조정
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=2,
        in_channels=192, # base_channels 48, 2번 inflate -> 48*2*2 = 192 channels
        dropout=0.5,     # 0.6 -> 0.5로 감소
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0, 1.5]
        )
    )
)

# ============================================================================
# Dataset Configuration
# ============================================================================
dataset_type = 'PoseDataset'
data_root = '/workspace/recognizer/test_data'
ann_file_train = '/workspace/recognizer/output/annotation/stage3_dataset/train.pkl'
ann_file_val = '/workspace/recognizer/output/annotation/stage3_dataset/val.pkl'
ann_file_test = '/workspace/recognizer/output/annotation/stage3_dataset/test.pkl'
# dataset_type = 'PoseDataset'
# data_root = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/UCF_Crime_test2'
# ann_file_train = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/RWF-2000_train_windows.pkl'
# ann_file_val = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/RWF-2000_val_windows.pkl'
# ann_file_test = '/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000/RWF-2000_test_windows.pkl'

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
    batch_size=16,
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
    batch_size=16,
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
        lr=0.00001,  # 0.0001 -> 0.00001로 대폭 감소
        weight_decay=0.005,
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,  # 3 -> 5 에폭으로 증가 (Warmup 기간 연장)
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
        T_max=10, # 15 - 5 = 10
        eta_min_ratio=0.001
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=15,
    val_begin=1,
    val_interval=1
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
        interval=5,
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
work_dir = '/workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_stable'
resume = False
randomness = dict(seed=42, deterministic=False)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
auto_scale_lr = dict(base_batch_size=128, enable=True)
