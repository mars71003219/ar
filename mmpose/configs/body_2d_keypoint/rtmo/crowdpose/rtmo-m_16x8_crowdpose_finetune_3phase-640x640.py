_base_ = ['../../../_base_/default_runtime.py']

# ==============================================================================
# 1. 런타임 및 스케줄러 설정
# - Body7 사전학습 모델 기반 효율적 도메인 적응
# - 총 120 에폭: 일반 학습 80 + 강화 미세조정 40
# ==============================================================================

train_cfg = dict(
    max_epochs=120, 
    val_interval=5,  # 더 자주 검증하여 모니터링 강화
    dynamic_intervals=[(100, 1)]  # 마지막 20에폭은 매 에폭마다 검증
)

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        max_keep_ckpts=3,
    )
)

# 점진적 학습률 감소 전략
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.0015, weight_decay=0.05),  # 중간 수준 학습률
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({
            'neck.encoder': dict(lr_mult=0.1),    # 인코더 낮은 학습률
            'backbone': dict(lr_mult=0.05),       # 백본 매우 낮은 학습률 (사전학습됨)
            'head': dict(lr_mult=1.0)             # 헤드는 정상 학습률
        })
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

# 4단계 학습률 스케줄링
param_scheduler = [
    # Stage 1: 워밍업 (0-3 에폭)
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True
    ),
    # Stage 2: 일반 학습 (3-80 에폭)
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0001,
        begin=3,
        T_max=77,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    # Stage 3: 1차 미세조정 (80-100 에폭)
    dict(
        type='CosineAnnealingLR',
        eta_min=0.00005,
        begin=80,
        T_max=20,
        end=100,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    # Stage 4: 2차 미세조정 (100-120 에폭)
    dict(
        type='CosineAnnealingLR',
        eta_min=0.00002,
        begin=100,
        T_max=20,
        end=120,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

# ==============================================================================
# 2. 데이터 처리 파이프라인
# ==============================================================================

input_size = (640, 640)
metafile = 'configs/_base_/datasets/coco.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

# CrowdPose to COCO 키포인트 매핑
crowdpose_coco = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (11, 16),
]

# Stage 1: 강한 증강 (0-80 에폭)
train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.02,
        rotate_factor=5,
        scale_factor=(0.9, 1.1),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# Stage 2: 중간 증강 (80-100 에폭)
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.05,
        rotate_factor=8,
        scale_factor=(0.9, 1.1),
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    # dict(type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# Stage 3: 약한 증강 (100-120 에폭)
train_pipeline_stage3 = [
    dict(type='LoadImage'),
    dict(type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.02,
        rotate_factor=5,
        scale_factor=(0.95, 1.05),
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='RandomFlip'),
    # dict(type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# 검증 파이프라인
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', 
        input_size=input_size, 
        pad_val=(114, 114, 114)
    ),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale')
    )
]

# ==============================================================================
# 3. 데이터셋 설정
# ==============================================================================

data_mode = 'bottomup'
data_root = 'data/'

# 훈련 데이터셋
dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='pose/CrowdPose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=train_pipeline_stage1,
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_crowdpose
)

# 검증 데이터셋
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# ==============================================================================
# 4. 평가 설정
# ==============================================================================

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator

# ==============================================================================
# 5. 학습 단계별 전환 Hooks
# ==============================================================================

custom_hooks = [
    # 1차 파이프라인 전환 (80 에폭)
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=40,  # 마지막 40에폭 (80-120)
        new_train_pipeline=train_pipeline_stage2,
        priority=48
    ),
    # 2차 파이프라인 전환 (100 에폭)
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=20,  # 마지막 20에폭 (100-120)
        new_train_pipeline=train_pipeline_stage3,
        priority=47
    ),
    # 손실 함수 가중치 단계별 조정
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            # 1차 미세조정 (80 에폭)
            80: {
                'proxy_target_cc': True,
                'overlaps_power': 0.7,
                'loss_cls.loss_weight': 1.3,
                'loss_mle.loss_weight': 2.0,
                'loss_oks.loss_weight': 25.0
            },
            # 2차 미세조정 (100 에폭)
            100: {
                'proxy_target_cc': True,
                'overlaps_power': 1.0,
                'loss_cls.loss_weight': 1.8,
                'loss_mle.loss_weight': 4.0,
                'loss_oks.loss_weight': 18.0
            },
        },
        priority=48
    ),
    # 기타 안정화 Hooks
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49
    ),
]

# ==============================================================================
# 6. 모델 구조 (Body7 사전학습 모델 기반)
# ==============================================================================

widen_factor = 0.75
deepen_factor = 0.67

model = dict(
    type='BottomupPoseEstimator',
    # Body7 사전학습 모델 로드
    init_cfg=dict(
        type='Pretrained',
        checkpoint='checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth',
    ),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1
            ),
        ]
    ),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='HybridEncoder',
        in_channels=[192, 384, 768],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')
            )
        ),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=384,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2
        )
    ),
    head=dict(
        type='RTMOHead',
        num_keypoints=17,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=384,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')
        ),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo=metafile)
        ),
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]
        ),
        dcc_cfg=dict(
            in_channels=384,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(
                s=128,
                expansion_factor=2,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn='SiLU',
                pos_enc='add'
            )
        ),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0
        ),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,
            loss_weight=30.0
        ),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1e-3,
        ),
        loss_bbox_aux=dict(
            type='L1Loss', 
            reduction='sum', 
            loss_weight=1.0
        ),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    )
)

# ==============================================================================
# 7. 학습 요약
# ==============================================================================
"""
학습 전략 요약:
- 총 120 에폭 (Body7 대비 20 에폭 추가)
- 0-80: 강한 증강 + 일반 학습률로 도메인 적응
- 80-100: 중간 증강 + 1차 미세조정 + 손실 함수 조정
- 100-120: 약한 증강 + 2차 미세조정 + 강화된 손실 함수 조정

예상 성능 향상:
- 더 세밀한 키포인트 정확도 (MLE 손실 점진적 증가)
- 더 정확한 사람 검출 (분류 손실 점진적 증가)
- 더 안정적인 수렴 (3단계 증강 전략)
"""