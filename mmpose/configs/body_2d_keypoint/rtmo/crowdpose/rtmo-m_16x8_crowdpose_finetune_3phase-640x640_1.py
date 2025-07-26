_base_ = ['../../../_base_/default_runtime.py']

# ==============================================================================
# 1. 런타임 및 스케줄러 설정
# ==============================================================================
train_cfg = dict(max_epochs=60, val_interval=5)
auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/AP',  # COCO AP 기준으로 최고의 모델 저장
        rule='greater'
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({
            'backbone': dict(lr_mult=0.1),
        })
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        begin=5,
        end=60,
        T_max=55,
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

crowdpose_coco = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (11, 16),
]

train_pipeline = [
    dict(type='KeypointConverter', num_keypoints=17, mapping=crowdpose_coco),
    dict(type='LoadImage', backend_args=None),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.05,
        rotate_factor=10,
        scale_factor=(0.8, 1.2),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
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

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CrowdPoseDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='pose/CrowdPose/annotations/mmpose_crowdpose_trainval.json',
        data_prefix=dict(img='pose/CrowdPose/images/'),
        metainfo=dict(from_file=metafile),
        pipeline=train_pipeline
    )
)

# 검증 데이터로더를 COCO val2017로 변경
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

# 검증 평가기를 COCO val2017 기준으로 변경
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)

test_evaluator = val_evaluator

# ==============================================================================
# 5. Hooks 설정
# ==============================================================================
custom_hooks = [
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
# 6. 모델 구조
# ==============================================================================
widen_factor = 0.75
deepen_factor = 0.67

model = dict(
    type='BottomupPoseEstimator',
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
            loss_weight=1e-2
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