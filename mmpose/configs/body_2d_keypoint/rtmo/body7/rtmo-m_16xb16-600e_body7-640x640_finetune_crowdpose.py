_base_ = ['../../../_base_/default_runtime.py']  # 기본 런타임 설정을 불러옵니다.

# ========================
# 러닝타임 및 학습 설정 (CrowdPose 미세조정용)
# ========================

# CrowdPose 미세조정을 위한 학습 설정 (임시: 80에폭부터 custom_hook 적용)
train_cfg = dict(max_epochs=100, val_interval=10, dynamic_intervals=[(80, 1)])  # 임시: 80에폭부터 custom_hook 적용

auto_scale_lr = dict(base_batch_size=256)  # 자동 러닝레이트 스케일링 기준 배치사이즈

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3))  # 20에폭마다 체크포인트 저장, 최대 3개 보관

# CrowdPose 미세조정을 위한 옵티마이저 설정 (더 낮은 학습률 사용)
optim_wrapper = dict(
    type='OptimWrapper',  # 옵티마이저 래퍼 타입
    constructor='ForceDefaultOptimWrapperConstructor',  # 강제 기본 옵티마이저 래퍼 생성자 사용
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),  # CrowdPose 미세조정: 더 낮은 학습률 0.0001 사용
    paramwise_cfg=dict(
        norm_decay_mult=0,  # 노멀라이즈 파라미터는 weight_decay 적용 안함
        bias_decay_mult=0,  # bias 파라미터도 weight_decay 적용 안함
        bypass_duplicate=True,  # 중복 파라미터 무시
        force_default_settings=True,  # 기본 설정 강제 적용
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),  # neck.encoder 파라미터는 러닝레이트 0.05배
    clip_grad=dict(max_norm=0.1, norm_type=2))  # 그래디언트 클리핑

# CrowdPose 미세조정을 위한 학습률 스케줄러 (더 짧은 워밍업, 단순한 스케줄)
param_scheduler = [
    dict(
        type='LinearLR',  # CrowdPose 미세조정: 선형 워밍업 사용
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',  # 코사인 러닝레이트 스케줄러
        eta_min=0.00001,  # CrowdPose 미세조정: 더 낮은 최소 학습률
        begin=5,
        T_max=95,  # CrowdPose 미세조정: 95에폭 동안 코사인 감소
        end=100,
        by_epoch=True,
        convert_to_iter_based=True),
]

# ========================
# 데이터셋 매핑 (COCO 17개 키포인트 유지)
# ========================

crowdpose_coco = [  # CrowdPose → COCO 포맷 매핑 (14개 키포인트 완전 매핑)
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (11, 16),
]


# ========================
# 데이터 및 파이프라인 설정 (CrowdPose 미세조정용, COCO 17개 키포인트 유지)
# ========================

input_size = (640, 640)  # 입력 이미지 크기
# CrowdPose 미세조정이지만 COCO 17개 키포인트 유지
metafile = 'configs/_base_/datasets/coco.py'  # COCO 메타 정보 파일 유지 (17개 키포인트)
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)  # 어노테이션 인코더


# CrowdPose 미세조정을 위한 단순화된 학습 파이프라인
train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),  # 이미지 로드
    dict(
        type='BottomupRandomAffine',  # 어파인 변환 (모자이크 제거하여 단순화)
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),  # 색상 증강
    dict(type='RandomFlip'),  # 좌우 반전
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),  # 어노테이션 필터링
    dict(type='GenerateTarget', encoder=codec),  # 타겟 생성
    dict(type='PackPoseInputs'),  # 입력 패킹
]

# CrowdPose 미세조정을 위한 단순화된 2단계 파이프라인
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(
        type='KeypointConverter', 
        num_keypoints=17,  # COCO: 17개 키포인트로 변환
        mapping=crowdpose_coco),  # CrowdPose → COCO 매핑
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# 데이터 모드 및 루트
data_mode = 'bottomup'
data_root = 'data/'


# ========================
# 학습 데이터셋 정의 (CrowdPose 미세조정용, COCO 17개 키포인트 유지)
# ========================

# CrowdPose 미세조정을 위한 데이터셋 설정 (COCO 17개 키포인트 유지)
dataset_coco = dict(
    type='CocoDataset',  # COCO 데이터셋
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',  # 어노테이션 파일
    data_prefix=dict(img='coco/train2017/'),  # 이미지 폴더
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,  # COCO: 17개 키포인트 유지
            mapping=[(i, i) for i in range(17)])  # COCO는 그대로 사용
    ],
)

# CombinedDataset용 CrowdPose 데이터셋 (pipeline 포함)
dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='pose/CrowdPose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', 
            num_keypoints=17,  # COCO: 17개 키포인트로 변환
            mapping=crowdpose_coco)  # CrowdPose → COCO 매핑
    ],
)

# YOLOXPoseModeSwitchHook용 CrowdPose 데이터셋 (pipeline 없음)
dataset_crowdpose_stage2 = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='pose/CrowdPose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=train_pipeline_stage2
)

# CrowdPose 미세조정을 위한 데이터셋 조합 (COCO + CrowdPose)
train_dataset = dict(
    type='CombinedDataset',  # 여러 데이터셋 조합
    metainfo=dict(from_file=metafile),  # COCO 메타 정보 (17개 키포인트)
    datasets=[
        dataset_coco,        # COCO: 기본 포즈 다양성 제공
        dataset_crowdpose,   # CrowdPose: 혼잡한 장면
    ],
    sample_ratio_factor=[0.3, 1.0],  # COCO: 30%, CrowdPose: 100%
    test_mode=False,
    pipeline=train_pipeline_stage1)

# 학습 데이터로더 설정
train_dataloader = dict(
    batch_size=16,  # 배치 사이즈
    num_workers=8,  # 데이터 로딩 워커 수
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # 셔플
    dataset=train_dataset
    )

# ========================
# 검증/테스트 데이터셋 및 파이프라인 (COCO 17개 키포인트 유지)
# ========================

# val datasets
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'input_size', 'input_center', 'input_scale'))
]

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

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox',
    nms_mode='none',
)
test_evaluator = val_evaluator

# ========================
# 커스텀 후크(Hook) 설정 (CrowdPose 미세조정용, COCO 17개 키포인트 유지)
# ========================

custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',  # 임시: 마지막 99에폭 동안 파이프라인/데이터셋 변경 (2에폭부터 적용)
        num_last_epochs=20, # 임시: 80에폭부터 적용
        new_train_dataset=dataset_crowdpose_stage2,  # CrowdPose 데이터셋으로 변경 (pipeline 없음)
        # new_train_pipeline=train_pipeline_stage2,    # 단순화된 파이프라인 (KeypointConverter 포함)
        priority=48),
    dict(
        type='RTMOModeSwitchHook',  # 50에폭에서 loss 등 하이퍼파라미터 변경 (CrowdPose 미세조정용)
        epoch_attributes={
            50: {  # CrowdPose 미세조정: 50에폭에서 변경
                'proxy_target_cc': True,
                'overlaps_power': 1.0,
                'loss_cls.loss_weight': 1.5,  # CrowdPose 미세조정: 더 보수적인 가중치
                'loss_mle.loss_weight': 3.0,  # CrowdPose 미세조정: 더 보수적인 가중치
                'loss_oks.loss_weight': 8.0   # CrowdPose 미세조정: 더 보수적인 가중치
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),  # SyncBN 동기화
    dict(
        type='EMAHook',  # EMA(지수이동평균) 적용
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

# ========================
# 모델 설정 (CrowdPose 미세조정용, COCO 17개 키포인트 유지)
# ========================

widen_factor = 0.75  # 백본 채널 확장 비율
deepen_factor = 0.67  # 백본 깊이 확장 비율

model = dict(
    type='BottomupPoseEstimator',  # 바텀업 포즈 추정기
    init_cfg=dict(
        type='Kaiming',  # Kaiming 초기화
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',  # 데이터 전처리
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',  # 배치 단위 랜덤 리사이즈
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1),
        ]),
    backbone=dict(
        type='CSPDarknet',  # CSPDarknet 백본
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/yolox_m_8x8_300e_coco_20230829.pth',  # COCO 사전훈련 모델 사용
            prefix='backbone.',
        )),
    neck=dict(
        type='HybridEncoder',  # 하이브리드 인코더
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
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=384,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2)),
    head=dict(
        type='RTMOHead',  # RTMO 헤드
        num_keypoints=17,  # COCO: 17개 키포인트 유지
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
            act_cfg=dict(type='Swish')),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo=metafile)),  # COCO 메타 정보 사용
        prior_generator=dict(
            type='MlvlPointGenerator',
            centralize_points=True,
            strides=[16, 32]),
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
                pos_enc='add')),
        overlaps_power=0.5,
        loss_cls=dict(
            type='VariFocalLoss',
            reduction='sum',
            use_target_weight=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo=metafile,  # COCO 메타 정보 사용
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1e-2,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))


visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])